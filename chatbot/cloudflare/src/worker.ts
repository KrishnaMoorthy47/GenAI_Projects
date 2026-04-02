/**
 * Krishna's Personal Assistant — Cloudflare Worker
 * RAG query API backed by OpenAI embeddings, Supabase pgvector, and Groq.
 *
 * POST /api/v1/query
 * Body: { question: string, session_id?: string, collection?: string }
 * Returns: { answer: string, session_id: string, sources: Source[], has_context: boolean }
 */

import OpenAI from "openai";
import { createClient, SupabaseClient } from "@supabase/supabase-js";
import Groq from "groq-sdk";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Env {
  OPENAI_API_KEY: string;
  SUPABASE_URL: string;
  SUPABASE_SERVICE_KEY: string;
  GROQ_API_KEY: string;
  API_KEY: string;
  HISTORY_KV: KVNamespace;
}

interface QueryRequest {
  question: string;
  session_id?: string;
  collection?: string;
}

interface Source {
  source: string;
  page_num: number;
  score: number;
}

interface ChunkRow {
  id: number;
  content: string;
  source: string;
  page_num: number;
  score: number;
}

interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_COLLECTION = "krishna-profile";
const MATCH_COUNT = 5;
const HISTORY_TTL = 3600; // 1 hour in seconds
const MAX_HISTORY_TURNS = 6; // keep last 3 user+assistant pairs

const SYSTEM_PROMPT = `You are a helpful assistant representing Krishna's professional profile. \
Answer the user's question based on the provided context about Krishna. \
If the context doesn't contain enough information, say so honestly. \
Be concise and accurate. Do not fabricate experience or skills not mentioned in the context.`;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function generateSessionId(): string {
  return `sess_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function historyKey(sessionId: string): string {
  return `hist:${sessionId}`;
}

async function loadHistory(kv: KVNamespace, sessionId: string): Promise<HistoryMessage[]> {
  const raw = await kv.get(historyKey(sessionId));
  if (!raw) return [];
  try {
    return JSON.parse(raw) as HistoryMessage[];
  } catch {
    return [];
  }
}

async function saveHistory(
  kv: KVNamespace,
  sessionId: string,
  history: HistoryMessage[]
): Promise<void> {
  // Trim to last MAX_HISTORY_TURNS messages before saving
  const trimmed = history.slice(-MAX_HISTORY_TURNS);
  await kv.put(historyKey(sessionId), JSON.stringify(trimmed), {
    expirationTtl: HISTORY_TTL,
  });
}

function buildContextBlock(chunks: ChunkRow[]): string {
  if (chunks.length === 0) return "";
  const parts = chunks.map((c, i) => `[${i + 1}] (${c.source}, p.${c.page_num})\n${c.content}`);
  return `Relevant context:\n\n${parts.join("\n\n---\n\n")}`;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Content-Type, x-api-key",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
    },
  });
}

// ---------------------------------------------------------------------------
// Main handler
// ---------------------------------------------------------------------------

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Headers": "Content-Type, x-api-key",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
        },
      });
    }

    // Health check
    if (url.pathname === "/health" && request.method === "GET") {
      return jsonResponse({ status: "ok", service: "krishna-assistant" });
    }

    // Only handle POST /api/v1/query
    if (url.pathname !== "/api/v1/query" || request.method !== "POST") {
      return jsonResponse({ error: "Not Found" }, 404);
    }

    // --- Auth ---
    const apiKey = request.headers.get("x-api-key");
    if (!apiKey || apiKey !== env.API_KEY) {
      return jsonResponse({ error: "Unauthorized" }, 401);
    }

    // --- Parse body ---
    let body: QueryRequest;
    try {
      body = (await request.json()) as QueryRequest;
    } catch {
      return jsonResponse({ error: "Invalid JSON body" }, 400);
    }

    const { question, collection = DEFAULT_COLLECTION } = body;
    if (!question || typeof question !== "string" || question.trim() === "") {
      return jsonResponse({ error: "question is required" }, 400);
    }

    const sessionId = body.session_id || generateSessionId();

    // --- Init clients ---
    const openai = new OpenAI({ apiKey: env.OPENAI_API_KEY });
    const supabase: SupabaseClient = createClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
    const groq = new Groq({ apiKey: env.GROQ_API_KEY });

    try {
      // 1. Embed question
      const embeddingResp = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: question.trim(),
      });
      const queryEmbedding = embeddingResp.data[0].embedding;

      // 2. Vector search via Supabase RPC
      const { data: rawChunks, error: rpcError } = await supabase.rpc("match_chunks", {
        query_embedding: queryEmbedding,
        collection_name: collection,
        match_count: MATCH_COUNT,
      });

      if (rpcError) {
        console.error("Supabase RPC error:", rpcError);
        return jsonResponse({ error: "Vector search failed" }, 502);
      }

      const chunks: ChunkRow[] = rawChunks ?? [];

      // 3. Dynamic threshold filter (0.8 × max_score, matching Python logic)
      let filteredChunks: ChunkRow[] = [];
      if (chunks.length > 0) {
        const maxScore = Math.max(...chunks.map((c) => c.score));
        const threshold = maxScore * 0.8;
        filteredChunks = chunks.filter((c) => c.score >= threshold);
      }

      const hasContext = filteredChunks.length > 0;

      // 4. Load session history from KV
      const history = await loadHistory(env.HISTORY_KV, sessionId);

      // 5. Build messages
      const contextBlock = buildContextBlock(filteredChunks);
      const userContent = hasContext
        ? `${contextBlock}\n\nQuestion: ${question}`
        : question;

      const messages: Array<{ role: "system" | "user" | "assistant"; content: string }> = [
        { role: "system", content: SYSTEM_PROMPT },
        ...history,
        { role: "user", content: userContent },
      ];

      // 6. Generate answer via Groq
      const completion = await groq.chat.completions.create({
        model: "llama-3.1-8b-instant",
        messages,
        temperature: 0.3,
        max_tokens: 512,
      });

      const answer = completion.choices[0]?.message?.content ?? "No response generated.";

      // 7. Save updated history to KV
      const updatedHistory: HistoryMessage[] = [
        ...history,
        { role: "user", content: question },
        { role: "assistant", content: answer },
      ];
      await saveHistory(env.HISTORY_KV, sessionId, updatedHistory);

      // 8. Build sources list
      const sources: Source[] = filteredChunks.map((c) => ({
        source: c.source,
        page_num: c.page_num,
        score: Math.round(c.score * 1000) / 1000,
      }));

      return jsonResponse({
        answer,
        session_id: sessionId,
        sources,
        has_context: hasContext,
      });
    } catch (err) {
      console.error("Worker error:", err);
      const message = err instanceof Error ? err.message : "Internal server error";
      return jsonResponse({ error: message }, 500);
    }
  },
};
