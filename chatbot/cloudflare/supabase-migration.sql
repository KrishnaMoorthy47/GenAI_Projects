-- Krishna's Assistant — Supabase Migration
-- Run once in the Supabase SQL editor.
--
-- Creates the match_chunks RPC function used by the Cloudflare Worker.
-- Requires pgvector extension (already enabled if the chatbot table exists).
--
-- The chatbot_chunks table schema (for reference — already exists):
--   id          bigserial primary key
--   collection  text not null
--   source      text
--   page_num    int
--   chunk_index int
--   content     text
--   embedding   vector(1536)

CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding vector(1536),
  collection_name text,
  match_count     int DEFAULT 5
)
RETURNS TABLE(
  id       bigint,
  content  text,
  source   text,
  page_num int,
  score    float
)
LANGUAGE sql STABLE AS $$
  SELECT
    id,
    content,
    source,
    page_num,
    1 - (embedding <=> query_embedding) AS score
  FROM chatbot_chunks
  WHERE collection = collection_name
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;

-- Grant execute to the service role used by the Worker
GRANT EXECUTE ON FUNCTION match_chunks(vector, text, int) TO service_role;
