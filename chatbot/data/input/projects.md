# Krishna's Projects

This document describes the three live projects on Krishna's portfolio. All
three are FastAPI services from github.com/KrishnaMoorthy47/GenAI_Projects.

## FinAgent — Multi-agent stock research with a human-in-the-loop approval gate

**What it does**: Input a stock ticker. A supervisor agent routes work across
a web-research agent (Tavily), a financial-data agent (yfinance + SEC EDGAR),
and a sentiment agent, then a report-writer agent drafts a structured
investment brief — and the graph pauses and waits for a human to approve it
before finishing.

**The challenge**: Free-text research queries flow directly into LLM
prompts, and live web-search results get appended into the same message
thread as system instructions — both are real prompt-injection surfaces
(OWASP LLM01), not hypothetical ones.

**The solution**: A two-layer prompt-injection guard runs before any agent
work starts: a regex/keyword heuristic (Unicode-normalized to catch
obfuscation) that's always on, backed by an optional LLM classifier for
inconclusive cases. Flagged requests get rejected with HTTP 400 before a
research thread even opens. Web-search results are wrapped in explicit
"this is data, not instructions" delimiters before being shown to the model.
Every request — flagged or not — is written to a structured audit log.

**The trade-off**: The human-approval gate is a hard pause in the graph,
checkpointed to Postgres — not a fire-and-forget notification. A stock
research agent that can act without a human in the loop is a liability, not
a feature.

**Architecture**: Research request → supervisor → web_research_agent
(Tavily) / financial_data_agent (yfinance + SEC EDGAR) / sentiment_agent
(earnings analysis) → report_writer → human_review (graph pauses) → human
approval → resumes → end.

**Metrics**: 2-layer prompt-injection defense (regex + optional LLM check).
Postgres checkpointing — state survives restarts. Query cap of 2,000
characters, control characters stripped.

**Tech stack**: LangGraph, FastAPI, yfinance, SEC EDGAR API, Tavily,
PostgreSQL, Psycopg 3.

---

## Personal RAG Chatbot — 11-step retrieval pipeline over your own documents

**What it does**: Drop PDFs, text, or HTML into the system, run ingestion
once, and ask questions in natural language. Answers come back grounded in
the source documents with page-level citations — not the model's general
knowledge.

**The challenge**: A RAG system that answers confidently from documents that
don't actually contain the answer is worse than one that says "I don't
know" — hallucination control has to be a first-class design decision, not
an afterthought.

**The solution**: A dynamic similarity threshold (keep only chunks scoring
at least 80% of the best match) filters out weak retrievals before they
reach the model, so an off-topic question doesn't get a confidently-wrong
answer. Every response reports has_context so the caller knows whether real
document content backed the answer. Multi-turn history is tracked per
session_id via Redis, with a documented graceful fallback to in-memory if
Redis isn't available. Every query passes OpenAI's moderation endpoint
before it reaches the retrieval step.

**The trade-off**: Chose a documented in-memory fallback over requiring
Redis to be up. A demo visitor's chat history resetting on a cold start is
an acceptable trade-off — a chatbot with a hard external-service dependency
isn't resilient, it's an outage waiting to happen.

**Architecture**: Ingestion chunks documents (575 tokens, 70% overlap) and
embeds them into a FAISS index. A query is validated, sanitized, and
moderation-checked, then embedded and matched against the FAISS index
(top-5), filtered by the dynamic threshold, assembled into a 4,000-token
context budget alongside Redis chat history, sent to the LLM, and the answer
is returned with source citations.

**Metrics**: FAISS IndexFlatIP, top-5, cosine similarity retrieval. Dynamic
80%-of-best-score threshold filter for hallucination control. 4,000-token
context budget, 5-turn conversation memory.

**Tech stack**: FastAPI, FAISS, LangChain, OpenAI (GPT-4o), Redis, tiktoken.

---

## AgentEval — Evaluation framework for a LangGraph SQL agent

**What it does**: Most engineers build agents. This one measures them. A
ReAct agent answers natural-language questions over a Tamil songs database,
and a separate eval harness runs 20 test cases through it, scoring each on
task success, tool-call accuracy, trajectory efficiency, and hallucination.

**The challenge**: Deterministic scoring can't judge whether a
natural-language answer is actually correct — but LLM-as-judge scoring
needs to be reliable enough to trust, and grounded specifically in what the
SQL query actually returned, not just "does this sound right."

**The solution**: Two of the four scorers are deterministic (tool-call
recall, and a trajectory-efficiency penalty of -0.1 per unnecessary step)
and two use GPT-4o as a judge — one checking whether the answer matches what
was expected, one checking whether the answer is actually grounded in the
SQL result returned, not just plausible-sounding. The four scores combine
into a single weighted score (40% task success, 20% tool accuracy, 20%
trajectory, 20% hallucination), with every run traced to LangSmith for
inspection.

**The trade-off**: Chose to combine two deterministic scorers with two
LLM-as-judge scorers instead of trusting either alone. A pure LLM judge can
rate a confidently wrong answer as convincing; pure deterministic scoring
can't tell whether the answer is actually correct.

**Architecture**: A query goes to a ReAct agent node, which loops with a
tool node against a SQLite database until it has an answer. An eval run
sends 20 test cases through the agent, scores each on the 4 dimensions, and
a report gives a per-case breakdown with LangSmith trace links.

**Metrics**: Test set of 20 cases (8 easy, 8 medium, 4 hard). Scoring across
4 dimensions, weighted 40/20/20/20. Test coverage of 42 tests, fully mocked
— no OpenAI calls needed to run them.

**Tech stack**: LangGraph, FastAPI, SQLite, LangSmith, OpenAI (GPT-4o judge).
