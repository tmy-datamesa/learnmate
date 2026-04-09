# LearnMate — Architecture

Personal RAG chatbot that teaches from an Obsidian wiki. Two modes: **teach** (Q&A with sources) and **test** (scenario-based quiz with evaluation).

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                      INGEST PIPELINE                    │
│                   (run once / on change)                │
│                                                         │
│  Obsidian wiki (.md)                                    │
│       ↓                                                 │
│  parser.py   →  frontmatter + body                      │
│       ↓                                                 │
│  embedder.py →  content hash diff                       │
│       ↓                                                 │
│  Chroma Cloud  ←  upsert (dense + sparse embeddings)    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     RUNTIME (per query)                 │
│                                                         │
│  User query                                             │
│       ↓                                                 │
│  Hybrid Search  →  Chroma Cloud (Qwen + Splade + RRF)   │
│       ↓                                                 │
│  Relevance filter  →  RELEVANCE_THRESHOLD = -0.010      │
│       ↓                                                 │
│  Context string  +  conversation history                │
│       ↓                                                 │
│  GPT-4o  →  streamed response (SSE)                     │
│       ↓                                                 │
│  FastAPI  →  Web UI / CLI                               │
└─────────────────────────────────────────────────────────┘
```

---

## Ingest Pipeline

**Files:** `src/ingest/pipeline.py` → `parser.py` → `embedder.py`

### Step 1 — Parse

`parser.py` splits each `.md` file into frontmatter metadata and body content:

```
---
type: concept
subfolder: concepts
updated: 2026-04-01
---

# RAG

Retrieval-Augmented Generation works by...
```

↓

```python
metadata = {"type": "concept", "subfolder": "concepts", "updated": "2026-04-01"}
body     = "# RAG\n\nRetrieval-Augmented Generation works by..."
```

Processed folders: `concepts/`, `entities/`, `sources/`. Index and log files are skipped.

Each document gets a stable ID: `concepts/RAG`, `sources/IBM - Is RAG Still Needed`, etc.

### Step 2 — Incremental Sync

The full wiki is not re-embedded on every run. Change detection is done via content hash:

```
document from wiki
      ↓
MD5(content) → "a3f4..."
      ↓
compare with hash stored in Chroma metadata
      ↓
┌─ no existing hash  → ADD    (new document)
├─ hash differs      → UPDATE (content changed)
├─ hash matches      → SKIP   (unchanged, zero API cost)
└─ in DB, not in wiki → DELETE
```

**Why this matters:** Even though Chroma Cloud handles embeddings internally, unnecessary upserts consume Chroma credits and add network round-trips. Incremental sync keeps costs at zero for unchanged documents.

### Step 3 — Embedding Schema

```python
schema = Schema()

# Dense: semantic similarity search
schema.create_index(VectorIndexConfig(
    space="cosine",
    embedding_function=ChromaCloudQwenEmbeddingFunction(
        model=ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B,
        task=None,
    ),
))

# Sparse: keyword matching
schema.create_index(
    SparseVectorIndexConfig(
        source_key=K.DOCUMENT,
        embedding_function=ChromaCloudSpladeEmbeddingFunction(),
    ),
    key="sparse_embedding",
)
```

Chroma Cloud builds **two separate indexes** per document:
- `embedding` — Qwen3 dense vector (0.6B model)
- `sparse_embedding` — Splade sparse vector (learned BM25-style)

---

## Retrieval: Hybrid Search

**File:** `src/chat/retriever.py` — `hybrid_search()`

### Why cosine similarity alone is not enough

| Query | Dense (cosine) | Sparse (Splade) |
|-------|---------------|-----------------|
| "What is the difference between RAG and long context?" | Strong — finds semantically similar text | Weak — "long context" as exact keyword is rare |
| "ChromaDB collection count method" | May drift — model has seen "count" in many unrelated contexts | Strong — technical term, exact match |
| "Karpathy's take on vibe coding" | Medium | Strong — proper noun "Karpathy" found by direct lookup |

Hybrid search closes both gaps by combining the two methods.

### RRF — Reciprocal Rank Fusion

```python
Rrf(
    ranks=[
        Knn(query=query, return_rank=True, limit=20),          # dense
        Knn(query=query, key="sparse_embedding", limit=20),    # sparse
    ],
    weights=[0.7, 0.3],   # dense weighted higher
    k=60,
)
```

RRF score for each document:

```
score(d) = Σ  weight_i / (k + rank_i(d))
```

Example — document ranked 3rd in dense, 1st in sparse:
`0.7/(60+3) + 0.3/(60+1) = 0.0111 + 0.0049 = 0.0160`

Document ranked 1st in dense only:
`0.7/(60+1) = 0.0115`

The first document wins — a document that ranks well in **both** lists beats one that dominates only one list.

**Why 0.7 / 0.3?**  
Most queries are conceptual ("explain RAG trade-offs") where semantic similarity is more reliable. A 0.5/0.5 split was tested but gave odd results for technical terms like "vibe coding" where sparse over-weighted rare proper nouns.

### Relevance Filter

Chroma Cloud returns RRF scores as **negative numbers**. More negative = better match.

```
Poor match  →  score  -0.002  (close to zero)
Good match  →  score  -0.090  (far from zero, strongly ranked)

RELEVANCE_THRESHOLD = -0.010
if score > RELEVANCE_THRESHOLD: continue   ← skip scores close to zero
```

Without this filter, an off-topic query (e.g. "Python syntax") would still return Chroma's best guess. GPT-4o would then generate a hallucinated answer from an unrelated document. The filter forces the model to say "I don't have relevant information" instead.

---

## Generation: GPT-4o + Conversation Memory

**File:** `src/chat/retriever.py` — `TeachSession`

### Message structure (per query)

```
SystemMessage  →  "You are LearnMate, a personal AI tutor..."
HumanMessage   →  (turn 1 question)
AIMessage      →  (turn 1 answer)
HumanMessage   →  (turn 2 question)
AIMessage      →  (turn 2 answer)
...
HumanMessage   →  "Context from wiki:\n[documents]\n\nQuestion: [current question]"
```

### Context vs History separation

Only the question and answer are saved to history — **not the context**:

```python
# Save to history — context intentionally excluded
self._history.append(HumanMessage(content=question))   # question only
self._history.append(AIMessage(content=answer))         # answer only
# context is fetched fresh via hybrid_search on every query
```

**Why:** Context is already retrieved fresh from Chroma on every query. Storing it in history would accumulate tokens across turns, pushing costs up and potentially causing the model to anchor on stale retrieved content.

### History Trim

```python
MAX_HISTORY_TURNS = 20   # 20 Q&A turns = 40 messages
if len(self._history) > MAX_HISTORY_TURNS * 2:
    self._history = self._history[-MAX_HISTORY_TURNS * 2:]
```

Oldest turns are dropped when the limit is hit. This prevents unbounded context window growth across long sessions.

### Temperature settings

| Use case | Temperature | Reason |
|----------|------------|--------|
| Teach mode answers | 0.3 | Consistent, source-grounded explanations |
| Quiz question generation | 0.7 | Variety across questions in the same session |
| Answer evaluation | 0.2 | Deterministic, fair scoring |

---

## SSE Streaming

**Files:** `src/api/app.py` + `src/api/static/index.html`

The CLI uses `.invoke()` (blocking). The web UI streams tokens as they arrive so the user sees a live response.

```
POST /teach
      ↓
TeachSession.ask_stream()
      ↓
LangChain ChatOpenAI.stream()  →  yields tokens one by one
      ↓
SSE event: {"type": "sources", "sources": [...]}  ← sent first
SSE event: {"type": "token", "token": "RAG"}      ← one per token
SSE event: {"type": "token", "token": " works"}
...
SSE event: {"type": "done"}                        ← signals completion
```

Frontend accumulates tokens as plain text during the stream. On `done`, the full text is passed to `marked.js` for markdown rendering. Rendering mid-stream would produce broken output (split tags, partial `**bold**`).

---

## Quiz Mode

**File:** `src/chat/quiz.py` — `TestSession`

Same hybrid search infrastructure as teach mode. The difference: the LLM both **generates** a question and **evaluates** the answer.

```
generate_question(topic)
      ↓
hybrid_search(topic, n_results=3)
      ↓
filter out entity docs         ← quiz is concept-focused, not person-focused
      ↓
QUESTION_PROMPT + context → GPT-4o → question
      ↓
[user answers]
      ↓
evaluate_answer(user_answer)
      ↓
EVALUATE_PROMPT + context + question + answer → GPT-4o → evaluation
      ↓
Score: correct / partial / wrong
```

**Why filter entity documents?**  
A search for "AI concepts" sometimes returns `entities/Karpathy.md`. Without filtering, the prompt would generate "What does Karpathy think about X?" — a question that tests recall of a person's opinion, not understanding of a concept. Entity documents are skipped so questions always target ideas, not people.

**Why enforce scenario-based questions?**  
"What is RAG?" tests memorization. "You're building a chatbot for a 100k-item product catalog. Would you use RAG or long context, and why?" tests understanding. The system prompt forces the LLM into this framing.

---

## Evolution: v1 → v2

### v1 — Local ChromaDB + OpenAI Embeddings

```
wiki → parse → OpenAI text-embedding-3-small → ChromaDB (local file) → cosine search
```

**Problems:**
- OpenAI embedding API cost on every ingest run
- Cosine-only search — technical terms occasionally missed
- Local file storage — not portable, no cloud sync

### v2 — Chroma Cloud + Hybrid Search (current)

```
wiki → parse → Chroma Cloud (Qwen dense + Splade sparse) → RRF hybrid search
```

**Gains:**
- Embedding cost dropped to zero (included in Chroma Cloud credits)
- Hybrid search improved retrieval quality, especially for technical terms
- Managed cloud storage — no local ChromaDB directory to maintain

### Decisions that did not change

| Decision | Reason |
|----------|---------|
| 1 chunk = 1 wiki file | Files are 800–4k chars — splitting would break context and create fragments too small to be useful |
| LangChain only in the chat layer | Ingest is just embed + store — LangChain's chain abstractions add complexity with no benefit there |
| Vanilla HTML, no framework | Zero build step, SSE works natively with `fetch`, Tailwind CDN handles styling |

---

## Data Model

Each document stored in Chroma Cloud:

```
id:       "concepts/RAG"
document: "# RAG\n\nRetrieval-Augmented Generation..."
metadata: {
    filename:     "RAG",
    subfolder:    "concepts",
    type:         "concept",        ← concept | source | entity
    updated:      "2026-04-01",
    content_hash: "a3f4b8...",      ← used for incremental sync
    source_url:   ""
}
```

`subfolder: sources` → document name appears in the Sources citation list  
`type: entity` → skipped during quiz question generation  
`type: concept | source` → used as context in both teach and test modes

---

## Dependency Map

```
src/
├── utils/config.py          ← env vars (OPENAI_API_KEY, CHROMA_*)
├── ingest/
│   ├── parser.py            ← markdown parse, frontmatter extraction
│   ├── embedder.py          ← Chroma Cloud client, schema, incremental sync
│   └── pipeline.py          ← ingest entry point
└── chat/
    ├── retriever.py         ← hybrid_search(), TeachSession, SSE stream
    ├── quiz.py              ← TestSession (reuses hybrid_search)
    └── cli.py               ← CLI entry point
src/api/
    ├── app.py               ← FastAPI endpoints (teach, test, reset)
    └── static/index.html    ← single-page UI (Tailwind CDN, marked.js)
```

Each layer depends only on layers below it: `api → chat → ingest → utils`  
No circular dependencies.
