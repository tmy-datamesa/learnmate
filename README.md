# LearnMate

Personal AI tutor that teaches you from your own notes.

LearnMate embeds your Obsidian wiki into Chroma Cloud, then gives you two ways to learn from it: ask questions (teach mode) and get quizzed (test mode). It uses your own knowledge base as the source of truth — not the internet, not a generic model.
![learnmate4](https://github.com/user-attachments/assets/c5332d5d-581e-4594-99c8-bc99868e0139)

---

## The Problem

I was consuming a lot — podcasts, articles, wiki notes — but never testing myself. Passive consumption with zero active recall. The wiki was growing but I had no way to verify what I actually understood.

LearnMate solves this by turning a static knowledge base into a conversational tutor.

---

## How It Works

```
Obsidian wiki (.md files)
        ↓
   ingest pipeline
   (parse → embed)
        ↓
   Chroma Cloud
   (hybrid index)
        ↓
 query at runtime
   (dense + sparse
      via RRF)
        ↓
  GPT-4o generates
  answer / question
```

**Hybrid search** combines two retrieval methods via Reciprocal Rank Fusion:
- **Dense** (Qwen3 embeddings) — semantic similarity
- **Sparse** (Splade) — keyword matching

This gives better retrieval than cosine-only search, especially for technical terms.

---

## Two Modes

### Teach Mode
Ask anything from your wiki. LearnMate answers with source citations and suggests follow-up questions to push you deeper.

```
You: RAG ile long context arasındaki fark ne?

LearnMate: RAG ve long context arasındaki temel fark...
           [cevap]

Daha derine:
1. RAG pipeline'ında retrieval kalitesi nasıl ölçülür?
2. Long context modellerinin sınırı nerede?

Sources: IBM - Is RAG Still Needed, Divy Yadav - 9 RAG Architectures
```

### Test Mode
LearnMate generates a scenario-based quiz question from your wiki. You answer. It evaluates.

Questions are always scenario-based ("Bir e-ticaret sitesi için...") — never flat definitions. Evaluation is concept-focused, not keyword-matching.

```
Soru: Bir RAG sisteminde retrieval sonuçları kötüyse, sorun
      chunking'de mi, embedding'de mi, yoksa query'de mi?
      Nasıl ayırt edersin?

[cevap]

Score: Kısmi ~
Değerlendirme: Chunking ve embedding'i doğru tespit ettin...
               [detaylı geri bildirim]
Doğru cevap: ...
```

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| LLM | GPT-4o | Generation quality |
| Embeddings | Chroma Cloud Qwen3 + Splade | No per-token cost, hybrid search |
| Vector DB | Chroma Cloud | Managed, hybrid search built-in |
| Retrieval | RRF (dense 0.7 + sparse 0.3) | Better than cosine-only |
| Framework | LangChain (chat only) | Conversation memory, message formatting |
| Web UI | FastAPI + vanilla HTML | Zero build step, SSE streaming |
| Knowledge source | Obsidian markdown | Personal wiki, structured with frontmatter |

---

## Project Structure

```
learnmate/
├── src/
│   ├── ingest/
│   │   ├── parser.py       # Markdown parsing, frontmatter extraction
│   │   ├── embedder.py     # Chroma Cloud client, Qwen+Splade schema
│   │   ├── pipeline.py     # Incremental sync (content hash)
│   │   └── migrate.py      # One-time migration from local ChromaDB
│   ├── chat/
│   │   ├── retriever.py    # Hybrid search, TeachSession, streaming
│   │   └── quiz.py         # Question generation, answer evaluation
│   ├── api/
│   │   ├── app.py          # FastAPI endpoints (teach, test, reset)
│   │   └── static/
│   │       └── index.html  # Single-page UI
│   └── utils/
│       └── config.py       # Env vars
├── decisions.md            # Architecture decisions log
├── .env.example
└── requirements.txt
```

---

## Key Decisions

Full log in [`decisions.md`](decisions.md). Short version:

**Chroma Cloud over local ChromaDB** — OpenAI embedding costs add up on every ingest run. Chroma Cloud's Qwen + Splade are included in Chroma credits. Hybrid search is a bonus.

**1 chunk per wiki file** — Wiki files are 800–4000 chars. Splitting would break context. Full-document retrieval works better at this scale.

**LangChain only for chat** — Ingest uses ChromaDB's native client directly. LangChain adds value in the conversation layer (message history, ChatOpenAI) but would be overhead in the pipeline.

**FastAPI + vanilla HTML** — No build step. SSE streaming works natively. Tailwind CDN handles styling. For a personal tool, this is enough.

**Scenario-based quiz questions** — Flat definition questions ("RAG nedir?") test memorization, not understanding. Forcing scenario framing ("Bir startup'ta...") makes questions harder and more useful.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/tmy-datamesa/learnmate
cd learnmate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=...
WIKI_PATH=/path/to/your/obsidian/wiki
CHROMA_API_KEY=...        # from trychroma.com
CHROMA_TENANT=...
CHROMA_DATABASE=...
```

### 3. Embed your wiki

```bash
python -m src.ingest.pipeline
```

Incremental — only new or changed files are re-embedded.

### 4. Run

**Web UI:**
```bash
uvicorn src.api.app:app --reload
# → http://localhost:8000
```

**CLI:**
```bash
python -m src.chat.cli
```

---

## Wiki Structure

LearnMate expects Obsidian markdown with frontmatter:

```markdown
---
type: concept        # concept | source | entity
subfolder: concepts
---

# RAG

...
```

The `type` field controls how documents are used:
- `concept` / `source` — used in both teach and test context
- `entity` — used in teach context only (skipped in quiz generation)

Only documents from `subfolder: sources` appear in the Sources citation list.

---
