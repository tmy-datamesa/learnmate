# Decision Log

Every significant technical decision is recorded here.
Claude updates this file automatically after each decision.

---

## Decisions

### 2026-04-08 | CLI-first, no web UI for MVP

**Decision**: Start with a terminal-based chatbot, skip web UI entirely.
**Why**: Fastest path to a working RAG pipeline. Learning the retrieval chain matters more than building a frontend right now.
**Alternatives**: Streamlit (easy but adds dependency), Gradio (same), FastAPI + Next.js (too much for MVP).
**Impact**: No frontend code needed. Focus stays on ingest pipeline + LangChain conversation chain.

### 2026-04-08 | LangChain + ChromaDB over raw OpenAI calls

**Decision**: Use LangChain for retrieval chain and ChromaDB for local vector storage.
**Why**: Learning LangChain is a goal. ChromaDB already familiar. Local persistent storage means no external DB setup.
**Alternatives**: Raw OpenAI API + manual embedding management (simpler but no learning value), LlamaIndex (less familiar), Pinecone (cloud dependency, overkill for personal use).
**Impact**: Adds LangChain as core dependency. Project doubles as a LangChain learning exercise.

### 2026-04-08 | Three-branch git strategy (develop / main / prod)

**Decision**: Use develop for active work, main for stable, prod for production-ready.
**Why**: Practice real git workflow discipline. Matches the project templates we built.
**Alternatives**: Single main branch (simpler but no release discipline).
**Impact**: All feature branches come from develop. PRs required for every merge.

### 2026-04-08 | Wiki = 1 chunk per file, no splitting

**Decision**: Each wiki markdown file becomes one document in ChromaDB without further chunking.
**Why**: Wiki files are short (800–4000 chars). Splitting would break context and create fragments too small to be useful. Retrieval quality is better with whole documents at this scale.
**Alternatives**: Split by heading (standard for long docs), fixed-size chunks with overlap (LangChain default).
**Impact**: Simpler pipeline — no chunking logic needed for wiki. Raw sources will use heading-based chunking later since they're longer.

### 2026-04-08 | ChromaDB native embedding over LangChain wrapper

**Decision**: Use ChromaDB's built-in `OpenAIEmbeddingFunction` instead of LangChain's embedding classes for the ingest pipeline.
**Why**: Ingest is just embed + store. LangChain's VectorStore abstraction adds complexity without benefit here. LangChain will be used in the chat/retrieval module where its chain/graph features actually matter.
**Alternatives**: LangChain Chroma integration (langchain-chroma) for both ingest and retrieval.
**Impact**: Fewer dependencies at ingest time. Clean separation: ingest uses ChromaDB directly, chat uses LangChain.
