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
