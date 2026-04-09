# Project: LearnMate

## What This Project Does

Personal AI tutor that embeds my Obsidian wiki and raw learning sources into ChromaDB, then teaches me back through conversational RAG. Two modes: "teach" (explain topics, answer questions with sources) and "test" (quiz me, evaluate my answers, show gaps).
## Tech Stack

- **Language**: Python 3.12+
- **LLM Provider**: OpenAI API (GPT-4o for generation)
- **Embeddings**: Chroma Cloud Qwen3 (dense) + Splade (sparse) — no OpenAI embedding cost
- **Framework**: LangChain (chat/retrieval only)
- **Vector Store**: Chroma Cloud (hybrid search: RRF dense + sparse)
- **Web UI**: FastAPI + single-page HTML (SSE streaming, Tailwind CDN)
- **Deployment**: Local (CLI + web UI)
- **Other**: Obsidian markdown files as knowledge source

## Project Structure

```
learnmate/
├── src/
│   ├── ingest/          # Markdown parsing, embedding pipeline, migration
│   ├── chat/            # Teach/test sessions, hybrid search, CLI
│   ├── api/             # FastAPI app + static HTML
│   └── utils/           # Config, env vars
├── .env.example
├── .gitignore
├── CLAUDE.md
├── decisions.md
├── requirements.txt
└── README.md
```

## Rules

### Communication

- Speak Turkish with me. Use technical terms in English as-is.
- Explain what you are doing and why — teach me as you build.
- I don't need to understand every line of code, but I must understand every workflow and how the pieces connect.
- If you are unsure about something, do not assume. Ask me, explain why the question matters, and what changes depending on my answer.

### Code Quality

- Python: black + ruff. Run both before every commit.
- Google style docstrings on every function — no exceptions.
- Type hints on every function signature.
- Comments in plain English — simple enough for a junior developer to understand.
- No over-engineering. Build the simplest thing that works. Add complexity only when needed.

### Git & PR

- **Merge flow: `feature/*` → `develop` (PR) → `main` (only with explicit approval). Never skip develop.**
- `develop` is the default working branch. All feature/fix branches come from `develop` and merge back to `develop` via PR.
- `main` is stable. Only updated from `develop` when I explicitly approve.
- Never push directly to `develop` or `main`.
- Every task starts with a GitHub issue. No branch without an issue.
- Use `feature/#X-short-description` or `fix/#X-short-description` for branch names.
- Include `closes #X` in PR descriptions to auto-close issues on merge.
- Each PR = one feature or one fix. Do not bundle unrelated changes.
- PR description must explain: what was done, why, and what to look for during review.
- Commit messages: short summary line + body if needed. English.
- Promotions (`develop` → `main` → `prod`) only with my explicit approval.

### Decision Logging

- Log every significant decision to `decisions.md` automatically.
- A significant decision = choosing a library, changing architecture, picking an approach over alternatives, or any choice that would be hard to understand later without context.
- Format: date, decision, reasoning, alternatives considered.

### What NOT to Do

- Do not add features I did not ask for.
- Do not refactor working code unless I ask.
- Do not install new packages without telling me why.
- Do not write tests unless I ask (I will ask when ready).
- Do not create documentation files (README, guides) unless I ask.

## Constraints

- OpenAI API rate limits and cost — keep token usage efficient.
- No GPU — all inference via API, all embeddings via API.
- OpenAI API cost — only generation (GPT-4o), embeddings via Chroma credits
- No GPU — all inference via API
- Knowledge source is ~56 wiki pages (small corpus)

## Current Status

- 2026-04-08: MVP complete — teach mode, test mode, CLI, incremental sync.
- 2026-04-09: Chroma Cloud migration (hybrid search, Qwen+Splade, no OpenAI embeddings).
- 2026-04-09: Web UI (FastAPI + HTML, SSE streaming, markdown render).
- 2026-04-09: Bug fixes — history trim, empty input, duplicate summary, RRF threshold.
