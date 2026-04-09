"""FastAPI web server for LearnMate.

Serves the single-page UI and exposes REST endpoints for teach and test modes.
Run: uvicorn src.api.app:app --reload
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.chat.quiz import TestSession
from src.chat.retriever import TeachSession

# --- Session state (single-user personal tool) ---
_teach: TeachSession | None = None
_test: TestSession | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize sessions on startup."""
    global _teach, _test
    _teach = TeachSession()
    _test = TestSession()
    yield


app = FastAPI(title="LearnMate", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"


# --- Request/response models ---


class QuestionRequest(BaseModel):
    question: str


class TopicRequest(BaseModel):
    topic: str = ""


class AnswerRequest(BaseModel):
    answer: str


# --- Teach endpoints ---


@app.post("/teach")
async def teach(req: QuestionRequest):
    """Stream a teach-mode answer for the given question."""
    sources, stream = _teach.ask_stream(req.question)

    # Send sources as first SSE event, then stream answer tokens
    async def generate():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        for token in stream:
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/teach/reset")
async def teach_reset():
    """Clear teach session history."""
    global _teach
    _teach = TeachSession()
    return {"ok": True}


# --- Test endpoints ---


@app.post("/test/question")
async def test_question(req: TopicRequest):
    """Generate a quiz question for the given topic."""
    question, sources = _test.generate_question(req.topic)
    return {"question": question, "sources": sources}


@app.post("/test/evaluate")
async def test_evaluate(req: AnswerRequest):
    """Evaluate the user's answer to the current question."""
    result = _test.evaluate_answer(req.answer)
    return result


@app.get("/test/summary")
async def test_summary():
    """Return the current test session summary."""
    return {"summary": _test.get_summary()}


@app.post("/test/reset")
async def test_reset():
    """Start a new test session."""
    global _test
    _test = TestSession()
    return {"ok": True}


# --- Static files ---

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """Serve the single-page UI."""
    return FileResponse(STATIC_DIR / "index.html")
