import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from models import Message, Session, get_db, init_db

load_dotenv()
init_db()

app = FastAPI(title="LLM Chat")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).parent
CONDITIONS = {
    "tangible":    BASE_DIR / "prompt_tangible.txt",
    "tangible-2":  BASE_DIR / "prompt_tangible_2.txt",
    "authorless":  BASE_DIR / "prompt_authorless.txt",
    "authorless-2": BASE_DIR / "prompt_authorless_2.txt",
}


def load_condition_prompt(condition: str) -> str:
    path = CONDITIONS.get(condition)
    if path and path.exists():
        return path.read_text().strip()
    raise HTTPException(status_code=404, detail=f"Unknown condition: {condition}")


# ---------- Pydantic schemas ----------

class SessionCreate(BaseModel):
    session_id: str
    condition: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class ChatRequest(BaseModel):
    session_id: str
    message: str


class SystemPromptUpdate(BaseModel):
    system_prompt: str


# ---------- Page routes ----------

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html",
                                      context={"condition": None})


@app.get("/{condition}")
async def condition_page(request: Request, condition: str):
    if condition not in CONDITIONS:
        raise HTTPException(status_code=404, detail="Not found")
    return templates.TemplateResponse(request=request, name="index.html",
                                      context={"condition": condition})


# ---------- API routes ----------

@app.post("/api/sessions")
def create_session(body: SessionCreate, db: DBSession = Depends(get_db)):
    existing = db.get(Session, body.session_id)
    if existing:
        return {"session_id": existing.id, "system_prompt": existing.system_prompt}

    if body.condition:
        system_prompt = load_condition_prompt(body.condition)
    else:
        system_prompt = body.system_prompt

    session = Session(id=body.session_id, condition=body.condition, system_prompt=system_prompt)
    db.add(session)
    db.commit()
    return {"session_id": session.id, "system_prompt": session.system_prompt}


@app.get("/api/sessions/{session_id}/history")
def get_history(session_id: str, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = [
        {
            "role": m.role,
            "content": m.content,
            "prompt_tokens": m.prompt_tokens,
            "completion_tokens": m.completion_tokens,
            "created_at": m.created_at.isoformat(),
        }
        for m in session.messages
    ]
    return {"system_prompt": session.system_prompt, "messages": messages}


@app.patch("/api/sessions/{session_id}/system-prompt")
def update_system_prompt(session_id: str, body: SystemPromptUpdate, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.system_prompt = body.system_prompt
    session.updated_at = datetime.now(timezone.utc)
    db.query(Message).filter(Message.session_id == session_id).delete()
    db.commit()
    return {"ok": True}


@app.post("/api/chat")
def chat(body: ChatRequest, db: DBSession = Depends(get_db)):
    session = db.get(Session, body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    history = [{"role": "system", "content": session.system_prompt}]
    for m in session.messages:
        history.append({"role": m.role, "content": m.content})
    history.append({"role": "user", "content": body.message})

    user_msg = Message(session_id=session.id, role="user", content=body.message)
    db.add(user_msg)
    db.commit()

    def generate():
        collected = []
        prompt_tokens = 0
        completion_tokens = 0

        try:
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=history,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    collected.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            db.delete(user_msg)
            db.commit()
            return

        full_response = "".join(collected)
        assistant_msg = Message(
            session_id=session.id,
            role="assistant",
            content=full_response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        db.add(assistant_msg)
        db.commit()

        yield f"data: {json.dumps({'type': 'done', 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
