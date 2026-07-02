import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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

PROMPTS_DIR = Path(__file__).parent / "prompts"

PROMPT_POOL = {
    "animal_testing":   PROMPTS_DIR / "kor_argument_pool_animal_testing.txt",
    "social_media_ban": PROMPTS_DIR / "kor_argument_pool_social_media.txt",
}

PROMPT_TURN1 = {
    "animal_testing":   PROMPTS_DIR / "prompt_shared_core_turn1.txt",
    "social_media_ban": PROMPTS_DIR / "prompt_shared_core_2_turn1.txt",
}

PROMPT_TURN2PLUS = {
    "animal_testing":   PROMPTS_DIR / "prompt_shared_core_turn2plus.txt",
    "social_media_ban": PROMPTS_DIR / "prompt_shared_core_2_turn2plus.txt",
}

STARTER_HINTS = {
    "animal_testing":   ["동물실험 찬반 논쟁에 대해 어떻게 생각해?", "인간의 질병 치료를 위해 동물실험을 해도 된다고 봐?"],
    "social_media_ban": ["청소년 SNS 금지 법안 제정에 대해 어떻게 생각해?", "청소년들의 SNS 사용을 법적으로 금지해야 한다고 봐?"],
}

CONDITION_BLOCKS = {
    "animal_testing": {
        "interest":   PROMPTS_DIR / "prompt_condition_interest.txt",
        "neutral":    PROMPTS_DIR / "prompt_condition_neutral.txt",
        "authorless": PROMPTS_DIR / "prompt_condition_authorless.txt",
    },
    "social_media_ban": {
        "interest":   PROMPTS_DIR / "prompt_condition_interest_2.txt",
        "neutral":    PROMPTS_DIR / "prompt_condition_neutral_2.txt",
        "authorless": PROMPTS_DIR / "prompt_condition_authorless_2.txt",
    },
}

# URL 조건 키 → (topic_key, condition_name)
# 참가자에게 노출되는 키는 익명 코드로 관리 (연구자 참고용 매핑은 아래 주석)
# condition1: animal_testing / authorless
# condition2: animal_testing / neutral
# condition3: animal_testing / interest
# condition4: social_media_ban / authorless
# condition5: social_media_ban / neutral
# condition6: social_media_ban / interest
CONDITION_MAP = {
    "condition1": ("animal_testing",   "authorless"),
    "condition2": ("animal_testing",   "neutral"),
    "condition3": ("animal_testing",   "interest"),
    "condition4": ("social_media_ban", "authorless"),
    "condition5": ("social_media_ban", "neutral"),
    "condition6": ("social_media_ban", "interest"),
}


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


def load_condition_prompt(condition: str, turn: int) -> str:
    topic_key, condition_name = CONDITION_MAP[condition]
    pool = PROMPT_POOL[topic_key].read_text(encoding="utf-8").strip()
    core_path = PROMPT_TURN1[topic_key] if turn == 1 else PROMPT_TURN2PLUS[topic_key]
    shared = core_path.read_text(encoding="utf-8").strip()
    block = CONDITION_BLOCKS[topic_key][condition_name].read_text(encoding="utf-8").strip()
    return shared + "\n\n" + block + "\n\n" + pool


def build_redirect_url(base_url: str, panel_id: str, status: str = "001") -> str:
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update({"panel_id": panel_id, "status": status})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


# ---------- Pydantic schemas ----------

class SessionCreate(BaseModel):
    session_id: str
    condition: str | None = None
    panel_id: str | None = None
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
                                      context={"condition": None, "panel_id": None})


@app.get("/{condition}")
async def condition_page(request: Request, condition: str, panel_id: str | None = None):
    if condition not in CONDITION_MAP:
        raise HTTPException(status_code=404, detail="Not found")
    topic_key, _ = CONDITION_MAP[condition]
    return templates.TemplateResponse(request=request, name="index.html",
                                      context={"condition": condition, "panel_id": panel_id,
                                               "starter_hints": STARTER_HINTS[topic_key]})


# ---------- API routes ----------

@app.post("/api/sessions")
def create_session(body: SessionCreate, db: DBSession = Depends(get_db)):
    existing = db.get(Session, body.session_id)
    if existing:
        return {"session_id": existing.id, "system_prompt": existing.system_prompt}

    system_prompt = load_condition_prompt(body.condition, turn=1) if body.condition else body.system_prompt
    session = Session(
        id=body.session_id,
        condition=body.condition,
        panel_id=body.panel_id,
        system_prompt=system_prompt,
    )
    db.add(session)
    db.commit()
    return {"session_id": session.id, "system_prompt": session.system_prompt}


@app.get("/api/sessions/{session_id}/history")
def get_history(session_id: str, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = [
        {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
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


@app.post("/api/sessions/{session_id}/complete")
def complete_session(session_id: str, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    redirect_base = os.environ.get("REDIRECT_BASE_URL", "")
    if not redirect_base or not session.panel_id:
        return {"redirect_url": None}

    redirect_url = build_redirect_url(redirect_base, session.panel_id)
    return {"redirect_url": redirect_url}


@app.post("/api/chat")
def chat(body: ChatRequest, db: DBSession = Depends(get_db)):
    session = db.get(Session, body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    turn = sum(1 for m in session.messages if m.role == "user") + 1
    if session.condition:
        system_prompt = load_condition_prompt(session.condition, turn)
    else:
        system_prompt = session.system_prompt

    history = [{"role": "system", "content": system_prompt}]
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
            stream = get_openai_client().chat.completions.create(
                model="gpt-5.4-mini",
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

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
