"""
InternTA API Server
==================

OpenAI-compatible chat completions API backed by vLLM serving
DeepSeek-R1-Distill-Qwen-7B.

Endpoint: POST /v1/chat/completions
Protocol: https://docs.ecopi.chat/api-reference/create-a-chat-completion
Default port: 6006
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, model_validator
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from safety_rag import (
    build_system_prompt,
    EVAL_SHEET_DEFAULT,
    EvaluationRAG,
    KeywordGuardrail,
    load_safety_resources,
    resolve_direct_reply as _resolve_direct_reply,
)

try:
    from vllm.utils import random_uuid
except ImportError:  # pragma: no cover
    def random_uuid() -> str:
        return uuid.uuid4().hex

load_dotenv()

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

API_TOKEN = os.getenv("API_TOKEN", "sk-kfm-ba20250820")
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "DeepSeek-R1-Distill-Qwen-7B"
    if os.path.isdir("DeepSeek-R1-Distill-Qwen-7B")
    else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
)
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "KFM")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "6006"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
SYSTEM_FINGERPRINT = os.getenv("SYSTEM_FINGERPRINT", "fp_2f57f81c11")

ROOT_DIR = Path(__file__).resolve().parent
KEYWORD_XLSX = Path(
    os.getenv("KEYWORD_XLSX", ROOT_DIR / "data" / "附件4-拦截关键词列表.xlsx")
)
EVAL_XLSX = Path(
    os.getenv("EVAL_XLSX", ROOT_DIR / "data" / "附件5-评估测试题.xlsx")
)
EVAL_SHEET = os.getenv("EVAL_SHEET", EVAL_SHEET_DEFAULT)
EXTRA_COVERAGE_BANK = Path(
    os.getenv(
        "EXTRA_COVERAGE_BANK",
        ROOT_DIR / "data" / "extra_coverage_bank.json",
    )
)

engine: Optional[AsyncLLMEngine] = None
keyword_guardrail: Optional[KeywordGuardrail] = None
evaluation_rag: Optional[EvaluationRAG] = None


# ---------- OpenAI / InternTA schema ----------


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: Optional[List[Message]] = None
    dialogue: Optional[List[Message]] = None  # alias for messages (ecopi protocol)
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 8000
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0

    @model_validator(mode="after")
    def resolve_dialogue_alias(self):
        """Accept either `messages` or `dialogue` as the conversation payload."""
        if self.messages is None and self.dialogue is not None:
            self.messages = self.dialogue
        if not self.messages:
            raise ValueError("Either 'messages' or 'dialogue' must be a non-empty array")
        return self


class ChatMessage(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[dict] = None
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: str = SYSTEM_FINGERPRINT


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "internta"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


# ---------- Guardrail / RAG wiring ----------


def extract_latest_user_content(messages: Sequence[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content:
            return message.content
    return ""


def resolve_direct_reply(user_content: str) -> Optional[str]:
    return _resolve_direct_reply(user_content, keyword_guardrail, evaluation_rag)


def augment_messages_with_system_and_rag(
    messages: Sequence[Message],
    user_content: Optional[str] = None,
) -> List[Message]:
    query = user_content if user_content is not None else extract_latest_user_content(messages)

    context = ""
    if evaluation_rag is not None and query:
        hits = evaluation_rag.retrieve(query)
        if hits and hits[0][2] < 1.0:
            context = evaluation_rag.format_context(hits)

    system_content = build_system_prompt()
    if context:
        system_content = f"{system_content}\n\n{context}"

    augmented: List[Message] = [Message(role="system", content=system_content)]
    for message in messages:
        augmented.append(message)
    return augmented


def strip_thinking(text: str) -> str:
    """Keep only content after the last </think>; drop chain-of-thought."""
    marker = "</think>"
    idx = text.rfind(marker)
    if idx == -1:
        return text
    return text[idx + len(marker) :].lstrip()


def build_direct_response(
    request: ChatCompletionRequest,
    content: str,
    request_id: Optional[str] = None,
) -> ChatCompletionResponse:
    msgs = request.messages or []
    prompt_tokens = max(1, sum(len(m.content) for m in msgs) // 2)
    completion_tokens = max(1, len(content) // 2)
    return ChatCompletionResponse(
        id=request_id or f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model or SERVED_MODEL_NAME,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                logprobs=None,
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        system_fingerprint=SYSTEM_FINGERPRINT,
    )


# ---------- Auth ----------


async def verify_token(authorization: Optional[str] = Header(None)):
    if not API_TOKEN:
        return True

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        scheme, token = authorization.split(None, 1)
    except ValueError as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if scheme.lower() != "bearer" or token != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


# ---------- Engine lifecycle ----------


async def _create_engine() -> AsyncLLMEngine:
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine, keyword_guardrail, evaluation_rag
    keyword_guardrail, evaluation_rag = load_safety_resources(
        KEYWORD_XLSX,
        EVAL_XLSX,
        sheet_name=EVAL_SHEET,
        extra_bank_path=EXTRA_COVERAGE_BANK,
    )
    engine = await _create_engine()
    yield
    engine = None


app = FastAPI(
    title="InternTA Chat Completions API",
    description="OpenAI-compatible chat completions via vLLM (DeepSeek-R1-Distill-Qwen-7B)",
    version="1.0.0",
    lifespan=lifespan,
)


async def _get_tokenizer():
    assert engine is not None
    # vLLM API differs slightly across versions
    if hasattr(engine, "get_tokenizer"):
        tok = engine.get_tokenizer()
        if hasattr(tok, "__await__"):
            return await tok
        return tok
    return engine.engine.get_tokenizer()


def _build_prompt(tokenizer, messages: List[Message]) -> str:
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    # Fallback ChatML (DeepSeek-R1-Distill-Qwen family)
    parts = ["<s>"]
    for m in messages:
        parts.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": SERVED_MODEL_NAME,
        "engine_ready": engine is not None,
        "guardrail_ready": keyword_guardrail is not None,
        "rag_ready": evaluation_rag is not None,
        "rag_size": len(evaluation_rag.qa_map) if evaluation_rag else 0,
        "extra_coverage_bank": str(EXTRA_COVERAGE_BANK),
        "extra_coverage_exists": EXTRA_COVERAGE_BANK.is_file(),
        "keyword_size": len(keyword_guardrail.keywords) if keyword_guardrail else 0,
    }


@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    return ModelList(
        data=[
            ModelCard(id=SERVED_MODEL_NAME, created=int(time.time())),
            ModelCard(id=MODEL_PATH, created=int(time.time())),
        ]
    )


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(verify_token)],
)
async def create_chat_completion(request: ChatCompletionRequest):
    # messages is normalized from dialogue by ChatCompletionRequest validator
    assert request.messages is not None

    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported yet")

    user_content = extract_latest_user_content(request.messages)
    direct = resolve_direct_reply(user_content)
    if direct is not None:
        return build_direct_response(request, direct)

    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not ready")

    messages = augment_messages_with_system_and_rag(request.messages, user_content=user_content)
    tokenizer = await _get_tokenizer()
    prompt = _build_prompt(tokenizer, messages)

    temperature = 0.0 if request.temperature is None else max(0.0, float(request.temperature))
    top_p = 1.0 if request.top_p is None else min(1.0, max(0.0, float(request.top_p)))
    max_tokens = 8000 if request.max_tokens is None else int(request.max_tokens)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p if top_p > 0 else 1.0,
        max_tokens=max_tokens,
        stop=request.stop,
        presence_penalty=request.presence_penalty or 0.0,
        frequency_penalty=request.frequency_penalty or 0.0,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    # vLLM internal ids prefer random_uuid()
    engine_req_id = random_uuid()

    try:
        results = engine.generate(prompt, sampling_params, engine_req_id)
        final_output = None
        async for output in results:
            final_output = output
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if final_output is None or not final_output.outputs:
        raise HTTPException(status_code=500, detail="Empty generation result")

    completion = final_output.outputs[0]
    content = strip_thinking(completion.text)
    finish_reason = completion.finish_reason or "stop"

    prompt_tokens = len(final_output.prompt_token_ids or [])
    completion_tokens = len(completion.token_ids or [])

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=request.model or SERVED_MODEL_NAME,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                logprobs=None,
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        system_fingerprint=SYSTEM_FINGERPRINT,
    )


def main():
    uvicorn.run(
        "api:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
