from __future__ import annotations

import asyncio
import json
import math
import re
import time
import uuid
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class TextContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "developer", "user", "assistant"]
    content: str | List[Dict[str, Any]]

    @model_validator(mode="after")
    def validate_text_content(self):
        if isinstance(self.content, str):
            return self
        text_parts: List[Dict[str, Any]] = []
        for index, part in enumerate(self.content):
            part_type = part.get("type")
            if part_type != "text":
                raise ValueError(
                    f"unsupported content part type {part_type!r} at index {index}; "
                    "only text parts are supported"
                )
            unknown = set(part) - {"type", "text"}
            if unknown:
                raise ValueError(
                    f"unsupported fields in text content part at index {index}: "
                    f"{', '.join(sorted(unknown))}"
                )
            if not isinstance(part.get("text"), str):
                raise ValueError(f"text content part at index {index} requires a string 'text'")
            text_parts.append(part)
        self.content = text_parts
        return self

    def to_prompt_message(self) -> dict[str, str]:
        content = self.content
        if not isinstance(content, str):
            content = "".join(part["text"] for part in content)
        return {"role": self.role, "content": content}


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: bool = False


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)

    prompt: str | None = None
    messages: List[Message] = Field(min_length=1)

    max_tokens: int = 16
    max_completion_tokens: int | None = None
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | List[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False
    user: str | None = None
    metadata: Dict[str, Any] | None = None

    tools: Any = None
    tool_choice: Any = None
    parallel_tool_calls: Any = None
    functions: Any = None
    function_call: Any = None
    response_format: Any = None
    logprobs: Any = None
    top_logprobs: Any = None
    logit_bias: Any = None
    modalities: Any = None
    audio: Any = None
    prediction: Any = None
    seed: Any = None

    @model_validator(mode="after")
    def validate_supported_contract(self):
        if self.prompt is not None:
            raise ValueError("'prompt' is not supported by /v1/chat/completions; use 'messages'")
        if self.output_token_limit <= 0:
            raise ValueError("max_tokens/max_completion_tokens must be greater than zero")
        if self.n != 1:
            raise ValueError("multiple choices are not supported; 'n' must be 1")
        stop = [self.stop] if isinstance(self.stop, str) else self.stop or []
        if any(value for value in stop):
            raise ValueError(
                "custom stop sequences are not supported; 'stop' must be null or empty"
            )
        if self.presence_penalty != 0:
            raise ValueError("presence_penalty is not implemented and must be 0")
        if self.frequency_penalty != 0:
            raise ValueError("frequency_penalty is not implemented and must be 0")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be a finite non-negative number")
        if not math.isfinite(self.top_p) or not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 or a positive integer")

        unsupported = {
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
            "functions": self.functions,
            "function_call": self.function_call,
            "response_format": self.response_format,
            "top_logprobs": self.top_logprobs,
            "logit_bias": self.logit_bias,
            "modalities": self.modalities,
            "audio": self.audio,
            "prediction": self.prediction,
            "seed": self.seed,
        }
        for name, value in unsupported.items():
            if value is not None:
                raise ValueError(f"'{name}' is not supported by this text-only endpoint")
        if self.logprobs not in (None, False):
            raise ValueError("'logprobs' is not supported by this endpoint")
        return self

    @property
    def output_token_limit(self) -> int:
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        return self.max_tokens

    @property
    def include_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


def _error_body(
    message: str,
    *,
    param: str | None = None,
    code: str = "invalid_request",
    error_type: str = "invalid_request_error",
) -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def _usage_from_reply(reply: UserReply | None) -> Dict[str, int] | None:
    if reply is None or reply.prompt_tokens is None or reply.completion_tokens is None:
        return None
    return {
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens,
        "total_tokens": reply.prompt_tokens + reply.completion_tokens,
    }


def _map_finish_reason(reason: str | None) -> str:
    if reason == "length":
        return "length"
    return "stop"


def _new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _infer_error_param(message: str, fallback: str | None) -> str | None:
    if fallback and not fallback.rsplit(".", 1)[-1].isdigit():
        return fallback
    parameter_names = (
        "max_completion_tokens",
        "max_tokens",
        "parallel_tool_calls",
        "presence_penalty",
        "frequency_penalty",
        "response_format",
        "top_logprobs",
        "tool_choice",
        "function_call",
        "logit_bias",
        "stream_options",
        "modalities",
        "prediction",
        "messages",
        "content",
        "logprobs",
        "functions",
        "metadata",
        "temperature",
        "top_k",
        "top_p",
        "tools",
        "audio",
        "model",
        "prompt",
        "stop",
        "seed",
        "n",
    )
    for name in parameter_names:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", message):
            return name
    return fallback


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        try:
            event = self.event_map[uid]

            while True:
                await event.wait()
                event.clear()

                pending = self.ack_map[uid]
                self.ack_map[uid] = []
                ack = None
                for ack in pending:
                    yield ack
                if ack and ack.finished:
                    break
        finally:
            self.ack_map.pop(uid, None)
            self.event_map.pop(uid, None)

    async def stream_generate(self, uid: int):
        async with aclosing(self.wait_for_ack(uid)) as replies:
            async for ack in replies:
                yield f"data: {ack.incremental_output}\n".encode()
                if ack.finished:
                    break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(
        self,
        uid: int,
        *,
        completion_id: str,
        created: int,
        model: str,
        include_usage: bool,
    ):
        first_chunk = True
        final_reply = None
        async with aclosing(self.wait_for_ack(uid)) as replies:
            async for ack in replies:
                if ack.error:
                    error = _error_body(
                        ack.error,
                        code="backend_error",
                        error_type=(
                            "invalid_request_error"
                            if ack.finish_reason == "length_rejected"
                            else "server_error"
                        ),
                    )
                    yield f"data: {json.dumps(error)}\n\n".encode()
                    final_reply = ack
                    break

                delta = {}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False
                if ack.incremental_output:
                    delta["content"] = ack.incremental_output

                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()

                if ack.finished:
                    final_reply = ack
                    break

        if final_reply is not None and not final_reply.error:
            end_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "delta": {},
                        "index": 0,
                        "finish_reason": _map_finish_reason(final_reply.finish_reason),
                    }
                ],
            }
            yield f"data: {json.dumps(end_chunk)}\n\n".encode()
            usage = _usage_from_reply(final_reply)
            if include_usage and usage is not None:
                usage_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": usage,
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        self.ack_map.pop(uid, None)
        self.event_map.pop(uid, None)
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def openai_validation_error(request: Request, exc: RequestValidationError):
    if not request.url.path.startswith("/v1/"):
        return await request_validation_exception_handler(request, exc)
    first_error = exc.errors()[0] if exc.errors() else {}
    location = [str(item) for item in first_error.get("loc", ()) if item != "body"]
    param = ".".join(location) or None
    message = first_error.get("msg", "Invalid request")
    if message.startswith("Value error, "):
        message = message[len("Value error, ") :]
    param = _infer_error_param(message, param)
    return JSONResponse(
        status_code=400,
        content=_error_body(message, param=param, code=first_error.get("type", "invalid_request")),
    )


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if not state.config.accepts_model(req.model):
        return JSONResponse(
            status_code=404,
            content=_error_body(
                f"The model {req.model!r} does not exist.",
                param="model",
                code="model_not_found",
            ),
        )
    prompt = [msg.to_prompt_message() for msg in req.messages]
    completion_id = _new_completion_id()
    created = int(time.time())
    served_model = state.config.resolved_served_model_name

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.output_token_limit,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(
                state.stream_chat_completions(
                    uid,
                    completion_id=completion_id,
                    created=created,
                    model=served_model,
                    include_usage=req.include_usage,
                ),
                request,
                uid,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    full_content = ""
    final_reply = None
    async with aclosing(state.wait_for_ack(uid)) as replies:
        async for ack in replies:
            full_content += ack.incremental_output
            if ack.finished:
                final_reply = ack
                break

    if final_reply is None:
        return JSONResponse(
            status_code=500,
            content=_error_body(
                "backend ended the request without a final response",
                code="backend_error",
                error_type="server_error",
            ),
        )
    if final_reply.error:
        is_invalid_request = final_reply.finish_reason == "length_rejected"
        return JSONResponse(
            status_code=400 if is_invalid_request else 500,
            content=_error_body(
                final_reply.error,
                code="backend_error",
                error_type="invalid_request_error" if is_invalid_request else "server_error",
            ),
        )

    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": served_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": _map_finish_reason(final_reply.finish_reason),
            }
        ],
    }
    usage = _usage_from_reply(final_reply)
    if usage is not None:
        response["usage"] = usage
    return response


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(
        data=[
            ModelCard(
                id=state.config.resolved_served_model_name,
                root=state.config.model_path,
            )
        ]
    )


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    prompt = [msg.to_prompt_message() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.output_token_limit,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )


async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="shell",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    start_backend()

    logger.info(
        "Served model name: %s; configured model path: %s; model-path compatibility "
        "alias accepted: yes",
        config.resolved_served_model_name,
        config.model_path,
    )
    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
