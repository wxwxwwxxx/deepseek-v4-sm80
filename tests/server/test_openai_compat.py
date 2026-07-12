from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from minisgl.message import AbortMsg, TokenizeMsg, UserReply
from minisgl.server import api_server
from minisgl.server.api_server import FrontendManager, OpenAICompletionRequest
from openai import AsyncOpenAI, OpenAI

FIXTURES = Path(__file__).parent / "fixtures" / "openai"


@dataclass
class FakeConfig:
    model_path: str = "/models/DeepSeek-V4-Flash"
    served_model_name: str | None = "deepseek-v4-flash"

    @property
    def resolved_served_model_name(self) -> str:
        return self.served_model_name or Path(self.model_path).name

    def accepts_model(self, model: str) -> bool:
        return model in {self.resolved_served_model_name, self.model_path}


class FakeFrontend:
    stream_chat_completions = FrontendManager.stream_chat_completions
    stream_with_cancellation = FrontendManager.stream_with_cancellation

    def __init__(self) -> None:
        self.config = FakeConfig()
        self.uid_counter = 0
        self.sent: list[TokenizeMsg | AbortMsg] = []
        self.aborted: list[int] = []

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        return uid

    async def send_one(self, msg: TokenizeMsg | AbortMsg) -> None:
        self.sent.append(msg)

    async def wait_for_ack(self, uid: int):
        yield UserReply(uid=uid, incremental_output="hello ", finished=False)
        yield UserReply(
            uid=uid,
            incremental_output="world",
            finished=True,
            finish_reason="length",
            prompt_tokens=7,
            completion_tokens=2,
        )

    async def abort_user(self, uid: int) -> None:
        self.aborted.append(uid)

    def shutdown(self) -> None:
        pass


def make_client(frontend: FakeFrontend | None = None) -> tuple[TestClient, FakeFrontend]:
    frontend = frontend or FakeFrontend()
    api_server._GLOBAL_STATE = frontend
    return TestClient(api_server.app), frontend


def parse_sse(response) -> list[dict | str]:
    events: list[dict | str] = []
    for line in response.iter_lines():
        if not line:
            continue
        assert line.startswith("data: ")
        data = line.removeprefix("data: ")
        events.append(data if data == "[DONE]" else json.loads(data))
    return events


def assert_openai_error(response, *, param: str | None) -> None:
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == param
    assert error["message"]
    assert error["code"]


def assert_frontend_state_cleared(frontend: FrontendManager, uid: int) -> None:
    assert uid not in frontend.ack_map
    assert uid not in frontend.event_map


class NoWeightQueue:
    async def get(self):
        await asyncio.Future()

    async def put(self, _msg) -> None:
        pass

    def stop(self) -> None:
        pass


def make_real_frontend() -> FrontendManager:
    queue = NoWeightQueue()
    return FrontendManager(config=FakeConfig(), send_tokenizer=queue, recv_tokenizer=queue)


def finish_request(
    frontend: FrontendManager,
    uid: int,
    *,
    error: str | None = None,
) -> None:
    frontend.ack_map[uid].append(
        UserReply(
            uid=uid,
            incremental_output="" if error else "done",
            finished=True,
            finish_reason="length_rejected" if error else "stop",
            error=error,
            prompt_tokens=2,
            completion_tokens=1,
        )
    )
    frontend.event_map[uid].set()


def test_real_frontend_cleans_normal_stream_and_non_stream_completion() -> None:
    async def run() -> None:
        frontend = make_real_frontend()

        non_stream_uid = frontend.new_user()
        finish_request(frontend, non_stream_uid)
        replies = frontend.wait_for_ack(non_stream_uid)
        async for ack in replies:
            assert ack.finished
            break
        await replies.aclose()
        assert_frontend_state_cleared(frontend, non_stream_uid)

        stream_uid = frontend.new_user()
        finish_request(frontend, stream_uid)
        chunks = [
            chunk
            async for chunk in frontend.stream_chat_completions(
                stream_uid,
                completion_id="chatcmpl-test",
                created=1,
                model="deepseek-v4-flash",
                include_usage=True,
            )
        ]
        assert chunks[-1] == b"data: [DONE]\n\n"
        assert_frontend_state_cleared(frontend, stream_uid)

    asyncio.run(run())


def test_real_frontend_cleans_backend_error_and_early_generator_close() -> None:
    async def run() -> None:
        frontend = make_real_frontend()

        error_uid = frontend.new_user()
        finish_request(frontend, error_uid, error="backend rejected request")
        chunks = [
            chunk
            async for chunk in frontend.stream_chat_completions(
                error_uid,
                completion_id="chatcmpl-error",
                created=1,
                model="deepseek-v4-flash",
                include_usage=False,
            )
        ]
        assert b'"error"' in chunks[0]
        assert_frontend_state_cleared(frontend, error_uid)

        close_uid = frontend.new_user()
        finish_request(frontend, close_uid)
        replies = frontend.wait_for_ack(close_uid)
        await anext(replies)
        await replies.aclose()
        assert_frontend_state_cleared(frontend, close_uid)

    asyncio.run(run())


def test_real_frontend_cleans_cancellation_abort_race_and_disconnect() -> None:
    class DisconnectedRequest:
        async def is_disconnected(self) -> bool:
            return True

    async def run() -> None:
        frontend = make_real_frontend()

        cancel_uid = frontend.new_user()
        replies = frontend.wait_for_ack(cancel_uid)
        task = asyncio.create_task(anext(replies))
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await replies.aclose()
        assert_frontend_state_cleared(frontend, cancel_uid)

        abort_uid = frontend.new_user()
        abort_task = asyncio.create_task(frontend.abort_user(abort_uid))
        replies = frontend.wait_for_ack(abort_uid)
        await replies.aclose()
        await abort_task
        assert_frontend_state_cleared(frontend, abort_uid)

        disconnect_uid = frontend.new_user()
        finish_request(frontend, disconnect_uid)
        source = frontend.stream_chat_completions(
            disconnect_uid,
            completion_id="chatcmpl-disconnect",
            created=1,
            model="deepseek-v4-flash",
            include_usage=False,
        )
        wrapped = frontend.stream_with_cancellation(
            source, DisconnectedRequest(), disconnect_uid
        )
        try:
            await anext(wrapped)
        except asyncio.CancelledError:
            pass
        await wrapped.aclose()
        await source.aclose()
        await asyncio.sleep(0.11)
        assert_frontend_state_cleared(frontend, disconnect_uid)

    asyncio.run(run())


def test_real_frontend_repeated_requests_leave_no_state() -> None:
    async def run() -> None:
        frontend = make_real_frontend()
        for _ in range(10):
            uid = frontend.new_user()
            finish_request(frontend, uid)
            replies = frontend.wait_for_ack(uid)
            async for ack in replies:
                assert ack.finished
            assert_frontend_state_cleared(frontend, uid)
        assert len(frontend.ack_map) == 0
        assert len(frontend.event_map) == 0

    asyncio.run(run())


def test_vllm_openai_chat_benchmark_payload_is_supported() -> None:
    payload = json.loads((FIXTURES / "vllm_0_21_openai_chat.json").read_text())
    request = OpenAICompletionRequest.model_validate(payload)

    assert request.output_token_limit == 32
    assert request.messages[0].to_prompt_message() == {
        "role": "user",
        "content": "benchmark prompt",
    }


def test_legacy_content_output_limit_precedence_and_developer_role() -> None:
    request = OpenAICompletionRequest(
        model="deepseek-v4-flash",
        messages=[
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "hello"},
        ],
        max_tokens=16,
        max_completion_tokens=24,
    )

    assert request.output_token_limit == 24
    assert request.messages[0].to_prompt_message() == {
        "role": "developer",
        "content": "Be concise.",
    }


def test_non_stream_wire_contract_and_exact_usage() -> None:
    client, frontend = make_client()
    with client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "/models/DeepSeek-V4-Flash",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
                "temperature": 0,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)
    assert body["model"] == "deepseek-v4-flash"
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello world"},
            "finish_reason": "length",
        }
    ]
    assert body["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }
    sent = frontend.sent[0]
    assert isinstance(sent, TokenizeMsg)
    assert sent.sampling_params.max_tokens == 4


def test_stream_wire_contract_sse_and_usage_chunk() -> None:
    client, _ = make_client()
    with client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            events = parse_sse(response)

    assert response.status_code == 200
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    ids = {chunk["id"] for chunk in chunks}
    created = {chunk["created"] for chunk in chunks}
    models = {chunk["model"] for chunk in chunks}
    objects = {chunk["object"] for chunk in chunks}
    assert len(ids) == len(created) == 1
    assert models == {"deepseek-v4-flash"}
    assert objects == {"chat.completion.chunk"}
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": "hello "}
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 9


def test_models_identity_is_coherent() -> None:
    client, _ = make_client()
    with client:
        body = client.get("/v1/models").json()
    assert body["data"][0]["id"] == "deepseek-v4-flash"
    assert body["data"][0]["root"] == "/models/DeepSeek-V4-Flash"


def test_model_validation_accepts_public_name_and_full_path_alias() -> None:
    client, frontend = make_client()
    base_payload = {"messages": [{"role": "user", "content": "hello"}]}
    with client:
        served = client.post(
            "/v1/chat/completions",
            json={**base_payload, "model": "deepseek-v4-flash"},
        )
        path_alias = client.post(
            "/v1/chat/completions",
            json={**base_payload, "model": "/models/DeepSeek-V4-Flash"},
        )
        unknown = client.post(
            "/v1/chat/completions",
            json={**base_payload, "model": "gpt-4o"},
        )

    assert served.status_code == 200
    assert path_alias.status_code == 200
    assert served.json()["model"] == "deepseek-v4-flash"
    assert path_alias.json()["model"] == "deepseek-v4-flash"
    assert len(frontend.sent) == 2
    assert unknown.status_code == 404
    assert unknown.json() == {
        "error": {
            "message": "The model 'gpt-4o' does not exist.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }


def test_supported_noop_forms_and_minisgl_extensions_reach_sampling() -> None:
    client, frontend = make_client()
    with client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "stop": [],
                "n": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "logprobs": False,
                "top_k": 3,
                "ignore_eos": True,
                "user": "client-user-id",
                "metadata": {"trace": "abc"},
            },
        )
    assert response.status_code == 200
    sent = frontend.sent[0]
    assert isinstance(sent, TokenizeMsg)
    assert sent.sampling_params.top_k == 3
    assert sent.sampling_params.ignore_eos is True


def test_unsupported_and_unknown_parameters_are_structured_errors() -> None:
    cases = [
        ({"stop": "END"}, "stop"),
        ({"n": 2}, "n"),
        ({"presence_penalty": 0.5}, "presence_penalty"),
        ({"frequency_penalty": -0.5}, "frequency_penalty"),
        ({"tools": [{"type": "function"}]}, "tools"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
        ({"logprobs": True}, "logprobs"),
        ({"seed": 1}, "seed"),
        ({"best_of": 2}, "best_of"),
    ]
    client, _ = make_client()
    with client:
        for fields, param in cases:
            payload = {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "hello"}],
                **fields,
            }
            assert_openai_error(client.post("/v1/chat/completions", json=payload), param=param)


def test_multimodal_parts_and_invalid_base_contract_are_structured_errors() -> None:
    client, _ = make_client()
    with client:
        image = client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
            },
        )
        empty_messages = client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4-flash", "messages": []},
        )
        missing_model = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
    assert_openai_error(image, param="content")
    assert_openai_error(empty_messages, param="messages")
    assert_openai_error(missing_model, param="model")


def test_sync_openai_sdk_non_streaming_and_streaming() -> None:
    client, _ = make_client()
    with client:
        sdk = OpenAI(base_url="http://testserver/v1", api_key="dummy", http_client=client)
        completion = sdk.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=4,
        )
        stream = sdk.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = list(stream)
    assert completion.choices[0].message.content == "hello world"
    assert completion.usage and completion.usage.total_tokens == 9
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == (
        "hello world"
    )
    assert chunks[-1].usage and chunks[-1].usage.total_tokens == 9


def test_async_openai_sdk_non_streaming_and_streaming() -> None:
    frontend = FakeFrontend()
    api_server._GLOBAL_STATE = frontend

    async def run() -> None:
        http_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app),
            base_url="http://testserver",
        )
        sdk = AsyncOpenAI(
            base_url="http://testserver/v1",
            api_key="dummy",
            http_client=http_client,
        )
        completion = await sdk.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "hello"}],
        )
        stream = await sdk.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]
        await sdk.close()
        assert completion.choices[0].message.content == "hello world"
        assert chunks[-1].choices[0].finish_reason == "length"

    asyncio.run(run())


def test_disconnect_schedules_abort() -> None:
    frontend = FakeFrontend()

    class DisconnectedRequest:
        async def is_disconnected(self) -> bool:
            return True

    async def source():
        yield b"data: ignored\n\n"

    async def run() -> None:
        try:
            async for _ in frontend.stream_with_cancellation(source(), DisconnectedRequest(), 9):
                pass
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)

    asyncio.run(run())
    assert frontend.aborted == [9]


def test_backend_errors_are_not_assistant_content() -> None:
    class ErrorFrontend(FakeFrontend):
        async def wait_for_ack(self, uid: int):
            yield UserReply(
                uid=uid,
                incremental_output="",
                finished=True,
                finish_reason="length_rejected",
                error="requested sequence is too long",
                prompt_tokens=100,
                completion_tokens=0,
            )

    client, _ = make_client(ErrorFrontend())
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hello"}],
    }
    with client:
        non_stream = client.post("/v1/chat/completions", json=payload)
        with client.stream(
            "POST", "/v1/chat/completions", json={**payload, "stream": True}
        ) as response:
            stream_events = parse_sse(response)

    assert_openai_error(non_stream, param=None)
    assert stream_events[0]["error"]["type"] == "invalid_request_error"
    assert "choices" not in stream_events[0]
    assert stream_events[-1] == "[DONE]"
