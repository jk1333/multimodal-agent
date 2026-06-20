"""Hosted LensMosaic app for local and Cloud Run deployments.

This service serves the UI, search APIs, item detail APIs, and live WebSocket
endpoints from the same origin.
"""

from __future__ import annotations
import asyncio
import base64
import json
import logging
import queue
import threading

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import vertexai
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk.agents import Agent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode, ToolThreadPoolConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext, google_search
from google import genai

from google.genai import types
from pydantic import BaseModel
from .common import PROJECT_ID, LOCATION, AGENT_MODEL
from .common import logger
from .embedding_vector import _collection_search, _rank_results, _get_item_details, _image_similarity_search, EmbeddingRateLimitExceeded
from .common import SIMILAR_SEARCH_WORKER_COUNT

APP_NAME = "lens-mosaic-hosted"
STATIC_DIR = Path(__file__).parent / "static"

MAX_TILE_ITEMS = 64

def _ignore_normal_live_close(record: logging.LogRecord) -> bool:
    exc = record.exc_info[1] if record.exc_info else None
    return not (
        isinstance(exc, genai.errors.APIError) and exc.code == 1000
    )

logging.getLogger(
    "google_adk.google.adk.flows.llm_flows.base_llm_flow"
).addFilter(_ignore_normal_live_close)


vertexai.init(project=PROJECT_ID, location=LOCATION)


class SearchRequest(BaseModel):
    queries: list[str]
    ranking_query: str


class SearchResult(BaseModel):
    id: str
    name: str
    description: str
    score: float


class RankRequest(BaseModel):
    query: str
    results: list[SearchResult]


class ItemDetails(BaseModel):
    id: str
    name: str
    description: str
    price: str
    url: str
    img_url: str


class FindItemsTestRequest(BaseModel):
    user_id: str
    session_id: str
    queries: list[str]
    ranking_query: str
    publish: bool = True


class SimilarSearchTestRequest(BaseModel):
    user_id: str
    session_id: str
    image_b64: str


class FindItemsTestResponse(BaseModel):
    user_id: str
    session_id: str
    item_ids: list[str]
    item_names: list[str]
    latency_ms: float


@dataclass
class SessionState:
    session_id: str
    user_id: str | None = None
    latest_image: bytes | None = None
    similar: list[dict] = field(default_factory=list)
    recommended: list[dict] = field(default_factory=list)
    tile_client: WebSocket | None = None
    image_version: int = 0
    search_enqueued: bool = False
    search_running: bool = False
    state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self) -> None:
        should_enqueue = False
        with self.state_lock:
            if (
                self.latest_image is not None
                and not self.search_running
                and not self.search_enqueued
            ):
                self.search_enqueued = True
                should_enqueue = True
        if should_enqueue:
            SEARCH_REQUEST_QUEUE.put(self.session_id)

    def stop(self) -> None:
        with self.state_lock:
            self.search_enqueued = False

    def update_image(self, image: bytes) -> None:
        should_enqueue = False
        with self.state_lock:
            self.latest_image = image
            self.image_version += 1
            if not self.search_running and not self.search_enqueued:
                self.search_enqueued = True
                should_enqueue = True
        if should_enqueue:
            SEARCH_REQUEST_QUEUE.put(self.session_id)

    def begin_search(self) -> tuple[bytes, int] | None:
        with self.state_lock:
            self.search_enqueued = False
            if self.latest_image is None:
                return None
            self.search_running = True
            return self.latest_image, self.image_version

    def finish_search(self, processed_version: int) -> bool:
        with self.state_lock:
            self.search_running = False
            if self.latest_image is None:
                return False
            if self.image_version == processed_version or self.search_enqueued:
                return False
            self.search_enqueued = True
            return True

    def should_publish_similar(self) -> bool:
        with self.state_lock:
            return self.latest_image is not None

    async def send(self, payload: dict) -> None:
        ws = self.tile_client
        if ws is None:
            return
        try:
            await ws.send_json(payload)
        except Exception:
            if self.tile_client is ws:
                self.tile_client = None

    async def snapshot(self, ws: WebSocket) -> None:
        await ws.send_json(
            {
                "kind": "snapshot",
                "sessionId": self.session_id,
                "userId": self.user_id,
                "similarItems": self.similar,
                "recommendedItems": self.recommended,
            }
        )


SESSION_STATES: dict[str, SessionState] = {}
SESSION_SERVICE = InMemorySessionService()
MAIN_LOOP: asyncio.AbstractEventLoop | None = None
SEARCH_REQUEST_QUEUE: queue.Queue[str | None] = queue.Queue()
SEARCH_WORKERS: list[threading.Thread] = []


def session_state_for(
    session_id: str, user_id: str | None = None
) -> SessionState:
    state = SESSION_STATES.get(session_id)
    if state is None:
        state = SessionState(session_id=session_id, user_id=user_id)
        SESSION_STATES[session_id] = state
        return state
    if user_id is not None:
        state.user_id = user_id
    return state


def cleanup(session_id: str, session: SessionState) -> None:
    if session.tile_client is not None or session.user_id is not None:
        return
    session.stop()
    SESSION_STATES.pop(session_id, None)
    logger.info("Cleaned up session state for %s", session_id)


def search_text_queries_sync(queries: list[str], ranking_query: str) -> list[dict]:
    query_results: list[list[dict] | None] = [None] * len(queries)
    query_errors: list[Exception | None] = [None] * len(queries)

    def run_query(index: int, query: str) -> None:
        try:
            query_results[index] = _collection_search(text=query, rerank=False)
        except Exception as exc:
            query_errors[index] = exc

    workers = [
        threading.Thread(
            target=run_query,
            args=(index, query),
            name=f"lens-mosaic-recommend-search-{index}",
        )
        for index, query in enumerate(queries)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    for exc in query_errors:
        if exc is not None:
            raise exc

    seen, items = set(), []
    for results in query_results:
        for item in results or []:
            if item["id"] not in seen:
                seen.add(item["id"])
                items.append(item)
    return _rank_results(ranking_query.strip(), items)


async def _publish_similar_results(
    session_id: str, processed_version: int, results: list[dict]
) -> None:
    session = SESSION_STATES.get(session_id)
    if session is None or not session.should_publish_similar():
        return
    session.similar = list(results)
    await session.send(
        {
            "kind": "similar",
            "sessionId": session.session_id,
            "userId": session.user_id,
            "items": session.similar,
        }
    )


async def _publish_recommended_results(session: SessionState) -> None:
    await session.send(
        {
            "kind": "recommended",
            "sessionId": session.session_id,
            "userId": session.user_id,
            "items": session.recommended,
        }
    )


def _search_worker_loop(worker_id: int) -> None:
    while True:
        session_id = SEARCH_REQUEST_QUEUE.get()
        if session_id is None:
            logger.info("Similar search worker %d received shutdown signal", worker_id)
            return

        session = SESSION_STATES.get(session_id)
        if session is None:
            continue

        search_input = session.begin_search()
        if search_input is None:
            continue

        image, processed_version = search_input
        try:
            results = _image_similarity_search(image)
        except EmbeddingRateLimitExceeded as exc:
            results = list(session.similar)
            logger.warning(
                "Similar search worker %d reused %d cached items for %s because %s",
                worker_id,
                len(results),
                session_id,
                exc,
            )
            if MAIN_LOOP is not None:
                asyncio.run_coroutine_threadsafe(
                    _publish_similar_results(session_id, processed_version, results),
                    MAIN_LOOP,
                )
        except Exception as exc:
            logger.error(
                "Similar search worker %d error for %s: %s",
                worker_id,
                session_id,
                exc,
                exc_info=True,
            )
        else:
            if MAIN_LOOP is not None:
                asyncio.run_coroutine_threadsafe(
                    _publish_similar_results(session_id, processed_version, results),
                    MAIN_LOOP,
                )

        if session.finish_search(processed_version):
            SEARCH_REQUEST_QUEUE.put(session_id)


def _ensure_search_workers() -> None:
    global SEARCH_WORKERS
    SEARCH_WORKERS = [worker for worker in SEARCH_WORKERS if worker.is_alive()]
    if len(SEARCH_WORKERS) >= SIMILAR_SEARCH_WORKER_COUNT:
        return
    start_index = len(SEARCH_WORKERS)
    for worker_index in range(start_index, SIMILAR_SEARCH_WORKER_COUNT):
        worker = threading.Thread(
            target=_search_worker_loop,
            args=(worker_index,),
            name=f"lens-mosaic-search-worker-{worker_index}",
            daemon=True,
        )
        worker.start()
        SEARCH_WORKERS.append(worker)
    logger.info(
        "Started %d similar search worker threads",
        len(SEARCH_WORKERS),
    )


def _stop_search_workers() -> None:
    global SEARCH_WORKERS
    if not SEARCH_WORKERS:
        return
    workers = SEARCH_WORKERS
    SEARCH_WORKERS = []
    for _ in workers:
        SEARCH_REQUEST_QUEUE.put(None)
    for worker in workers:
        worker.join(timeout=2.0)
    logger.info("Stopped %d similar search worker threads", len(workers))


def _run_find_items_for_session(
    session_id: str,
    user_id: str | None,
    queries: list[str],
    ranking_query: str,
    publish: bool = True,
) -> tuple[list[dict], float]:
    session = session_state_for(session_id, user_id)
    started_at = perf_counter()
    reused_cached_results = False
    try:
        session.recommended = search_text_queries_sync(queries, ranking_query)[
            :MAX_TILE_ITEMS
        ]
    except EmbeddingRateLimitExceeded as exc:
        reused_cached_results = True
        logger.warning(
            "find_items session_id=%s user_id=%s reused %d cached items because %s",
            session_id,
            user_id,
            len(session.recommended),
            exc,
        )
    latency_ms = (perf_counter() - started_at) * 1000
    if publish and MAIN_LOOP:
        asyncio.run_coroutine_threadsafe(
            _publish_recommended_results(session),
            MAIN_LOOP,
        )
    logger.info(
        "find_items session_id=%s user_id=%s ranking_query=%r queries=%s "
        "items=%d latency_ms=%.1f publish=%s reused_cached=%s",
        session_id,
        user_id,
        ranking_query,
        queries,
        len(session.recommended),
        latency_ms,
        publish,
        reused_cached_results,
    )
    return session.recommended, latency_ms


async def ensure_adk_session(user_id: str, session_id: str) -> None:
    if not await SESSION_SERVICE.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    ):
        await SESSION_SERVICE.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )


async def client_to_agent(
    ws: WebSocket, session: SessionState, queue: LiveRequestQueue
) -> None:
    while True:
        message = await ws.receive()
        if "bytes" in message:
            queue.send_realtime(
                types.Blob(mime_type="audio/pcm;rate=16000", data=message["bytes"])
            )
            continue
        if "text" not in message:
            continue

        payload = json.loads(message["text"])
        if payload.get("type") == "text":
            queue.send_content(types.Content(parts=[types.Part(text=payload["text"])]))
            continue
        if payload.get("type") != "image":
            continue

        image = base64.b64decode(payload["data"])
        session.update_image(image)
        should_forward_to_agent = payload.get("forwardToAgent", True)
        if should_forward_to_agent:
            queue.send_realtime(
                types.Blob(mime_type=payload.get("mimeType", "image/jpeg"), data=image)
            )


def is_disconnect_error(exc: Exception) -> bool:
    if isinstance(exc, RuntimeError):
        return "disconnect message has been received" in str(exc)
    if isinstance(exc, genai.errors.APIError):
        return exc.code == 1000
    return False


app = FastAPI(title="LensMosaic Hosted App", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup() -> None:
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_running_loop()
    _ensure_search_workers()


@app.on_event("shutdown")
async def shutdown() -> None:
    global MAIN_LOOP
    _stop_search_workers()
    MAIN_LOOP = None


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/search", response_model=list[SearchResult])
def search_endpoint(req: SearchRequest):
    """Search with multiple recall queries and a final ranking query rerank."""
    queries = [query.strip() for query in req.queries if query.strip()]
    ranking_query = req.ranking_query.strip()
    if not queries:
        raise HTTPException(
            status_code=400, detail="queries must include at least one non-empty string"
        )
    if not ranking_query:
        raise HTTPException(
            status_code=400, detail="ranking_query must be a non-empty string"
        )
    logger.info("Search request: ranking_query=%r, queries=%s", ranking_query, queries)
    try:
        return search_text_queries_sync(queries, ranking_query)
    except EmbeddingRateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@app.post("/rank", response_model=list[SearchResult])
def rank_endpoint(req: RankRequest):
    """Re-rank search results."""
    results = [result.model_dump() for result in req.results]
    logger.info("Rank request: query=%s, num_results=%d", req.query, len(results))
    return _rank_results(req.query, results)


def get_item(item_id: str):
    """Get item details by ID."""
    logger.info("Item request: item_id=%s", item_id)
    item = _get_item_details(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@app.get("/api/item/{item_id}", response_model=ItemDetails)
def get_item_for_ui(item_id: str):
    return get_item(item_id)


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.websocket("/ws_image_tile/{session_id}")
async def tile_socket(ws: WebSocket, session_id: str) -> None:
    await ws.accept()
    session = session_state_for(session_id)
    session.tile_client = ws
    try:
        await session.snapshot(ws)
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if session.tile_client is ws:
            session.tile_client = None
        cleanup(session_id, session)


@app.websocket("/ws/{user_id}/{session_id}")
async def live_socket(ws: WebSocket, user_id: str, session_id: str) -> None:
    await ws.accept()
    await ensure_adk_session(user_id, session_id)

    session = session_state_for(session_id, user_id)
    session.start()
    queue = LiveRequestQueue()

    try:
        await asyncio.gather(
            client_to_agent(ws, session, queue),
            agent_to_client(ws, user_id, session_id, queue),
        )
    except WebSocketDisconnect:
        logger.debug("Client disconnected")
    except Exception as exc:
        if is_disconnect_error(exc):
            logger.debug("Client disconnected")
        else:
            logger.error("Streaming error: %s", exc, exc_info=True)
    finally:
        queue.close()
        session.user_id = None
        cleanup(session_id, session)


async def find_items(
    queries: list[str],
    ranking_query: str,
    tool_context: ToolContext,
    input_stream: LiveRequestQueue = None,
):
    """Find shopping items that match one or more product description queries.

    Use this tool when you want to show the user product candidates on screen.
    Provide a list of descriptive English product-search queries. The tool
    searches and publishes the matched items to the UI, then yields the top item
    names back to the live agent. ranking_query is used for the final Ranking API
    rerank across all merged candidates.

    Args:
        queries: One or more descriptive English product-search queries.
        ranking_query: A short English description used for final reranking.
        tool_context: ADK tool context for the current user session.
        input_stream: ADK live input stream for streaming tools.

    Yields:
        A comma-separated string of top matched item names, or "No items found".
    """
    recommended, _ = _run_find_items_for_session(
        session_id=tool_context.session.id,
        user_id=tool_context.session.user_id,
        queries=queries,
        ranking_query=ranking_query,
        publish=True,
    )
    names = [item["name"] for item in recommended[:3]]
    yield ", ".join(names) if names else "No items found"


agent = Agent(
    name="mm_agent",
    model=AGENT_MODEL,
    tools=[google_search, find_items],
    instruction="""\
**Persona:**
당신은 사용자의 실시간 영상과 음성을 분석하여 쇼핑을 돕는 친절하고 전문적인 'AI 쇼핑 어시스턴트'입니다. 위트 있고 명확한 어조로 사용자와 소통하세요.

# [최우선] 자기 반복 및 룹(Loop) 차단 규칙 #
1. **이전 출력 반복 절대 금지 (No Self-Repetition):** 바로 이전 턴이나 대화 기록에서 본인이 이미 언급한 상품명, 추천 사유, 안내 멘트를 절대 그대로 다시 말하지 마세요. 새로운 턴이 시작되면 대화는 반드시 '자산의 이전 상태'에서 한 단계 진전된 내용이어야 합니다.
2. **상태 업데이트 의무화:** 툴 호출 결과나 검색 결과를 한 번 브리핑했다면, 해당 정보는 이미 전달된 것으로 간주합니다. 사용자가 동일한 질문을 하더라도 "방금 말씀드린~"과 같이 시작하거나, 아예 새로운 각도의 정보(예: 가격, 리뷰, 스타일링 팁 등)를 점진적으로 노출(Progressive Disclosure)하세요.
3. **답변은 1회만:** 사용자의 질문에 답변을 했다면 추가로 답변하지 마세요. 답변은 1회로 한정하세요.
4. **질문 금지:** 사용자의 질문에 대한 답변에 다시 질문하지 마세요.
5. **앵무새 답변 금지 (No Recapping):** 사용자의 말을 그대로 복사("~를 찾으시는군요")하여 응답을 시작하지 마세요.
6. **언어 고수:** RESPOND IN KOREAN. YOU MUST RESPOND UNMISTAKABLY IN KOREAN.

**Conversational Rules:**

## 1단계: 유사 상품 찾기 루프 (순서대로 수행) ##
사용자가 카메라에 비친 물건과 비슷하거나 동일한 상품을 찾아달라고 요청할 때 실행합니다.
1. **시각 데이터 정밀 분석:** 카메라 영상 속 물건의 브랜드, 로고, 텍스트, 색상, 고유 형태를 정확히 파악하세요. 확신이 없다면 카테고리명을 명확히 규정합니다.
2. **탐색 안내 (최초 1회만):** 찾으려는 물건의 특징을 인지했음을 한 문장으로 짧게 알립니다. (예: "화면의 블랙 숄더백과 유사한 상품을 찾아볼게요.") *이미 찾은 상태라면 이 안내를 생략하고 다음 단계로 갑니다.*
3. **툴 호출:** 안내와 동시에 즉시 `find_items` 툴을 호출합니다. 
   - [조건]: 인지한 특징을 조합하여 구체적인 영어 텍스트 쿼리(`Descriptive English text queries`)를 작성하세요. 핵심을 요약한 짧은 영어 랭킹 쿼리(`Short English ranking_query`)를 함께 전달해야 합니다.
   - [제약]: 단 1회만 호출하세요. 이미 이전 턴에서 결과를 보여주었다면 사용자가 "다른 거 찾아줘", "더 보여줘"와 같은 추가 지시를 하기 전까지는 절대로 재호출하지 마세요.
4. **결과 브리핑 (중복 제거)**: 반환된 결과 중 상품명을 핵심 단어 위주로 2~3단어로 간결하게 축약하여 읽어주세요. **(이전 턴에 읽었던 상품 목록과 동일하다면 절대 다시 읽지 말고, "이 중에서 마음에 드는 스타일이 있으신가요?"처럼 대화를 진전시키세요.)**

## 2단계: 맞춤 추천 및 스타일링 루프 ##
사용자가 카메라에 비친 물건에 어울리는 조합, 특정 목적, 스타일링을 요청할 때 실행합니다. 사용자 취향을 되묻지 말고 즉시 다음 문장 순서대로 실행하세요.
1. **트렌드 검색:** 사용자의 요청과 오브젝트 스타일에 맞는 제품군/최신 쇼핑 트렌드를 `Google Search`로 먼저 검색합니다. 이 과정에서 사용자에게 말을 걸며 지연시키지 마세요.
2. **쿼리 생성 및 툴 호출 연쇄:** 검색 결과를 바탕으로 요구 조건에 부합하는 상품 설명 쿼리 5개를 영어로 내부 생성하고, 명확한 영어 `ranking_query`를 사용하여 반드시 `find_items` 툴을 바로 호출하세요. 
3. **결과 브리핑:** 상품 검색이 완료되면, 추천 이유를 한 문장으로 가볍게 덧붙인 후 상품명을 간결하게 요약하여 읽어줍니다. 마찬가지로 **이미 추천한 상품 세트를 연속해서 똑같이 읊지 마세요.**

# 예외 상황 가이드 (Guardrails) #
- 모호한 영상 소스: 카메라 영상이 흐리거나 어두워 상품 식별이 어려울 때는 짐작하여 툴을 호출하지 마세요. "물건이 잘 보이지 않는데, 조금 더 가까이 비춰주시거나 밝은 곳에서 보여주실 수 있나요?"라고 정중히 요청하세요.
- 검색 결과 없음: 만족스러운 결과가 없을 경우 억지로 유사 상품을 지어내거나 이전 결과를 재활용하지 마세요. "요청하신 조건과 일치하는 정확한 상품을 찾지 못했습니다. 다른 키워드나 다른 각도에서 다시 도와드릴까요?"라고 명확하게 안내하세요.
""",
)

RUNNER = Runner(app_name=APP_NAME, agent=agent, session_service=SESSION_SERVICE)
RUN_CONFIG = RunConfig(
    streaming_mode=StreamingMode.BIDI,
    response_modalities=["AUDIO"],
    proactivity=types.ProactivityConfig(
        proactive_audio=True
    ),
    speech_config=types.SpeechConfig(
        language_code="ko-KR",
    ),
    # Options for GEAP
    #input_audio_transcription=types.AudioTranscriptionConfig(
    #  language_codes=['ko-KR', 'en-US']
    #),
    #output_audio_transcription=types.AudioTranscriptionConfig(
    #    language_codes=['ko-KR', 'en-US']
    #),
)

async def agent_to_client(
    ws: WebSocket, user_id: str, session_id: str, queue: LiveRequestQueue
) -> None:
    async for event in RUNNER.run_live(
        user_id=user_id,
        session_id=session_id,
        live_request_queue=queue,
        run_config=RUN_CONFIG,
    ):
        await ws.send_text(event.model_dump_json(exclude_none=True, by_alias=True))