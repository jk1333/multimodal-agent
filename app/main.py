"""Hosted LensMosaic app for local and Cloud Run deployments.

This service serves the UI, search APIs, item detail APIs, and live WebSocket
endpoints from the same origin.
"""

from __future__ import annotations
import asyncio
import base64
import json
import logging
import threading

from pathlib import Path
from time import perf_counter

import vertexai
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk.agents import Agent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner

from google.adk.tools import ToolContext, google_search
from google import genai

from google.genai import types
from pydantic import BaseModel
from .common import PROJECT_ID, LOCATION, AGENT_MODEL
from .common import logger
from .embedding_vector import (
    _collection_search,
    _rank_results,
    _get_item_details,
    _image_similarity_search,
    EmbeddingRateLimitExceeded,
    ACTIVE_COLLECTION,
    start_warmup_background,
)
from .common import SIMILAR_SEARCH_WORKER_COUNT, MAX_TILE_ITEMS
from .prompt import AGENT_PROMPT
from .session import SESSION_SERVICE, SEARCH_REQUEST_QUEUE, SESSION_STATES
from .session import SessionState, cleanup, session_state_for
from .common import clean_agent_card

APP_NAME = "lens-mosaic-hosted"
STATIC_DIR = Path(__file__).parent / "static"

TEST_ENDPOINTS_ENABLED = False

def _collection_path() -> str:
    return ACTIVE_COLLECTION.collection_id

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


MAIN_LOOP: asyncio.AbstractEventLoop | None = None
SEARCH_WORKERS: list[threading.Thread] = []


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
    start_warmup_background()


@app.on_event("shutdown")
async def shutdown() -> None:
    global MAIN_LOOP
    _stop_search_workers()
    MAIN_LOOP = None


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/.well-known/agent-card.json")
@app.get("/agent-card.json")
@app.get("/agent-card")
async def get_agent_card():
    """Retrieve the Agent Card for registry discovery."""
    try:
        from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
        builder = AgentCardBuilder(agent=agent)
        card = await builder.build()
        card_dict = card.model_dump() if hasattr(card, "model_dump") else card.dict()
        return clean_agent_card(card_dict)
    except Exception as exc:
        logger.error("Failed to generate Agent Card: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate Agent Card: {exc}"
        )


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


@app.post("/test/find_items", response_model=FindItemsTestResponse)
def find_items_test_endpoint(req: FindItemsTestRequest):
    if not TEST_ENDPOINTS_ENABLED:
        raise HTTPException(status_code=404, detail="Test endpoints are disabled")
    
    processed_queries = [q.strip() for q in req.queries if q.strip()]
    processed_ranking_query = req.ranking_query.strip()
    
    recommended, latency_ms = _run_find_items_for_session(
        session_id=req.session_id,
        user_id=req.user_id,
        queries=processed_queries,
        ranking_query=processed_ranking_query,
        publish=req.publish,
    )
    
    return FindItemsTestResponse(
        user_id=req.user_id,
        session_id=req.session_id,
        item_ids=[item["id"] for item in recommended],
        item_names=[item["name"] for item in recommended],
        latency_ms=latency_ms,
    )


@app.post("/test/similar")
def similar_test_endpoint(req: SimilarSearchTestRequest):
    if not TEST_ENDPOINTS_ENABLED:
        raise HTTPException(status_code=404, detail="Test endpoints are disabled")
    
    try:
        image_bytes = base64.b64decode(req.image_b64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="image_b64 must be valid base64") from exc
        
    session = session_state_for(req.session_id, req.user_id)
    session.update_image(image_bytes)
    
    return {
        "status": "accepted",
        "user_id": req.user_id,
        "session_id": req.session_id,
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

    **Invocation Condition:** 
    1. 사용자가 '유사한 아이템을 찾아달라'고 할 때 카메라 분석 후 즉시 호출합니다.
    2. 사용자가 '어울리는 물건'이나 '선물' 등을 추천해 달라고 할 때, `google_search` 도구를 먼저 호출하여 트렌드 검색을 완료한 직후에 보강된 키워드로 이어서 호출합니다.

    **Tool Description:**

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
    instruction=AGENT_PROMPT,
)

RUNNER = Runner(app_name=APP_NAME, agent=agent, session_service=SESSION_SERVICE)
RUN_CONFIG = RunConfig(
    streaming_mode=StreamingMode.BIDI,
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        language_code="ko-KR",
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name="Kore"
            )
        )

    ),
    #input_audio_transcription=types.AudioTranscriptionConfig(
    #    language_codes=['ko-KR', 'en-US']
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