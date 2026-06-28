
import queue
from google.adk.sessions import InMemorySessionService
from dataclasses import dataclass, field
from fastapi import WebSocket
import threading
from .common import logger

SEARCH_REQUEST_QUEUE: queue.Queue[str | None] = queue.Queue()
SESSION_SERVICE = InMemorySessionService()

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