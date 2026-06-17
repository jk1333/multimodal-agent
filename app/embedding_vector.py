from google import genai
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import vectorsearch_v1beta
from time import perf_counter, monotonic, sleep
from .common import PROJECT_ID, LOCATION
from .common import EMBEDDING_MAX_REQUESTS_PER_MINUTE
import os
from google.genai import types
from collections import deque
from dataclasses import dataclass, field
import threading
from .common import logger

RANKING_CONFIG = (
    f"projects/{PROJECT_ID}/locations/global/rankingConfigs/default_ranking_config"
)
COLLECTION_ID = "compact-amazon-product-dataset-image-text-768"
DEFAULT_IMAGE_MIME_TYPE = "image/jpeg"

EMBEDDING_MAX_RETRIES = 3
EMBEDDING_RETRY_BASE_DELAY_SECONDS = 0.5
SEARCH_TOP_K = 100
TEXT_QUERY_HYBRID_WEIGHTS = [1.35, 0.65]
IMAGE_QUERY_HYBRID_WEIGHTS = [0.65, 1.35]

@dataclass(frozen=True)
class CollectionConfig:
    collection_id: str
    embedding_model: str
    text_vector_field: str
    image_vector_field: str
    output_dimensionality: int | None = None

SUPPORTED_COLLECTIONS: dict[str, CollectionConfig] = {
    COLLECTION_ID: CollectionConfig(
        collection_id=COLLECTION_ID,
        embedding_model="gemini-embedding-2",
        text_vector_field="text_embedding",
        image_vector_field="image_embedding",
        output_dimensionality=768,
    ),
}

try:
    ACTIVE_COLLECTION = SUPPORTED_COLLECTIONS[COLLECTION_ID]
except KeyError as exc:
    supported = ", ".join(sorted(SUPPORTED_COLLECTIONS))
    raise RuntimeError(
        "Unsupported LENS_MOSAIC_COLLECTION_ID "
        f"{COLLECTION_ID!r}. Supported values: {supported}"
    ) from exc

@dataclass
class RollingWindowRateLimiter:
    max_requests: int
    window_seconds: float = 60.0
    timestamps: deque[float] = field(default_factory=deque, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve(self) -> tuple[bool, int]:
        if self.max_requests <= 0:
            return True, 0

        now = monotonic()
        cutoff = now - self.window_seconds
        with self.lock:
            while self.timestamps and self.timestamps[0] <= cutoff:
                self.timestamps.popleft()
            current = len(self.timestamps)
            if current >= self.max_requests:
                return False, current
            self.timestamps.append(now)
            return True, current + 1

    def current_count(self) -> int:
        if self.max_requests <= 0:
            return 0

        now = monotonic()
        cutoff = now - self.window_seconds
        with self.lock:
            while self.timestamps and self.timestamps[0] <= cutoff:
                self.timestamps.popleft()
            return len(self.timestamps)

EMBEDDING_RATE_LIMITER = RollingWindowRateLimiter(
    max_requests=EMBEDDING_MAX_REQUESTS_PER_MINUTE
)

embedding_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location="global",
)
search_client = vectorsearch_v1beta.DataObjectSearchServiceClient()
data_client = vectorsearch_v1beta.DataObjectServiceClient()
rank_client = discoveryengine.RankServiceClient()

def _collection_path() -> str:
    return f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{ACTIVE_COLLECTION.collection_id}"

def _search_result_to_dict(result: vectorsearch_v1beta.SearchResult) -> dict | None:
    obj = result.data_object
    if obj is None:
        return None
    item_id = obj.data_object_id or obj.name.split("/")[-1]
    data = obj.data
    if data is None:
        details = _get_item_details(item_id)
        if details is None:
            logger.warning("Skipping search result with missing data for item %s", item_id)
            return None
        data = details
    return {
        "id": item_id,
        "name": data.get("name", ""),
        "description": data.get("description", ""),
        "score": result.distance,
    }

class EmbeddingRateLimitExceeded(RuntimeError):
    """Raised when the app-side Gemini embedding RPM budget has been exhausted."""

def _embed_with_gemini_embedding_2(
    text: str | None = None,
    image: bytes | None = None,
) -> list[float]:
    """Generate a Gemini Embedding 2 vector from text or image input."""
    if embedding_client is None:
        raise RuntimeError("Gemini embedding client is not configured")

    contents: str | types.Part
    if text is not None:
        contents = text
    else:
        contents = types.Part.from_bytes(data=image, mime_type=DEFAULT_IMAGE_MIME_TYPE)

    config = types.EmbedContentConfig(
        output_dimensionality=ACTIVE_COLLECTION.output_dimensionality
    )
    for attempt in range(EMBEDDING_MAX_RETRIES + 1):
        allowed, current_rpm = EMBEDDING_RATE_LIMITER.reserve()
        if not allowed:
            raise EmbeddingRateLimitExceeded(
                "Gemini embedding RPM budget exceeded: "
                f"{current_rpm}/{EMBEDDING_MAX_REQUESTS_PER_MINUTE} requests "
                "in the last 60 seconds"
            )
        try:
            response = embedding_client.models.embed_content(
                model=ACTIVE_COLLECTION.embedding_model,
                contents=contents,
                config=config,
            )
            if not response.embeddings:
                raise RuntimeError("Gemini embedding request returned no embeddings")
            return list(response.embeddings[0].values)
        except genai.errors.APIError as exc:
            if exc.status != "RESOURCE_EXHAUSTED" or attempt >= EMBEDDING_MAX_RETRIES:
                raise
            delay_seconds = EMBEDDING_RETRY_BASE_DELAY_SECONDS * (2**attempt)
            logger.warning(
                "Embedding request hit RESOURCE_EXHAUSTED; retrying in %.1fs "
                "(attempt %d/%d)",
                delay_seconds,
                attempt + 1,
                EMBEDDING_MAX_RETRIES,
            )
            sleep(delay_seconds)

    raise RuntimeError("Embedding retry loop exited unexpectedly")


def _generate_query_embedding(
    text: str | None = None,
    image: bytes | None = None,
) -> tuple[list[float], float]:
    """Generate the Gemini embedding query vector."""
    if text is None and image is None:
        raise ValueError("Either text or image must be provided for embedding")

    started_at = perf_counter()
    embedding = _embed_with_gemini_embedding_2(text=text, image=image)
    embed_ms = (perf_counter() - started_at) * 1000
    return embedding, embed_ms


def _collection_search(
    text: str | None = None,
    image: bytes | None = None,
    rerank: bool = True,
) -> list[dict]:
    """Search the active Gemini Embedding collection by text or image."""
    started_at = perf_counter()
    source = "text" if text is not None else "image"
    results, embed_ms, batch_search_ms, rerank_ms = (
        _hybrid_collection_search(text=text, image=image, rerank=rerank)
    )
    total_ms = (perf_counter() - started_at) * 1000
    logger.info(
        "Search latency: model=%s source=%s rerank=%s embed_ms=%.1f "
        "batch_search_ms=%.1f rerank_ms=%.1f total_ms=%.1f results=%d",
        ACTIVE_COLLECTION.embedding_model,
        source,
        rerank,
        embed_ms,
        batch_search_ms,
        rerank_ms,
        total_ms,
        len(results),
    )
    return results


def _image_similarity_search(image: bytes) -> list[dict]:
    """Search the active collection by image similarity only."""
    started_at = perf_counter()
    results, embed_ms, search_ms = _image_similarity_collection_search(image=image)
    total_ms = (perf_counter() - started_at) * 1000
    logger.info(
        "Search latency: model=%s source=image-similarity rerank=%s embed_ms=%.1f "
        "search_ms=%.1f total_ms=%.1f results=%d",
        ACTIVE_COLLECTION.embedding_model,
        False,
        embed_ms,
        search_ms,
        total_ms,
        len(results),
    )
    return results


def _hybrid_collection_search(
    text: str | None = None,
    image: bytes | None = None,
    rerank: bool = True,
) -> tuple[list[dict], float, float, float]:
    """Search Gemini Embedding 2 collections via VS2 batch search with built-in RRF."""
    embedding, embed_ms = _generate_query_embedding(text=text, image=image)
    weights = TEXT_QUERY_HYBRID_WEIGHTS if text is not None else IMAGE_QUERY_HYBRID_WEIGHTS
    batch_started_at = perf_counter()
    request = vectorsearch_v1beta.BatchSearchDataObjectsRequest(
        parent=_collection_path(),
        searches=[
            vectorsearch_v1beta.Search(
                vector_search=vectorsearch_v1beta.VectorSearch(
                    search_field=ACTIVE_COLLECTION.text_vector_field,
                    vector=vectorsearch_v1beta.DenseVector(values=embedding),
                    top_k=SEARCH_TOP_K,
                    output_fields=vectorsearch_v1beta.OutputFields(
                        data_fields=["name", "description"]
                    ),
                )
            ),
            vectorsearch_v1beta.Search(
                vector_search=vectorsearch_v1beta.VectorSearch(
                    search_field=ACTIVE_COLLECTION.image_vector_field,
                    vector=vectorsearch_v1beta.DenseVector(values=embedding),
                    top_k=SEARCH_TOP_K,
                    output_fields=vectorsearch_v1beta.OutputFields(
                        data_fields=["name", "description"]
                    ),
                )
            ),
        ],
        combine=vectorsearch_v1beta.BatchSearchDataObjectsRequest.CombineResultsOptions(
            ranker=vectorsearch_v1beta.Ranker(
                rrf=vectorsearch_v1beta.ReciprocalRankFusion(
                    weights=weights
                )
            ),
            output_fields=vectorsearch_v1beta.OutputFields(
                data_fields=["name", "description"]
            ),
            top_k=SEARCH_TOP_K,
        ),
    )
    response = search_client.batch_search_data_objects(request)
    batch_search_ms = (perf_counter() - batch_started_at) * 1000
    fused_response = response.results[0].results if response.results else []
    fused_results: list[dict] = []
    for result in fused_response:
        item = _search_result_to_dict(result)
        if item is not None:
            fused_results.append(item)
    if rerank:
        rerank_started_at = perf_counter()
        ranked_results = _rank_results(text or "", fused_results)
        rerank_ms = (perf_counter() - rerank_started_at) * 1000
    else:
        ranked_results = fused_results
        rerank_ms = 0.0
    return ranked_results, embed_ms, batch_search_ms, rerank_ms


def _image_similarity_collection_search(image: bytes) -> tuple[list[dict], float, float]:
    """Search Gemini Embedding 2 collections with the image embedding field only."""
    embedding, embed_ms = _generate_query_embedding(image=image)
    search_started_at = perf_counter()
    request = vectorsearch_v1beta.SearchDataObjectsRequest(
        parent=_collection_path(),
        vector_search=vectorsearch_v1beta.VectorSearch(
            search_field=ACTIVE_COLLECTION.image_vector_field,
            vector=vectorsearch_v1beta.DenseVector(values=embedding),
            top_k=SEARCH_TOP_K,
            output_fields=vectorsearch_v1beta.OutputFields(
                data_fields=["name", "description"]
            ),
        ),
    )
    response = search_client.search_data_objects(request)
    search_ms = (perf_counter() - search_started_at) * 1000
    results: list[dict] = []
    for result in response.results:
        item = _search_result_to_dict(result)
        if item is not None:
            results.append(item)
    return results, embed_ms, search_ms


def _rank_results(query: str, results: list[dict]) -> list[dict]:
    """Re-rank search results using the Vertex AI Ranking API."""
    if not results or not query:
        return results

    records = [
        discoveryengine.RankingRecord(
            id=item["id"],
            title=item["name"],
            content=item.get("description", ""),
        )
        for item in results
    ]
    request = discoveryengine.RankRequest(
        ranking_config=RANKING_CONFIG,
        query=query,
        records=records,
        top_n=len(records),
    )
    response = rank_client.rank(request=request)

    ranked_by_id = {record.id: record.score for record in response.records}
    for item in results:
        item["score"] = ranked_by_id.get(item["id"], 0.0)
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _get_item_details(item_id: str) -> dict | None:
    """Fetch item details from the collection by ID."""
    name = f"{_collection_path()}/dataObjects/{item_id}"
    try:
        obj = data_client.get_data_object(
            vectorsearch_v1beta.GetDataObjectRequest(name=name)
        )
    except Exception:
        return None

    return {
        "id": item_id,
        #"name": "",
        #"description": "",
        "price": "",
        "url": "",
        "img_url": f"https://storage.googleapis.com/jk-amazon-products-thumbnail/{item_id}.webp",
        "name": str(obj.data.get("name", "")),
        "description": str(obj.data.get("description", "")),
        #"price": str(obj.data.get("price", "")),
        #"url": str(obj.data.get("url", "")),
        #"img_url": str(obj.data.get("img_url", "")),
    }