from dotenv import load_dotenv
import google.auth
from pathlib import Path
import os
import logging

load_dotenv(Path(__file__).parent / ".env", override=True)

_, PROJECT_ID = google.auth.default()

#For thumbnail server
IMAGE_SERVER = "https://thumbnail.aidemo.dev"
#IMAGE_SERVER = "https://storage.googleapis.com/jk-amazon-products-thumbnail"

#For gemini live to use VertexAI
#os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = "TRUE"
#os.environ['GOOGLE_CLOUD_LOCATION'] = "us-west1"
#AGENT_MODEL = "gemini-live-2.5-flash-native-audio"

#For gemini live to use AI Studio
AGENT_MODEL = "gemini-3.1-flash-live-preview"

#For Vector Search 2.0
LOCATION = "asia-southeast1"
#COLLECTION_ID = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/compact-amazon-product-dataset-image-text-768"
COLLECTION_ID = "projects/sandbox-373102/locations/asia-southeast1/collections/amazon-product-dataset-image-text-768-all"

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc

EMBEDDING_MAX_RPM_ENV = "LENS_MOSAIC_GEMINI_EMBEDDING_MAX_RPM"
EMBEDDING_MAX_REQUESTS_PER_MINUTE = _env_int(EMBEDDING_MAX_RPM_ENV, default=1500)

SIMILAR_SEARCH_WORKER_ENV = "LENS_MOSAIC_SIMILAR_SEARCH_WORKERS"

SIMILAR_SEARCH_WORKER_COUNT = max(1, _env_int(SIMILAR_SEARCH_WORKER_ENV, default=100))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)