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
LOCATION = "asia-northeast1"
COLLECTION_ID = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/amazon-product-768-compact"
#COLLECTION_ID = "projects/{PROJECT_ID}/locations/{LOCATION}/collections/amazon-product-768-compact-prebuilt"

MAX_TILE_ITEMS = 64

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

def clean_agent_card(data: dict) -> dict:
    """Ensure that required array fields, objects, and strings are represented as appropriate non-null defaults."""
    # 1. Top-level list fields
    for k in ["additionalInterfaces", "security", "signatures"]:
        if data.get(k) is None:
            data[k] = []
            
    # 2. Top-level string fields
    for k in ["documentationUrl", "iconUrl"]:
        if data.get(k) is None:
            data[k] = ""
            
    # 3. Top-level object fields
    if data.get("securitySchemes") is None:
        data["securitySchemes"] = {}
        
    if data.get("provider") is None or not isinstance(data.get("provider"), dict):
        data["provider"] = {
            "organization": "Google Cloud",
            "url": "https://cloud.google.com"
        }
    else:
        provider = data["provider"]
        if not provider.get("organization"):
            provider["organization"] = "Google Cloud"
        if not provider.get("url"):
            provider["url"] = "https://cloud.google.com"

            
    # 4. Capabilities object fields
    capabilities = data.get("capabilities")
    if isinstance(capabilities, dict):
        if capabilities.get("extensions") is None:
            capabilities["extensions"] = []
        for k in ["pushNotifications", "stateTransitionHistory", "streaming"]:
            if capabilities.get(k) is None:
                capabilities[k] = False
                
    # 5. Skills list fields
    if "skills" in data and isinstance(data["skills"], list):
        for skill in data["skills"]:
            if isinstance(skill, dict):
                for k in ["inputModes", "outputModes", "security", "examples"]:
                    if skill.get(k) is None:
                        skill[k] = []
    return data