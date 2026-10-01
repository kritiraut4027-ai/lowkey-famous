import gc
import json
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

# Configure ONNX Runtime for ultra-low memory footprint (<512 MB Free Tier)
try:
    import onnxruntime as ort

    _orig_ort_init = ort.InferenceSession.__init__

    def _lean_inference_session_init(self, path_or_bytes, *args, **kwargs):
        sess_options = kwargs.get("sess_options")
        if sess_options is None:
            sess_options = ort.SessionOptions()
        # Disable pre-allocated memory arena to prevent 200MB+ memory spikes on 512 MB limit
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        kwargs["sess_options"] = sess_options
        return _orig_ort_init(self, path_or_bytes, *args, **kwargs)

    ort.InferenceSession.__init__ = _lean_inference_session_init
except Exception:
    pass

# Define project roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "embedding"
DEFAULT_METADATA_FILE = PROJECT_ROOT / "dataset" / "metadata" / "celebrities.json"

# Legacy fallback celebrity display names for baseline MVP keys
CELEBRITY_DISPLAY_NAMES = {
    "akshay": "Akshay Kumar",
    "alia": "Alia Bhatt",
    "anushka": "Anushka Sharma",
    "dipika": "Deepika Padukone",
    "madhuri": "Madhuri Dixit",
    "salman": "Salman Khan",
    "srk": "Shah Rukh Khan",
    "varun": "Varun Dhawan",
}


class FaceAnalysisError(Exception):
    """Base exception for face analysis errors."""
    pass


class InvalidImageError(FaceAnalysisError):
    """Raised when an uploaded image cannot be opened or decoded."""
    pass


class NoFaceDetectedError(FaceAnalysisError):
    """Raised when zero faces are detected in the image."""
    pass


class MultipleFacesDetectedError(FaceAnalysisError):
    """Raised when more than one face is detected in the image."""
    pass


class ModelInitializationError(FaceAnalysisError):
    """Raised when the face analysis model fails to load."""
    pass


# Global singleton instances for process-level reuse
_face_analyzer = None
_cached_embeddings_dict: Optional[Dict[str, np.ndarray]] = None
_cached_matrix: Optional[np.ndarray] = None
_cached_keys_list: Optional[List[str]] = None
_cached_metadata: Optional[Dict[str, Any]] = None
_cached_image_map: Optional[Dict[str, str]] = None


def get_face_analyzer():
    """
    Lazy process-level singleton for FaceAnalysis.
    Restricts allowed_modules to ['detection', 'recognition'] to avoid loading
    unnecessary 3D landmark and gender/age models into RAM.
    Forces CPUExecutionProvider with ctx_id=-1 for resource-constrained hosting.
    """
    global _face_analyzer
    if _face_analyzer is not None:
        return _face_analyzer

    try:
        from insightface.app import FaceAnalysis

        # InsightFace root directory configurable via environment variable
        models_root = os.environ.get("INSIGHTFACE_ROOT", os.path.expanduser("~/.insightface"))
        model_name = os.environ.get("INSIGHTFACE_MODEL_NAME", "buffalo_s")

        analyzer = FaceAnalysis(
            name=model_name,
            root=models_root,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"]
        )
        # Fixed det_size prevents redundant multi-scale detection sessions and high RAM consumption
        analyzer.prepare(ctx_id=-1, det_size=(640, 640))
        _face_analyzer = analyzer
        return _face_analyzer
    except Exception as e:
        raise ModelInitializationError(f"Failed to initialize FaceAnalysis model: {e}") from e


def load_celebrity_metadata(metadata_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads celebrity metadata registry from JSON file once and caches in memory.
    Returns dictionary indexed by celebrity_key.
    """
    global _cached_metadata
    if _cached_metadata is not None:
        return _cached_metadata

    target_file = Path(metadata_file or DEFAULT_METADATA_FILE)
    metadata_map = {}

    if target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                metadata_map = data.get("celebrities", {})
        except Exception as e:
            print(f"[Warning] Could not parse metadata file {target_file}: {e}")

    _cached_metadata = metadata_map
    return _cached_metadata


def get_celebrity_image_map(dataset_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Scans dataset directory and metadata to map lowercased celebrity keys to their exact relative image paths.
    Prevents case-sensitivity issues on Linux and supports both flat and nested directory structures.
    """
    global _cached_image_map
    if _cached_image_map is not None:
        return _cached_image_map

    target_dir = Path(dataset_dir or DEFAULT_DATASET_DIR)
    mapping = {}

    # 1. First, populate from metadata if available
    metadata = load_celebrity_metadata()
    for key, record in metadata.items():
        img_p = record.get("image_path") or record.get("image_filename")
        if img_p:
            # Normalize to relative path within dataset folder
            p = Path(img_p)
            rel_name = p.name if not p.is_relative_to(target_dir) else str(p.relative_to(target_dir)).replace("\\", "/")
            mapping[key.lower()] = rel_name

    # 2. Next, dynamically scan dataset_dir for any flat files not yet mapped
    if target_dir.exists():
        for file in target_dir.iterdir():
            if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                stem_key = file.stem.lower()
                if stem_key not in mapping:
                    mapping[stem_key] = file.name

        # Also scan dataset/celebrities/ subdirectories if present
        celebrities_dir = target_dir / "celebrities"
        if celebrities_dir.exists() and celebrities_dir.is_dir():
            for celeb_folder in celebrities_dir.iterdir():
                if celeb_folder.is_dir():
                    folder_key = celeb_folder.name.lower()
                    for img_file in celeb_folder.glob("*.*"):
                        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                            rel_path = f"celebrities/{celeb_folder.name}/{img_file.name}"
                            mapping[folder_key] = rel_path
                            break

    _cached_image_map = mapping
    return _cached_image_map


def load_celebrity_embeddings(embedding_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """
    Loads all .npy celebrity embedding vectors into an in-memory dictionary and
    constructs a 2D matrix for vectorized sub-millisecond similarity matching.
    """
    global _cached_embeddings_dict, _cached_matrix, _cached_keys_list
    if _cached_embeddings_dict is not None and _cached_matrix is not None:
        return _cached_embeddings_dict

    target_dir = Path(embedding_dir or DEFAULT_EMBEDDING_DIR)
    if not target_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {target_dir}")

    embeddings = {}
    keys = []
    vectors = []

    # Get active filter from metadata if available
    metadata = load_celebrity_metadata()

    for file in sorted(target_dir.iterdir()):
        if file.suffix.lower() == ".npy":
            key = file.stem.lower()

            # Check if celebrity is explicitly marked inactive
            if key in metadata and metadata[key].get("status") == "inactive":
                continue

            try:
                emb = np.load(str(file))
                if emb.ndim != 1 or emb.shape[0] != 512:
                    raise ValueError(f"Unexpected embedding shape for {file.name}: {emb.shape}")

                # Ensure vector is L2 normalized
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                embeddings[key] = emb
                keys.append(key)
                vectors.append(emb)
            except Exception as e:
                print(f"[Warning] Could not load embedding {file.name}: {e}")

    if not embeddings:
        raise ValueError(f"No valid .npy embeddings found in {target_dir}")

    _cached_embeddings_dict = embeddings
    _cached_keys_list = keys
    _cached_matrix = np.stack(vectors, axis=0)  # Shape: (N, 512)

    return _cached_embeddings_dict


def reload_celebrity_data():
    """Flushes process caches to dynamically reload embeddings, metadata, and image maps."""
    global _cached_embeddings_dict, _cached_matrix, _cached_keys_list, _cached_metadata, _cached_image_map
    _cached_embeddings_dict = None
    _cached_matrix = None
    _cached_keys_list = None
    _cached_metadata = None
    _cached_image_map = None
    load_celebrity_metadata()
    load_celebrity_embeddings()
    get_celebrity_image_map()


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculates cosine similarity between two 1-D embedding vectors:
    dot(u, v) / (norm(u) * norm(v)).
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


def find_hamshakal(
    image_input,
    dataset_dir: Optional[Path] = None,
    embedding_dir: Optional[Path] = None
) -> Tuple[str, float, str, str, float]:
    """
    Processes an input image, extracts facial embedding, and compares it
    against all celebrity embeddings using vectorized matrix cosine similarity.

    Args:
        image_input: File path (str/Path) or numpy BGR array.
        dataset_dir: Directory where celebrity reference photos reside.
        embedding_dir: Directory where pre-computed .npy files reside.

    Returns:
        tuple: (best_match_key, match_percentage, celeb_image_filename, display_name, raw_similarity)
    """
    if isinstance(image_input, (str, Path)):
        image = cv2.imread(str(image_input))
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise InvalidImageError("Unsupported image input format.")

    if image is None or image.size == 0:
        raise InvalidImageError("Image could not be loaded or is empty.")

    # Downscale high-resolution images to max 1024px to prevent memory spikes on free tier
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1024:
        scale = 1024.0 / max_dim
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Get cached FaceAnalysis instance
    analyzer = get_face_analyzer()

    # Detect faces and extract embeddings
    faces = analyzer.get(image)

    if len(faces) == 0:
        raise NoFaceDetectedError("No face detected in the image.")

    if len(faces) > 1:
        raise MultipleFacesDetectedError("Multiple faces detected. Please upload an image with exactly one face.")

    user_embedding = faces[0].embedding
    if user_embedding is None or user_embedding.shape[0] != 512:
        raise FaceAnalysisError("Failed to extract 512-dimensional facial embedding.")

    # Normalize user embedding
    user_norm = np.linalg.norm(user_embedding)
    if user_norm > 0:
        user_embedding = user_embedding / user_norm

    # Ensure embeddings matrix and metadata are loaded
    load_celebrity_embeddings(embedding_dir)
    image_map = get_celebrity_image_map(dataset_dir)
    metadata = load_celebrity_metadata()

    global _cached_matrix, _cached_keys_list
    if _cached_matrix is None or len(_cached_keys_list) == 0:
        raise FaceAnalysisError("Celebrity embedding matrix is empty.")

    # Vectorized cosine similarity: dot product of (N, 512) matrix with (512,) vector
    # Computes similarities for all N celebrities in a single SIMD-accelerated C operation
    similarities = np.dot(_cached_matrix, user_embedding)

    best_idx = int(np.argmax(similarities))
    highest_similarity = float(similarities[best_idx])
    best_match_key = _cached_keys_list[best_idx]

    # Calculate similarity score percentage (clipped to [0, 100])
    percentage = round(float(max(0.0, highest_similarity) * 100.0), 2)

    # Resolve exact filename/path from image map
    celeb_filename = image_map.get(best_match_key, f"{best_match_key}.jpg")

    # Resolve display name: metadata -> legacy dict -> title-cased key
    display_name = None
    if best_match_key in metadata and metadata[best_match_key].get("celebrity_name"):
        display_name = metadata[best_match_key]["celebrity_name"]
    elif best_match_key in CELEBRITY_DISPLAY_NAMES:
        display_name = CELEBRITY_DISPLAY_NAMES[best_match_key]
    else:
        display_name = best_match_key.replace("_", " ").title()

    # Free local tensor memory immediately
    gc.collect()

    return best_match_key, percentage, celeb_filename, display_name, highest_similarity
