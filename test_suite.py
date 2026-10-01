import io
import json
import os
import sys
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.app import app
from backend.compare_faces import (
    find_hamshakal,
    load_celebrity_embeddings,
    load_celebrity_metadata,
    get_celebrity_image_map,
    cosine_similarity,
    get_face_analyzer,
    reload_celebrity_data,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    InvalidImageError,
    CELEBRITY_DISPLAY_NAMES,
)
from scripts.generate_embeddings import run_embedding_generation


class HamshakalFinderTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.dataset_dir = PROJECT_ROOT / "dataset"
        cls.embedding_dir = PROJECT_ROOT / "embedding"
        cls.metadata_file = PROJECT_ROOT / "dataset" / "metadata" / "celebrities.json"
        reload_celebrity_data()

    def test_01_embedding_loading_and_dimensions(self):
        """Test 8 & 9: Celebrity embedding loading and dimension validation (512 float32)."""
        embeddings = load_celebrity_embeddings(self.embedding_dir)
        self.assertGreaterEqual(len(embeddings), 8, "Expected at least 8 celebrity embeddings")
        for name, emb in embeddings.items():
            self.assertEqual(emb.ndim, 1, f"Embedding {name} should be 1-dimensional")
            self.assertEqual(emb.shape[0], 512, f"Embedding {name} must have shape (512,)")
            self.assertEqual(emb.dtype, np.float32, f"Embedding {name} should be float32")
        print(f"\n[PASS] Verified {len(embeddings)} celebrity embeddings with shape (512,)")

    def test_02_model_singleton_reuse(self):
        """Test 2 & 3: Model initialization only happens once (singleton)."""
        analyzer1 = get_face_analyzer()
        analyzer2 = get_face_analyzer()
        self.assertIs(analyzer1, analyzer2, "Analyzer must be a singleton in memory")
        self.assertIn("detection", analyzer1.models)
        self.assertIn("recognition", analyzer1.models)
        self.assertNotIn("landmark_3d_68", analyzer1.models)
        self.assertNotIn("genderage", analyzer1.models)
        print("[PASS] Model singleton verified: exactly 1 instance, only detection & recognition loaded")

    def test_03_cosine_similarity_math(self):
        """Test 10: Cosine similarity mathematical correctness."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v4 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0, places=5)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0, places=5)
        self.assertAlmostEqual(cosine_similarity(v1, v4), -1.0, places=5)
        # Test zero norm handling
        zero_v = np.zeros(3, dtype=np.float32)
        self.assertEqual(cosine_similarity(v1, zero_v), 0.0)
        print("[PASS] Cosine similarity mathematical edge cases verified")

    def test_04_successful_face_match_exact(self):
        """Test 4, 11: Successful face detection and highest matching celebrity selection."""
        test_img_path = self.dataset_dir / "akshay.jpg"
        self.assertTrue(test_img_path.exists(), "Dataset image akshay.jpg must exist")

        best_match, pct, filename, display_name, sim = find_hamshakal(test_img_path)
        self.assertEqual(best_match, "akshay")
        self.assertEqual(display_name, "Akshay Kumar")
        self.assertAlmostEqual(sim, 1.0, places=4)
        self.assertAlmostEqual(pct, 100.0, places=2)
        print(f"[PASS] Exact match verified: {best_match} -> {display_name} ({pct}% similarity)")

    def test_05_no_face_detection_error(self):
        """Test 5: Image with no face raises NoFaceDetectedError without terminating process."""
        blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
        with self.assertRaises(NoFaceDetectedError):
            find_hamshakal(blank_img)
        print("[PASS] NoFaceDetectedError correctly raised on blank image (no process exit)")

    def test_06_multiple_faces_detection_error(self):
        """Test 7: Image with multiple faces raises MultipleFacesDetectedError."""
        akshay = cv2.imread(str(self.dataset_dir / "akshay.jpg"))
        srk = cv2.imread(str(self.dataset_dir / "srk.jpg"))

        # Resize to same height and stitch horizontally
        h = min(akshay.shape[0], srk.shape[0])
        akshay_resized = cv2.resize(akshay, (int(akshay.shape[1] * h / akshay.shape[0]), h))
        srk_resized = cv2.resize(srk, (int(srk.shape[1] * h / srk.shape[0]), h))
        stitched = np.hstack([akshay_resized, srk_resized])

        with self.assertRaises(MultipleFacesDetectedError):
            find_hamshakal(stitched)
        print("[PASS] MultipleFacesDetectedError correctly raised on multi-face image")

    def test_07_invalid_image_error(self):
        """Test 6: Invalid/corrupt image raises InvalidImageError."""
        with self.assertRaises(InvalidImageError):
            find_hamshakal("non_existent_file_path.jpg")
        print("[PASS] InvalidImageError correctly raised on missing/corrupt image")

    def test_08_api_health_endpoint(self):
        """Test 1: Healthcheck endpoint returns 200 and JSON status."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertGreaterEqual(data["celebrities_indexed"], 8)
        print("[PASS] /health endpoint returned healthy status")

    def test_09_api_cors_headers(self):
        """Test 13: CORS configuration allows cross-origin requests."""
        resp = self.client.get("/health", headers={"Origin": "https://hamshakal.vercel.app"})
        self.assertIn("Access-Control-Allow-Origin", resp.headers)
        print("[PASS] CORS headers verified for cross-origin frontend")

    def test_10_api_upload_success(self):
        """Test 12: POST /upload with valid image returns expected JSON schema."""
        test_img_path = self.dataset_dir / "srk.jpg"
        with open(test_img_path, "rb") as f:
            data = {"image": (io.BytesIO(f.read()), "srk.jpg")}
            resp = self.client.post("/upload", data=data, content_type="multipart/form-data")

        self.assertEqual(resp.status_code, 200)
        json_data = resp.get_json()
        self.assertEqual(json_data["status"], "ok")
        self.assertEqual(json_data["celebrity_key"], "srk")
        self.assertEqual(json_data["celebrity_name"], "Shah Rukh Khan")
        self.assertIn("celebrity_image_url", json_data)
        self.assertGreaterEqual(json_data["match_percentage"], 90.0)
        self.assertEqual(json_data["score_type"], "cosine_similarity")
        print(f"[PASS] /upload API success: matched {json_data['celebrity_name']} ({json_data['match_percentage']}%)")

    def test_11_api_upload_no_face(self):
        """Test: POST /upload with blank image returns 200 status 'no_face'."""
        blank_img = np.zeros((200, 200, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", blank_img)
        data = {"image": (io.BytesIO(buf.tobytes()), "blank.jpg")}
        resp = self.client.post("/upload", data=data, content_type="multipart/form-data")

        self.assertEqual(resp.status_code, 200)
        json_data = resp.get_json()
        self.assertEqual(json_data["status"], "no_face")
        print("[PASS] /upload API graceful response for no face detected")

    def test_12_api_upload_invalid_file(self):
        """Test: POST /upload with text file returns 400 status 'invalid_image'."""
        data = {"image": (io.BytesIO(b"not an image file content"), "document.txt")}
        resp = self.client.post("/upload", data=data, content_type="multipart/form-data")

        self.assertEqual(resp.status_code, 400)
        json_data = resp.get_json()
        self.assertEqual(json_data["status"], "invalid_image")
        print("[PASS] /upload API rejected invalid file format with HTTP 400")

    def test_13_case_insensitive_dataset_serving(self):
        """Test: Serving dataset images works regardless of case (Anushka.jpg vs anushka.jpg)."""
        resp1 = self.client.get("/dataset/anushka.jpg")
        resp2 = self.client.get("/dataset/Anushka.jpg")
        self.assertEqual(resp1.status_code, 200, "Should serve anushka.jpg")
        self.assertEqual(resp2.status_code, 200, "Should serve Anushka.jpg")
        print("[PASS] Case-insensitive dataset image resolution verified")

    def test_14_temporary_upload_cleanup(self):
        """Test 11 & Privacy: Verifies temporary upload files are cleaned up from disk."""
        uploads_before = len(list((PROJECT_ROOT / "uploads").glob("*.*")))
        test_img_path = self.dataset_dir / "alia.jpg"
        with open(test_img_path, "rb") as f:
            data = {"image": (io.BytesIO(f.read()), "alia.jpg")}
            resp = self.client.post("/upload", data=data, content_type="multipart/form-data")

        self.assertEqual(resp.status_code, 200)
        uploads_after = len(list((PROJECT_ROOT / "uploads").glob("*.*")))
        self.assertEqual(uploads_before, uploads_after, "Temporary uploads should be deleted after processing")
        print("[PASS] Temporary upload file auto-cleanup verified")

    def test_15_metadata_registry_loading(self):
        """Test: Metadata file loads and contains required schema fields for celebrities."""
        metadata = load_celebrity_metadata()
        self.assertGreaterEqual(len(metadata), 8, "Expected at least 8 registered celebrities")
        for key, record in metadata.items():
            self.assertIn("celebrity_key", record)
            self.assertIn("celebrity_name", record)
            self.assertIn("image_path", record)
            self.assertIn("embedding_path", record)
            self.assertIn("status", record)
            self.assertEqual(record["status"], "active")
        print(f"[PASS] Metadata registry validated with {len(metadata)} schema-compliant records")

    def test_16_vectorized_similarity_equivalence(self):
        """Test: Vectorized matrix similarity matches individual scalar cosine similarity exactly."""
        from backend.compare_faces import _cached_matrix, _cached_keys_list
        reload_celebrity_data()

        # Dummy normalized probe vector
        np.random.seed(42)
        probe = np.random.randn(512).astype(np.float32)
        probe = probe / np.linalg.norm(probe)

        # Vectorized result
        vectorized_sims = np.dot(_cached_matrix, probe)

        # Scalar results
        embeddings = load_celebrity_embeddings()
        for idx, key in enumerate(_cached_keys_list):
            scalar_sim = cosine_similarity(probe, embeddings[key])
            self.assertAlmostEqual(vectorized_sims[idx], scalar_sim, places=5)
        print(f"[PASS] Vectorized matrix search mathematically identical to scalar cosine across {len(_cached_keys_list)} candidates")

    def test_17_incremental_generation_skips_existing(self):
        """Test: Incremental embedding generator skips already processed embeddings in --new mode."""
        report = run_embedding_generation(mode="new", force=False)
        self.assertEqual(report["queued"], 0, "No new candidates should be queued when all embeddings are up to date")
        self.assertEqual(report["successfully_generated"], 0)
        print("[PASS] Incremental generator successfully bypassed already computed embeddings")

    def test_18_nested_celebrity_image_serving(self):
        """Test: Serving nested structured celebrity images (dataset/celebrities/<key>/image.jpg)."""
        resp = self.client.get("/dataset/celebrities/deepika_padukone/image.jpg")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type, "image/jpeg")
        self.assertGreater(len(resp.data), 1000)
        print("[PASS] Nested structured celebrity image successfully served with HTTP 200")


if __name__ == "__main__":
    unittest.main(verbosity=2)
