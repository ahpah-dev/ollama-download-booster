"""Unit tests for models manager and hashing verification."""
import os
import shutil
import tempfile
import unittest
from ollama_booster.models_manager import OllamaManager
from ollama_booster.booster_engine import BoosterEngine


class TestEngineAndManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = OllamaManager(custom_models_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_directory_creation(self):
        self.assertTrue(os.path.isdir(self.manager.blobs_dir))
        self.assertTrue(os.path.isdir(self.manager.manifests_dir))

    def test_blob_paths(self):
        digest = "sha256:abc123def456"
        expected = os.path.join(self.manager.blobs_dir, "sha256-abc123def456")
        self.assertEqual(self.manager.get_blob_path(digest), expected)

    def test_hash_verification(self):
        test_file = os.path.join(self.temp_dir, "sample.bin")
        test_data = b"Hello, Ollama Booster Engine!"
        with open(test_file, "wb") as f:
            f.write(test_data)

        import hashlib
        hex_digest = hashlib.sha256(test_data).hexdigest()
        digest = f"sha256:{hex_digest}"

        self.assertTrue(BoosterEngine._verify_hash(test_file, digest))
        self.assertFalse(BoosterEngine._verify_hash(test_file, "sha256:0000000000000000000000000000000000000000000000000000000000000000"))


if __name__ == "__main__":
    unittest.main()
