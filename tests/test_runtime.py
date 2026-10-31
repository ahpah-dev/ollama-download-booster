"""Unit tests for RuntimeBooster and inference optimization."""
import unittest
from ollama_booster.runtime_booster import RuntimeBooster


class TestRuntimeBooster(unittest.TestCase):
    def setUp(self):
        self.booster = RuntimeBooster()

    def test_physical_cores_detection(self):
        cores = self.booster.physical_cores
        self.assertIsInstance(cores, int)
        self.assertGreaterEqual(cores, 1)

    def test_optimal_threads(self):
        threads = self.booster.optimal_threads
        self.assertIsInstance(threads, int)
        self.assertGreaterEqual(threads, 1)
        self.assertEqual(threads, self.booster.physical_cores)

    def test_boosted_options(self):
        opts = self.booster.get_boosted_options()
        self.assertIn("num_thread", opts)
        self.assertIn("num_gpu", opts)
        self.assertIn("num_batch", opts)
        self.assertIn("use_mmap", opts)
        self.assertEqual(opts["num_batch"], 1024)
        self.assertEqual(opts["num_gpu"], 99)
        self.assertTrue(opts["use_mmap"])


if __name__ == "__main__":
    unittest.main()
