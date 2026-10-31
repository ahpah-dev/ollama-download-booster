"""Unit tests for registry client and model parsing."""
import unittest
from ollama_booster.registry_client import RegistryClient


class TestRegistryClient(unittest.TestCase):
    def test_parse_model_ref_standard(self):
        reg, ns, model, tag = RegistryClient.parse_model_ref("llama3.2")
        self.assertEqual(reg, "registry.ollama.ai")
        self.assertEqual(ns, "library")
        self.assertEqual(model, "llama3.2")
        self.assertEqual(tag, "latest")

    def test_parse_model_ref_with_tag(self):
        reg, ns, model, tag = RegistryClient.parse_model_ref("deepseek-r1:7b")
        self.assertEqual(reg, "registry.ollama.ai")
        self.assertEqual(ns, "library")
        self.assertEqual(model, "deepseek-r1")
        self.assertEqual(tag, "7b")

    def test_parse_model_ref_namespaced(self):
        reg, ns, model, tag = RegistryClient.parse_model_ref("eriktonon/qwen3.8-uncensored:9b")
        self.assertEqual(reg, "registry.ollama.ai")
        self.assertEqual(ns, "eriktonon")
        self.assertEqual(model, "qwen3.8-uncensored")
        self.assertEqual(tag, "9b")

    def test_parse_model_ref_custom_registry(self):
        reg, ns, model, tag = RegistryClient.parse_model_ref("hub.example.com/team/custom-model:v1")
        self.assertEqual(reg, "hub.example.com")
        self.assertEqual(ns, "team")
        self.assertEqual(model, "custom-model")
        self.assertEqual(tag, "v1")

    def test_parse_model_ref_huggingface(self):
        # Direct hf.co notation
        reg, ns, model, tag = RegistryClient.parse_model_ref("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
        self.assertEqual(reg, "hf.co")
        self.assertEqual(ns, "bartowski")
        self.assertEqual(model, "Llama-3.2-1B-Instruct-GGUF")
        self.assertEqual(tag, "Q4_K_M")

        # Full browser https:// URL
        reg, ns, model, tag = RegistryClient.parse_model_ref("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
        self.assertEqual(reg, "hf.co")
        self.assertEqual(ns, "bartowski")
        self.assertEqual(model, "Llama-3.2-1B-Instruct-GGUF")
        self.assertEqual(tag, "Q4_K_M")


if __name__ == "__main__":
    unittest.main()
