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

    def test_is_valid_hf_token(self):
        from ollama_booster.registry_client import is_valid_hf_token
        self.assertTrue(is_valid_hf_token("hf_abcdefghijklmnopqrstuvwxyz123456"))
        self.assertTrue(is_valid_hf_token("  hf_abcdefghijklmnopqrstuvwxyz123456  "))
        self.assertFalse(is_valid_hf_token(None))
        self.assertFalse(is_valid_hf_token(""))
        self.assertFalse(is_valid_hf_token("not_a_token"))
        self.assertFalse(is_valid_hf_token("hf_short"))
        self.assertFalse(is_valid_hf_token("hf.co/XHToken/Spark-X2.6-4B-DS"))

    def test_extract_hf_token(self):
        from ollama_booster.registry_client import extract_hf_token
        raw = "Here is my token: hf_0123456789abcdefghijklmnopqrstuv and use it!"
        self.assertEqual(extract_hf_token(raw), "hf_0123456789abcdefghijklmnopqrstuv")
        self.assertIsNone(extract_hf_token("no token here at all"))
        self.assertIsNone(extract_hf_token("hf.co/user/repo"))

    def test_token_save_and_retrieve_env(self):
        import os
        from ollama_booster.registry_client import get_hf_token, save_hf_token
        test_tok = "hf_testtoken1234567890abcdefghijklm"
        save_hf_token(test_tok)
        self.assertEqual(os.environ.get("HF_TOKEN"), test_tok)
        self.assertEqual(get_hf_token(), test_tok)
        # Clear it
        save_hf_token("")
        self.assertNotIn("HF_TOKEN", os.environ)


if __name__ == "__main__":
    unittest.main()
