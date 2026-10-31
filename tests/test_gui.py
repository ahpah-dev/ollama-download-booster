"""Unit tests for CustomTkinter GUI."""
import unittest
from ollama_booster.gui import BoosterApp


class TestBoosterGUI(unittest.TestCase):
    def test_gui_instantiation_and_views(self):
        app = BoosterApp()
        for view_name in ["pull", "models", "inference", "benchmark", "settings"]:
            app.switch_view(view_name)
            app.update()
            self.assertIn(view_name, app.views)

        # Test token UI reflection
        app.hf_token_var.set("hf_testtoken1234567890abcdefghijklm")
        app._update_hf_token_ui()
        app.update()

        # Test toast notification
        app._flash_toast("Test Toast Notification", duration_ms=100)
        app.update()

        app.destroy()


if __name__ == "__main__":
    unittest.main()
