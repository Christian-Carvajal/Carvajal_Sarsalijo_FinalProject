import os
import sys
import unittest
import tempfile

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import db_tools
from fsm_controller import BistroFSMAgent


class TestNaturalAIDialogue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize an isolated test database
        cls.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.temp_db.close()
        db_tools.init_db(db_path=cls.temp_db.name, reset=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_db.name):
            try:
                os.remove(cls.temp_db.name)
            except Exception:
                pass

    def test_01_tagalog_menu_inquiry_thinks_dynamically(self):
        """
        Customer asks in Tagalog: 'ano pang iba niyong tinda?'
        The AI Waiter MUST reason dynamically via Ollama LLM.
        It must NOT return the static hardcoded fallback string.
        """
        agent = BistroFSMAgent(table_no=1, db_path=self.temp_db.name, language="filipino")
        result = agent.process_turn("ano pang iba niyong tinda?")
        
        resp = result.get("assistant_response", "").strip()
        static_fallback = "Nandito po ako para tulungan kayo sa Table #1! Maaari kayong mag-order ng burger, pasta, inumin, magtanong ng allergens, o tingnan ang inyong bill."
        
        self.assertNotEqual(
            resp, 
            static_fallback, 
            "AI Waiter must NOT return static fallback string! It must think dynamically."
        )
        self.assertGreater(len(resp), 20, "Response should be substantive.")
        # Engine must be ollama (not fallback_deterministic)
        engine = result.get("engine", "ollama")
        self.assertEqual(engine, "ollama", f"Expected engine to be 'ollama', but got '{engine}'.")

    def test_02_english_menu_inquiry_thinks_dynamically(self):
        """
        Customer asks in English: 'what other items do you sell?'
        The AI Waiter MUST reason dynamically via Ollama LLM.
        It must NOT return the static hardcoded fallback string.
        """
        agent = BistroFSMAgent(table_no=1, db_path=self.temp_db.name, language="english")
        result = agent.process_turn("what other items do you sell?")
        
        resp = result.get("assistant_response", "").strip()
        static_fallback = "I am here to assist you at Table #1! Feel free to browse our menu, ask about dietary allergens, add dishes to your tab, or review your bill."
        
        self.assertNotEqual(
            resp, 
            static_fallback, 
            "AI Waiter must NOT return static fallback string! It must think dynamically."
        )
        self.assertGreater(len(resp), 20, "Response should be substantive.")
        engine = result.get("engine", "ollama")
        self.assertEqual(engine, "ollama", f"Expected engine to be 'ollama', but got '{engine}'.")


if __name__ == "__main__":
    unittest.main()
