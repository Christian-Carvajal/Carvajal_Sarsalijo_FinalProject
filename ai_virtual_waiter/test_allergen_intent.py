import os
import sys
import tempfile
import unittest

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import db_tools
from fsm_controller import BistroFSMAgent


class TestAllergenAndOrderingIntent(unittest.TestCase):
    def setUp(self):
        # Use an isolated temporary database for clean test execution
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        db_tools.init_db(db_path=self.temp_db.name, reset=True)
        self.agent = BistroFSMAgent(table_no=1, db_path=self.temp_db.name, language="english")

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            try:
                os.remove(self.temp_db.name)
            except Exception:
                pass

    def test_allergen_inquiry_does_not_add_to_cart(self):
        """
        User asks: 'Does the Perpetual Classic Burger have dairy? I have mild lactose intolerance.'
        System MUST answer the allergen question and MUST NOT add the burger to the cart.
        """
        query = "Does the Perpetual Classic Burger have dairy? I have mild lactose intolerance."
        result = self.agent.process_turn(query)

        tools_called = [tc["tool"] for tc in result.get("tool_calls", [])]
        self.assertNotIn(
            "add_item_to_order",
            tools_called,
            f"add_item_to_order should NOT be called on allergen query! Called tools: {tools_called}"
        )

        cart = db_tools.view_current_order(self.agent.session_id, db_path=self.temp_db.name)
        self.assertEqual(
            cart["item_count"],
            0,
            f"Cart should have 0 items after an allergen question, but found: {cart['items']}"
        )

        resp = result.get("assistant_response", "").lower()
        self.assertTrue(
            "dairy" in resp or "cheese" in resp or "lactose" in resp,
            f"Response should address dairy/lactose! Got: {result.get('assistant_response')}"
        )

    def test_explicit_order_adds_to_cart(self):
        """
        User asks: 'Please add 1 Perpetual Classic Burger with extra cheese.'
        System MUST add 1x Perpetual Classic Burger to the cart.
        """
        query = "Please add 1 Perpetual Classic Burger with extra cheese."
        result = self.agent.process_turn(query)

        tools_called = [tc["tool"] for tc in result.get("tool_calls", [])]
        self.assertIn("add_item_to_order", tools_called)

        cart = db_tools.view_current_order(self.agent.session_id, db_path=self.temp_db.name)
        self.assertEqual(cart["item_count"], 1)
        self.assertEqual(cart["items"][0]["name"], "Perpetual Classic Burger")


if __name__ == "__main__":
    unittest.main()
