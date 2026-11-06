"""
stress_test_defense.py
Comprehensive Edge-Case Stress Testing Suite for Final Panel Defense
Course: BSCS 3112 - Artificial Intelligence (Lesson 7 Final Project)
University of Perpetual Help System DALTA

Authors: Christian Ezekiel L. Carvajal & John Miko P. Sarsalijo

Tests all 8 required edge cases and live defense scenarios:
1. Attempting to order out-of-stock item (Crispy Buffalo Chicken Tenders, stock=0)
2. Attempting to order out-of-stock item (Crispy Slice Salmon, stock=0)
3. Attempting to exceed inventory limit (Pan-Seared Salmon, requesting 5 when stock=2)
4. Illegal state transition shortcut (GREETING directly to CONFIRMED with empty cart)
5. Normal order lifecycle through FSM states: GREETING -> TAKING_ORDER -> REVIEW -> CONFIRMED
6. Post-confirmation state lock (attempting to add, remove, or modify items after checkout)
7. Allergen query isolation (asking allergen questions must NEVER add to cart)
8. Creator attribution (must identify Christian Carvajal and John Miko Sarsalijo)
"""

import os
import sys
import unittest
import tempfile
import sqlite3

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import db_tools
from fsm_controller import (
    BistroFSMAgent,
    STATE_GREETING,
    STATE_MENU_BROWSING,
    STATE_TAKING_ORDER,
    STATE_REVIEW,
    STATE_CONFIRMED,
)
from speech_engine import SpeechEngine


class TestFinalPanelDefenseStress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create an isolated temporary SQLite database for testing
        cls.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.temp_db.close()
        sql_schema = os.path.join(CURRENT_DIR, "restaurant_db.sql")
        db_tools.init_db(db_path=cls.temp_db.name, sql_path=sql_schema, reset=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_db.name):
            try:
                os.remove(cls.temp_db.name)
            except Exception:
                pass

    def setUp(self):
        # Clean sessions and line items between tests
        with db_tools.get_db_connection(self.temp_db.name) as conn:
            conn.execute("DELETE FROM order_line_items;")
            conn.execute("DELETE FROM table_sessions;")
            conn.execute("DELETE FROM kitchen_tickets;")
            conn.commit()

    def test_01_out_of_stock_chicken_tenders_blocked(self):
        """Edge Case 1: Out-of-stock item (Crispy Buffalo Chicken Tenders, stock=0) must be blocked."""
        agent = BistroFSMAgent(table_no=1, db_path=self.temp_db.name, language="english")
        res = agent.execute_tool("add_item_to_order", {"item_name": "Crispy Buffalo Chicken Tenders", "qty": 1})
        
        self.assertEqual(res.get("status"), "out_of_stock", "Stock=0 item must return out_of_stock status.")
        self.assertFalse(res.get("success"), "Success must be False for out-of-stock item.")
        self.assertIn("out of stock", res.get("message", "").lower())
        self.assertIn("recommended_alternatives", res, "System must provide alternative dishes.")
        
        # Verify cart remains empty
        cart = db_tools.view_current_order(agent.session_id, db_path=self.temp_db.name)
        self.assertEqual(cart["item_count"], 0, "Cart must remain empty when ordering out-of-stock item.")

    def test_02_out_of_stock_salmon_blocked(self):
        """Edge Case 2: Out-of-stock item (Crispy Slice Salmon, stock=0) must be blocked."""
        agent = BistroFSMAgent(table_no=1, db_path=self.temp_db.name, language="english")
        res = agent.execute_tool("add_item_to_order", {"item_name": "Crispy Slice Salmon", "qty": 1})
        
        self.assertEqual(res.get("status"), "out_of_stock")
        self.assertFalse(res.get("success"))
        self.assertIn("recommended_alternatives", res)
        
        cart = db_tools.view_current_order(agent.session_id, db_path=self.temp_db.name)
        self.assertEqual(cart["item_count"], 0)

    def test_03_inventory_limit_exceeded_blocked(self):
        """Edge Case 3: Requesting 5 Pan-Seared Salmon when stock is 2 must be blocked."""
        agent = BistroFSMAgent(table_no=1, db_path=self.temp_db.name, language="english")
        res = agent.execute_tool("add_item_to_order", {"item_name": "Pan-Seared Salmon", "qty": 5})
        
        self.assertEqual(res.get("status"), "out_of_stock")
        self.assertFalse(res.get("success"))
        self.assertEqual(res.get("stock_available"), 2)
        
        cart = db_tools.view_current_order(agent.session_id, db_path=self.temp_db.name)
        self.assertEqual(cart["item_count"], 0)

    def test_04_illegal_shortcut_greeting_to_confirmed_blocked(self):
        """Edge Case 4: Cannot jump from GREETING directly to CONFIRMED with an empty tab."""
        agent = BistroFSMAgent(table_no=1, db_path=self.temp_db.name, language="english")
        self.assertEqual(agent.state, STATE_GREETING)
        
        # Customer attempts to confirm without adding items
        res = agent.execute_tool("confirm_order_and_deduct_stock", {})
        self.assertNotEqual(res.get("status"), "success", "Confirming empty cart must not succeed.")
        self.assertNotEqual(agent.state, STATE_CONFIRMED, "FSM must not transition to CONFIRMED on empty cart.")

    def test_05_normal_order_lifecycle(self):
        """Verify normal lifecycle: GREETING -> TAKING_ORDER -> REVIEW -> CONFIRMED."""
        agent = BistroFSMAgent(table_no=2, db_path=self.temp_db.name, language="english")
        self.assertEqual(agent.state, STATE_GREETING)

        # 1. Add Perpetual Classic Burger
        res1 = agent.execute_tool("add_item_to_order", {
            "item_name": "Perpetual Classic Burger",
            "qty": 1,
            "instructions": "Extra Cheese"
        })
        self.assertEqual(res1.get("status"), "success")
        self.assertEqual(agent.state, STATE_TAKING_ORDER)

        # 2. Add Fresh Calamansi Iced Tea
        res2 = agent.execute_tool("add_item_to_order", {
            "item_name": "Fresh Calamansi Iced Tea",
            "qty": 1,
            "instructions": ""
        })
        self.assertEqual(res2.get("status"), "success")
        self.assertEqual(agent.state, STATE_TAKING_ORDER)

        # 3. View bill
        bill = agent.execute_tool("view_current_order", {})
        self.assertEqual(agent.state, STATE_REVIEW)
        self.assertEqual(bill["item_count"], 2)
        self.assertAlmostEqual(bill["grand_total"], 205.00, places=2)

        # 4. Confirm checkout
        checkout = agent.execute_tool("confirm_order_and_deduct_stock", {})
        self.assertEqual(checkout.get("status"), "success")
        self.assertEqual(agent.state, STATE_CONFIRMED)
        self.assertIn("receipt_id", checkout)

    def test_06_post_confirmation_state_lock(self):
        """Edge Case 6: Once in CONFIRMED, state is locked and all cart modifications are blocked."""
        agent = BistroFSMAgent(table_no=3, db_path=self.temp_db.name, language="english")
        
        # Add item and confirm
        agent.execute_tool("add_item_to_order", {"item_name": "Perpetual Classic Burger", "qty": 1})
        confirm_res = agent.execute_tool("confirm_order_and_deduct_stock", {})
        self.assertEqual(confirm_res.get("status"), "success")
        self.assertEqual(agent.state, STATE_CONFIRMED)

        # Attempt to add another item
        add_attempt = agent.execute_tool("add_item_to_order", {"item_name": "Loaded Cheese Fries", "qty": 1})
        self.assertEqual(add_attempt.get("status"), "error")
        self.assertIn("confirmed and locked", add_attempt.get("message", "").lower())
        self.assertEqual(agent.state, STATE_CONFIRMED, "FSM state must stay in STATE_CONFIRMED.")

        # Attempt to remove an item
        remove_attempt = agent.execute_tool("remove_item_from_order", {"item_name": "Perpetual Classic Burger"})
        self.assertEqual(remove_attempt.get("status"), "error")
        self.assertEqual(agent.state, STATE_CONFIRMED)

        # Attempt to change quantity
        qty_attempt = agent.execute_tool("update_item_quantity", {"item_name": "Perpetual Classic Burger", "qty": 3})
        self.assertEqual(qty_attempt.get("status"), "error")
        self.assertEqual(agent.state, STATE_CONFIRMED)

        # Attempt to force transition via set_dialogue_state
        trans_attempt = agent.execute_tool("set_dialogue_state", {"new_state": STATE_TAKING_ORDER})
        self.assertEqual(agent.state, STATE_CONFIRMED, "State transition out of CONFIRMED must be locked.")

    def test_07_allergen_query_never_mutates_cart(self):
        """Edge Case 7: Inquiring about allergens must return allergy info and NOT mutate cart."""
        agent = BistroFSMAgent(table_no=4, db_path=self.temp_db.name, language="english")
        res = agent.execute_tool("check_item_allergen", {"item_name": "Perpetual Classic Burger", "allergen": "Dairy"})
        
        self.assertTrue(res.get("contains_allergen"))
        self.assertEqual(agent.state, STATE_MENU_BROWSING)
        
        cart = db_tools.view_current_order(agent.session_id, db_path=self.temp_db.name)
        self.assertEqual(cart["item_count"], 0, "Allergen inquiry must never add items to cart.")

    def test_08_creator_attribution(self):
        """Edge Case 8: Creator questions must identify Christian Carvajal and John Miko Sarsalijo."""
        agent = BistroFSMAgent(table_no=5, db_path=self.temp_db.name, language="english")
        resp = agent.interact("who created you?")
        resp_lower = resp.lower()
        
        self.assertTrue(
            "carvajal" in resp_lower or "christian" in resp_lower,
            f"Expected response to mention Christian Carvajal, got: {resp}"
        )
        self.assertTrue(
            "sarsalijo" in resp_lower or "miko" in resp_lower or "john miko" in resp_lower,
            f"Expected response to mention John Miko Sarsalijo, got: {resp}"
        )


if __name__ == "__main__":
    unittest.main()
