"""
test_production_contracts.py
Comprehensive TDD Suite for Perpetual Bistro AI Virtual Waiter Production Contracts
Tests all interfaces, public seams, and defense invariants without internal mocks.
"""

import os
import sys
import uuid
from fastapi.testclient import TestClient

# Ensure ai_virtual_waiter is in python path
WAIT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai_virtual_waiter"))
if WAIT_DIR not in sys.path:
    sys.path.insert(0, WAIT_DIR)

import db_tools
from fsm_controller import (
    BistroFSMAgent,
    DISH_CATALOG,
    find_dishes_in_text,
    STATE_GREETING,
    STATE_TAKING_ORDER,
    STATE_CONFIRMED,
)
from app import app

client = TestClient(app)


# =====================================================================
# SLICE 1: FSM AGENT & CATALOG INTERACTION SEAMS
# =====================================================================

def test_fsm_agent_has_interact_method():
    """BistroFSMAgent must expose an interact(user_input: str) -> str method for conversational controllers."""
    agent = BistroFSMAgent(table_no=1)
    assert hasattr(agent, "interact"), "BistroFSMAgent is missing interact() method!"
    response = agent.interact("Hello BistroBot")
    assert isinstance(response, str), "interact() must return a string response!"
    assert len(response.strip()) > 0, "interact() returned an empty response!"


def test_catalog_executive_chefs_special_duo_matches_sqlite():
    """The Executive Chefs' Special Duo in DISH_CATALOG must match the exact SQLite menu_items seed name."""
    # Verify entity detection in DISH_CATALOG
    detected = find_dishes_in_text("I would like to order the Executive Chefs' Special Duo please")
    assert len(detected) > 0, "Executive Chefs' Special Duo was not detected from text!"
    dish, qty, _ = detected[0]
    assert dish["name"] == "Executive Chefs' Special Duo", f"Expected 'Executive Chefs\\' Special Duo', got '{dish['name']}'"
    
    # Verify item exists in SQLite
    with db_tools.get_db_connection() as conn:
        item = db_tools._find_menu_item(dish["name"], conn)
        assert item is not None, f"Item '{dish['name']}' not found in SQLite database!"
        assert item["name"] == "Executive Chefs' Special Duo"


def test_post_confirmation_hard_lockout_in_database():
    """Direct database tool calls must reject item additions or edits once order is in STATE_CONFIRMED."""
    session_id = f"test_lock_{uuid.uuid4().hex[:6]}"
    db_tools.create_or_get_session(table_no=1, session_id=session_id)
    
    # Add a valid item
    res_add = db_tools.add_item_to_order(session_id, "Perpetual Classic Burger", qty=1)
    assert res_add.get("success") is True, f"Failed to add initial item: {res_add}"
    
    # Confirm the order
    res_confirm = db_tools.confirm_order_and_deduct_stock(session_id)
    assert res_confirm.get("success") is True, f"Checkout failed: {res_confirm}"
    
    # Verify session is in STATE_CONFIRMED
    sess = db_tools.create_or_get_session(table_no=1, session_id=session_id)
    assert sess.get("status") == STATE_CONFIRMED
    
    # Attempt to add an item post-confirmation -> MUST BE REJECTED
    res_post_add = db_tools.add_item_to_order(session_id, "Golden French Fries", qty=1)
    assert res_post_add.get("success") is False, "Database allowed adding an item to a STATE_CONFIRMED order!"
    assert "confirmed" in res_post_add.get("error", "").lower() or "confirmed" in res_post_add.get("message", "").lower()
    
    # Attempt to remove an item post-confirmation -> MUST BE REJECTED
    res_post_rem = db_tools.remove_item_from_order(session_id, "Perpetual Classic Burger")
    assert res_post_rem.get("success") is False, "Database allowed removing an item from a STATE_CONFIRMED order!"
    
    # Attempt to update quantity post-confirmation -> MUST BE REJECTED
    res_post_upd = db_tools.update_item_quantity(session_id, "Perpetual Classic Burger", qty=5)
    assert res_post_upd.get("success") is False, "Database allowed updating quantity in a STATE_CONFIRMED order!"


# =====================================================================
# SLICE 2: TABLETOP CART & SESSION REST CONTRACTS
# =====================================================================

def test_api_cart_get_active_session():
    """GET /api/cart must return active table cart without requiring session_id path parameter."""
    res = client.get("/api/cart")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert "line_items" in data or "items" in data, "Cart summary must include line items"
    assert "grand_total" in data, "Cart summary must include grand_total"


def test_api_cart_add_kiosk_payload():
    """POST /api/cart/add must accept kiosk payload with qty and omitted session_id, returning data.cart."""
    payload = {
        "item_name": "Perpetual Classic Burger",
        "qty": 2,
        "instructions": "Extra crispy lettuce"
    }
    res = client.post("/api/cart/add", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data.get("status") == "success"
    assert "cart" in data, "Response must include 'cart' for kiosk UI rendering!"
    assert data["cart"]["grand_total"] >= 300.0, "Subtotal must reflect 2 burgers!"
    assert "fsm_state" in data, "Response must include 'fsm_state'!"


def test_api_cart_update_kiosk_payload():
    """POST /api/cart/update must accept kiosk payload with qty and return updated cart."""
    payload = {
        "item_name": "Perpetual Classic Burger",
        "qty": 3
    }
    res = client.post("/api/cart/update", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data.get("status") == "success"
    assert "cart" in data, "Response must include 'cart'!"
    assert data["cart"]["grand_total"] >= 450.0


def test_api_cart_remove_kiosk_payload():
    """POST /api/cart/remove must accept kiosk payload and return updated cart."""
    payload = {
        "item_name": "Perpetual Classic Burger",
        "qty": 0
    }
    res = client.post("/api/cart/remove", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data.get("status") == "success"
    assert "cart" in data, "Response must include 'cart'!"


def test_api_table_switch_returns_cart_and_fsm_state():
    """POST /api/table/switch must return table_no, fsm_state, and cart."""
    res = client.post("/api/table/switch", json={"table_no": 2})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data.get("table_no") == 2
    assert "fsm_state" in data, "Must include fsm_state"
    assert "cart" in data, "Must include cart"



# =====================================================================
# SLICE 3: DIALOGUE, LANGUAGE & SPEECH/VOICE REST CONTRACTS
# =====================================================================

def test_api_chat_kiosk_contract():
    """POST /api/chat must return assistant_response, speech_text, tool_calls, cart, and fsm_state."""
    payload = {
        "message": "Hi, what burgers do you recommend?",
        "table_no": 1
    }
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert "assistant_response" in data, "Must return assistant_response for kiosk chat bubbles!"
    assert len(data["assistant_response"]) > 0
    assert "response" in data, "Must include backward-compatible response field!"
    assert "cart" in data, "Must include updated cart!"
    assert "fsm_state" in data, "Must include fsm_state!"


def test_api_language_toggle_contract():
    """POST /api/language must switch persona language and return updated status."""
    res = client.post("/api/language", json={"language": "english"})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data.get("status") == "success"
    assert data.get("language") == "english"

    res2 = client.post("/api/language", json={"language": "filipino"})
    assert res2.status_code == 200
    assert res2.json().get("language") == "filipino"


def test_api_tts_post_json_contract():
    """POST /api/tts must accept JSON payload {text, language} and return audio binary data."""
    payload = {
        "text": "Salamat po sa inyong order para sa Table 1.",
        "language": "filipino"
    }
    res = client.post("/api/tts", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    assert "audio" in res.headers.get("content-type", "").lower()
    assert len(res.content) > 0, "TTS audio response must not be empty!"


def test_api_voice_listen_no_hang():
    """GET /api/voice/listen must return immediately without blocking or prompting terminal input."""
    res = client.get("/api/voice/listen")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert "status" in data
    assert "transcription" in data



# =====================================================================
# SLICE 4: PAYMENT, KDS & ADMIN REST CONTRACTS
# =====================================================================

def test_api_payment_checkout_kiosk_payload():
    """POST /api/payment/checkout must accept kiosk payload without session_id and return ticket, receipt, cart, and fsm_state."""
    # Ensure active tab has at least one item
    client.post("/api/cart/add", json={"item_name": "Perpetual Classic Burger", "qty": 1})
    
    payload = {
        "payment_method": "GCASH",
        "payment_ref": "GC-999888",
        "student_id": "2023-01842-LP",
        "table_no": 1
    }
    res = client.post("/api/payment/checkout", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data.get("status") == "success"
    assert "ticket" in data, "Must return kitchen ticket!"
    assert "receipt" in data, "Must return checkout receipt!"
    assert "cart" in data, "Must return updated cart for rendering!"
    assert data.get("fsm_state") == "STATE_CONFIRMED"


def test_api_kds_status_update():
    """POST /api/kds/status must accept both status and new_status fields."""
    tickets_res = client.get("/api/kds/tickets")
    tickets = tickets_res.json().get("tickets", [])
    assert len(tickets) > 0, "No kitchen tickets found to test status update!"
    tid = tickets[0]["ticket_id"]

    # Kiosk sends {ticket_id, status: "READY"}
    res = client.post("/api/kds/status", json={"ticket_id": tid, "status": "READY"})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data.get("status") == "success"
    assert data.get("new_status") == "READY"


def test_api_admin_analytics():
    """GET /api/admin/analytics must return total_revenue, total_orders, and all_inventory."""
    res = client.get("/api/admin/analytics")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert "total_revenue" in data
    assert "total_orders" in data
    assert "all_inventory" in data
    assert len(data["all_inventory"]) == 21


def test_api_admin_restock():
    """POST /api/admin/restock must restock an individual item or all items."""
    # Restock single item
    res1 = client.post("/api/admin/restock", json={"item_name": "Crispy Buffalo Chicken Tenders", "qty": 10})
    assert res1.status_code == 200, f"Expected 200, got {res1.status_code}: {res1.text}"
    data1 = res1.json()
    assert data1.get("status") == "success"

    # Restock all
    res2 = client.post("/api/admin/restock", json={"restock_all": True, "target_min": 20})
    assert res2.status_code == 200, f"Expected 200, got {res2.status_code}: {res2.text}"
    data2 = res2.json()
    assert data2.get("status") == "success"


if __name__ == "__main__":
    print("\n--- RUNNING SLICE 1 TESTS ---")
    try:
        test_fsm_agent_has_interact_method()
        print("✓ test_fsm_agent_has_interact_method passed")
    except Exception as e:
        print(f"✗ test_fsm_agent_has_interact_method FAILED: {e}")

    try:
        test_catalog_executive_chefs_special_duo_matches_sqlite()
        print("✓ test_catalog_executive_chefs_special_duo_matches_sqlite passed")
    except Exception as e:
        print(f"✗ test_catalog_executive_chefs_special_duo_matches_sqlite FAILED: {e}")

    try:
        test_post_confirmation_hard_lockout_in_database()
        print("✓ test_post_confirmation_hard_lockout_in_database passed")
    except Exception as e:
        print(f"✗ test_post_confirmation_hard_lockout_in_database FAILED: {e}")

    print("\n--- RUNNING SLICE 2 TESTS ---")
    try:
        test_api_cart_get_active_session()
        print("✓ test_api_cart_get_active_session passed")
    except Exception as e:
        print(f"✗ test_api_cart_get_active_session FAILED: {e}")

    try:
        test_api_cart_add_kiosk_payload()
        print("✓ test_api_cart_add_kiosk_payload passed")
    except Exception as e:
        print(f"✗ test_api_cart_add_kiosk_payload FAILED: {e}")

    try:
        test_api_cart_update_kiosk_payload()
        print("✓ test_api_cart_update_kiosk_payload passed")
    except Exception as e:
        print(f"✗ test_api_cart_update_kiosk_payload FAILED: {e}")

    try:
        test_api_cart_remove_kiosk_payload()
        print("✓ test_api_cart_remove_kiosk_payload passed")
    except Exception as e:
        print(f"✗ test_api_cart_remove_kiosk_payload FAILED: {e}")

    try:
        test_api_table_switch_returns_cart_and_fsm_state()
        print("✓ test_api_table_switch_returns_cart_and_fsm_state passed")
    except Exception as e:
        print(f"✗ test_api_table_switch_returns_cart_and_fsm_state FAILED: {e}")

    print("\n--- RUNNING SLICE 3 TESTS ---")
    try:
        test_api_chat_kiosk_contract()
        print("✓ test_api_chat_kiosk_contract passed")
    except Exception as e:
        print(f"✗ test_api_chat_kiosk_contract FAILED: {e}")

    try:
        test_api_language_toggle_contract()
        print("✓ test_api_language_toggle_contract passed")
    except Exception as e:
        print(f"✗ test_api_language_toggle_contract FAILED: {e}")

    try:
        test_api_tts_post_json_contract()
        print("✓ test_api_tts_post_json_contract passed")
    except Exception as e:
        print(f"✗ test_api_tts_post_json_contract FAILED: {e}")

    try:
        test_api_voice_listen_no_hang()
        print("✓ test_api_voice_listen_no_hang passed")
    except Exception as e:
        print(f"✗ test_api_voice_listen_no_hang FAILED: {e}")

    print("\n--- RUNNING SLICE 4 TESTS ---")
    try:
        test_api_payment_checkout_kiosk_payload()
        print("✓ test_api_payment_checkout_kiosk_payload passed")
    except Exception as e:
        print(f"✗ test_api_payment_checkout_kiosk_payload FAILED: {e}")

    try:
        test_api_kds_status_update()
        print("✓ test_api_kds_status_update passed")
    except Exception as e:
        print(f"✗ test_api_kds_status_update FAILED: {e}")

    try:
        test_api_admin_analytics()
        print("✓ test_api_admin_analytics passed")
    except Exception as e:
        print(f"✗ test_api_admin_analytics FAILED: {e}")

    try:
        test_api_admin_restock()
        print("✓ test_api_admin_restock passed")
    except Exception as e:
        print(f"✗ test_api_admin_restock FAILED: {e}")
