"""
db_tools.py
Database tool definitions and function bindings for Perpetual Bistro Order Agent.
Handles SQLite operations, transactional integrity, stock deduction, and allergen queries.
"""

import os
import sqlite3
import json
from typing import Dict, Any, List, Optional
import uuid

_BASE_DIR = os.path.dirname(__file__)
_RESOURCE_DB = os.path.join(_BASE_DIR, "resources", "restaurant.db")
_ROOT_DB = os.path.join(_BASE_DIR, "restaurant.db")

if os.path.exists(_RESOURCE_DB):
    DB_PATH = _RESOURCE_DB
elif os.path.exists(_ROOT_DB):
    DB_PATH = _ROOT_DB
elif os.path.isdir(os.path.join(_BASE_DIR, "resources")):
    DB_PATH = _RESOURCE_DB
else:
    DB_PATH = _ROOT_DB

SQL_INIT_PATH = os.path.join(_BASE_DIR, "init_restaurant_db.sql")


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database with dictionary rows and ensures tables exist."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Ensure kitchen_tickets table exists
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kitchen_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT UNIQUE NOT NULL,
            table_no INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            items_json TEXT NOT NULL,
            subtotal REAL NOT NULL,
            payment_method TEXT NOT NULL DEFAULT 'CASH',
            payment_ref TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'PREPARING',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    return conn


def init_db(db_path: str = DB_PATH, sql_path: str = SQL_INIT_PATH, reset: bool = True) -> None:
    """Initializes the database schema using init_restaurant_db.sql."""
    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"SQL init file not found at {sql_path}")
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_script = f.read()
    with get_db_connection(db_path) as conn:
        if reset:
            conn.execute("DROP VIEW IF EXISTS cart_items")
            conn.execute("DROP TABLE IF EXISTS order_line_items")
            conn.execute("DROP TABLE IF EXISTS kitchen_tickets")
            conn.execute("DROP TABLE IF EXISTS table_sessions")
            conn.execute("DROP TABLE IF EXISTS menu_items")
            conn.commit()
        conn.executescript(sql_script)
        conn.commit()


def create_or_get_session(table_no: int, session_id: Optional[str] = None, db_path: str = DB_PATH) -> Dict[str, Any]:
    """Creates a new table session or retrieves an existing active session."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if session_id:
            cursor.execute("SELECT * FROM table_sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)

        new_session_id = session_id or str(uuid.uuid4())[:8]
        cursor.execute(
            """
            INSERT OR REPLACE INTO table_sessions (session_id, table_no, status, updated_at)
            VALUES (?, ?, 'STATE_GREETING', CURRENT_TIMESTAMP)
            """,
            (new_session_id, table_no),
        )
        conn.commit()
        return {"session_id": new_session_id, "table_no": table_no, "status": "STATE_GREETING"}


def set_session_status(session_id: str, new_status: str, db_path: str = DB_PATH) -> bool:
    """Updates the FSM state stored in the table_sessions table."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE table_sessions SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (new_status, session_id),
        )
        conn.commit()
        return cursor.rowcount > 0


# Alias for compatibility
update_session_state = set_session_status



def get_all_categories(db_path: str = DB_PATH) -> List[str]:
    """Returns a list of distinct menu categories."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT category FROM menu_items ORDER BY category")
        return [row["category"] for row in cursor.fetchall()]


def _find_menu_item(item_query: str, conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    """Helper to locate a menu item by exact name or substring matching."""
    cursor = conn.cursor()
    # 1. Exact match (case insensitive)
    cursor.execute("SELECT * FROM menu_items WHERE LOWER(name) = LOWER(?)", (item_query.strip(),))
    row = cursor.fetchone()
    if row:
        return row

    # 2. Substring match
    cursor.execute("SELECT * FROM menu_items WHERE LOWER(name) LIKE LOWER(?) ORDER BY LENGTH(name) ASC LIMIT 1", (f"%{item_query.strip()}%",))
    return cursor.fetchone()


# =====================================================================
# Professor-Specified Tool Functions (bound directly to local LLM)
# =====================================================================

def get_menu_by_category(category: Optional[str] = None, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Retrieves menu items filtered by category.
    If category is None, empty, or 'all', returns the full bistro menu grouped by category.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if not category or category.strip().lower() in ["all", "everything", "menu"]:
            cursor.execute("SELECT id, name, category, price, stock_qty, allergens, description FROM menu_items ORDER BY category, name")
        else:
            cursor.execute(
                "SELECT id, name, category, price, stock_qty, allergens, description FROM menu_items WHERE LOWER(category) = LOWER(?) OR LOWER(category) LIKE LOWER(?) ORDER BY name",
                (category.strip(), f"%{category.strip()}%"),
            )
        rows = cursor.fetchall()
        items = [
            {
                "id": r["id"],
                "name": r["name"],
                "category": r["category"],
                "price": r["price"],
                "stock_qty": r["stock_qty"],
                "in_stock": r["stock_qty"] > 0,
                "allergens": r["allergens"],
                "description": r["description"],
            }
            for r in rows
        ]
        return {
            "status": "success",
            "category_queried": category or "all",
            "item_count": len(items),
            "items": items,
        }


def get_menu_items(category: Optional[str] = None, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves list of menu item dictionaries, optionally filtered by category."""
    res = get_menu_by_category(category, db_path=db_path)
    return res.get("items", [])


def check_item_allergen(item_name: str, allergen: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Checks if a menu item contains a specific allergen (e.g., dairy, gluten, nuts, egg, pork, shellfish).
    """
    with get_db_connection(db_path) as conn:
        row = _find_menu_item(item_name, conn)
        if not row:
            return {
                "status": "not_found",
                "message": f"Menu item '{item_name}' was not found in Perpetual Bistro catalog.",
                "contains_allergen": False,
            }

        allergens_str = (row["allergens"] or "").lower()
        query_allergen = allergen.strip().lower()
        contains = query_allergen in allergens_str

        return {
            "status": "success",
            "item_name": row["name"],
            "allergen_queried": allergen,
            "declared_allergens": row["allergens"],
            "contains_allergen": contains,
            "warning": (
                f"Caution: {row['name']} CONTAINS {allergen.upper()} ({row['allergens']})."
                if contains
                else f"Safe: {row['name']} does NOT list {allergen} in its recorded allergen profile."
            ),
        }


def check_availability(item_name: str, qty: int = 1, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Verifies real-time stock availability for a given menu item and quantity.
    """
    with get_db_connection(db_path) as conn:
        row = _find_menu_item(item_name, conn)
        if not row:
            return {
                "status": "not_found",
                "message": f"Menu item '{item_name}' does not exist in the bistro menu.",
                "available": False,
            }

        stock = row["stock_qty"]
        available = stock >= qty
        return {
            "status": "success",
            "item_name": row["name"],
            "requested_qty": qty,
            "current_stock": stock,
            "available": available,
            "message": (
                f"{row['name']} is available ({stock} in stock)."
                if available
                else f"Sorry, {row['name']} is OUT OF STOCK (available: {stock}, requested: {qty})."
            ),
        }


def add_item_to_order(session_id: str, item_name: str, qty: int = 1, instructions: str = "", db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Adds a menu item to the table's active order cart with optional customizations.
    Defensively checks stock availability before mutating order state.
    """
    if qty <= 0:
        return {"status": "error", "success": False, "error": "Quantity must be greater than zero.", "message": "Quantity must be greater than zero."}

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        item = _find_menu_item(item_name, conn)
        if not item:
            return {
                "status": "not_found",
                "success": False,
                "error": f"Item '{item_name}' was not recognized. Please check the bistro menu.",
                "message": f"Item '{item_name}' was not recognized. Please check the bistro menu.",
            }

        # Verify stock level
        if item["stock_qty"] < qty:
            # Out of stock or insufficient stock: recommend alternative from same category
            cursor.execute(
                "SELECT name, price, stock_qty FROM menu_items WHERE category = ? AND id != ? AND stock_qty > 0 LIMIT 2",
                (item["category"], item["id"]),
            )
            alternatives = [dict(r) for r in cursor.fetchall()]
            alt_names = ", ".join([a["name"] for a in alternatives]) if alternatives else "other menu categories"
            return {
                "status": "out_of_stock",
                "success": False,
                "error": f"Unable to add {item['name']}: item is currently out of stock ({item['stock_qty']} available).",
                "item_name": item["name"],
                "stock_available": item["stock_qty"],
                "requested_qty": qty,
                "message": f"Unable to add {item['name']}: item is currently out of stock ({item['stock_qty']} available).",
                "recommended_alternatives": alternatives,
                "suggestion": f"Would you like to try our {alt_names} instead?",
            }

        # Ensure session exists in table_sessions and check confirmation lockout
        cursor.execute("SELECT status FROM table_sessions WHERE session_id = ?", (session_id,))
        sess_row = cursor.fetchone()
        if not sess_row:
            cursor.execute(
                "INSERT OR REPLACE INTO table_sessions (session_id, table_no, status, updated_at) VALUES (?, 1, 'STATE_TAKING_ORDER', CURRENT_TIMESTAMP)",
                (session_id,),
            )
        elif sess_row["status"] == "STATE_CONFIRMED":
            return {
                "status": "error",
                "success": False,
                "error": "Cannot modify order. This tab has already been confirmed and finalized.",
                "message": "Cannot modify order. This tab has already been confirmed and finalized.",
            }

        # Check if identical item with identical instructions already exists in this session
        cursor.execute(
            """
            SELECT id, quantity FROM order_line_items
            WHERE session_id = ? AND item_id = ? AND LOWER(special_instructions) = LOWER(?)
            """,
            (session_id, item["id"], instructions.strip()),
        )
        existing_line = cursor.fetchone()

        if existing_line:
            new_qty = existing_line["quantity"] + qty
            if item["stock_qty"] < new_qty:
                return {
                    "status": "stock_limit_reached",
                    "success": False,
                    "error": f"Cannot add {qty} more of {item['name']}. Current cart has {existing_line['quantity']} and total inventory is {item['stock_qty']}.",
                    "message": f"Cannot add {qty} more of {item['name']}. Current cart has {existing_line['quantity']} and total inventory is {item['stock_qty']}.",
                }
            cursor.execute(
                "UPDATE order_line_items SET quantity = ? WHERE id = ?",
                (new_qty, existing_line["id"]),
            )
        else:
            cursor.execute(
                """
                INSERT INTO order_line_items (session_id, item_id, quantity, special_instructions, unit_price)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, item["id"], qty, instructions.strip(), item["price"]),
            )

        # Update session status to TAKING_ORDER if still in GREETING
        cursor.execute(
            "UPDATE table_sessions SET status = 'STATE_TAKING_ORDER', updated_at = CURRENT_TIMESTAMP WHERE session_id = ? AND status = 'STATE_GREETING'",
            (session_id,),
        )
        conn.commit()

        # Fetch updated cart total
        current_order = view_current_order(session_id, db_path=db_path)
        return {
            "status": "success",
            "success": True,
            "message": f"Added {qty}x {item['name']}" + (f" with instructions: '{instructions}'" if instructions else "") + " to your tab.",
            "added_item": {
                "name": item["name"],
                "quantity": qty,
                "unit_price": item["price"],
                "special_instructions": instructions.strip(),
            },
            "new_order_subtotal": current_order["subtotal"],
            "new_order_total": current_order["grand_total"],
            "total_items_in_cart": current_order["item_count"],
        }


def remove_item_from_order(session_id: str, item_name: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Safely removes an item or line item from the active table order.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM table_sessions WHERE session_id = ?", (session_id,))
        sess_row = cursor.fetchone()
        if sess_row and sess_row["status"] == "STATE_CONFIRMED":
            return {
                "status": "error",
                "success": False,
                "error": "Cannot modify order. This tab has already been confirmed and finalized.",
                "message": "Cannot modify order. This tab has already been confirmed and finalized.",
            }

        item = _find_menu_item(item_name, conn)
        if not item:
            return {"status": "not_found", "success": False, "error": f"'{item_name}' is not a recognized menu item.", "message": f"'{item_name}' is not a recognized menu item."}

        cursor.execute(
            "SELECT id, quantity FROM order_line_items WHERE session_id = ? AND item_id = ?",
            (session_id, item["id"]),
        )
        lines = cursor.fetchall()
        if not lines:
            return {
                "status": "not_in_cart",
                "success": False,
                "error": f"'{item['name']}' is not currently in your order cart.",
                "message": f"'{item['name']}' is not currently in your order cart.",
            }

        cursor.execute(
            "DELETE FROM order_line_items WHERE session_id = ? AND item_id = ?",
            (session_id, item["id"]),
        )
        conn.commit()

        updated_order = view_current_order(session_id, db_path=db_path)
        return {
            "status": "success",
            "success": True,
            "message": f"Removed all instances of {item['name']} from your order.",
            "removed_item": item["name"],
            "new_order_total": updated_order["grand_total"],
            "remaining_items": updated_order["item_count"],
        }


def update_item_quantity(session_id: str, item_name: str, qty: int, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Updates the quantity of an item in the active order. If qty <= 0, deletes the item.
    """
    if qty <= 0:
        return remove_item_from_order(session_id, item_name, db_path=db_path)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM table_sessions WHERE session_id = ?", (session_id,))
        sess_row = cursor.fetchone()
        if sess_row and sess_row["status"] == "STATE_CONFIRMED":
            return {
                "status": "error",
                "success": False,
                "error": "Cannot modify order. This tab has already been confirmed and finalized.",
                "message": "Cannot modify order. This tab has already been confirmed and finalized.",
            }

        item = _find_menu_item(item_name, conn)
        if not item:
            return {"status": "not_found", "success": False, "error": f"'{item_name}' was not recognized.", "message": f"'{item_name}' was not recognized."}

        if item["stock_qty"] < qty:
            return {
                "status": "insufficient_stock",
                "success": False,
                "error": f"Cannot set {item['name']} to {qty}: only {item['stock_qty']} are in stock.",
                "message": f"Cannot set {item['name']} to {qty}: only {item['stock_qty']} are in stock.",
            }

        cursor.execute(
            "SELECT id FROM order_line_items WHERE session_id = ? AND item_id = ? LIMIT 1",
            (session_id, item["id"]),
        )
        line = cursor.fetchone()
        if not line:
            return {"status": "not_in_cart", "success": False, "error": f"'{item['name']}' is not in your current order.", "message": f"'{item['name']}' is not in your current order."}

        cursor.execute(
            "UPDATE order_line_items SET quantity = ? WHERE id = ?",
            (qty, line["id"]),
        )
        conn.commit()

        updated_order = view_current_order(session_id, db_path=db_path)
        return {
            "status": "success",
            "success": True,
            "message": f"Updated {item['name']} quantity to {qty}.",
            "new_order_total": updated_order["grand_total"],
        }


def view_current_order(session_id: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Calculates and returns the complete itemized tab for the table session,
    including unit prices, special instruction badges, 12% VAT, and total.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT table_no, status FROM table_sessions WHERE session_id = ?", (session_id,))
        session_row = cursor.fetchone()
        table_no = session_row["table_no"] if session_row else "Unknown"
        status = session_row["status"] if session_row else "STATE_GREETING"

        cursor.execute(
            """
            SELECT oli.id, oli.quantity, oli.special_instructions, oli.unit_price,
                   mi.name, mi.category, mi.allergens
            FROM order_line_items oli
            JOIN menu_items mi ON oli.item_id = mi.id
            WHERE oli.session_id = ?
            ORDER BY oli.id ASC
            """,
            (session_id,),
        )
        lines = cursor.fetchall()

        items = []
        subtotal = 0.0
        total_qty = 0

        for line in lines:
            line_total = line["quantity"] * line["unit_price"]
            subtotal += line_total
            total_qty += line["quantity"]
            items.append({
                "line_id": line["id"],
                "name": line["name"],
                "category": line["category"],
                "quantity": line["quantity"],
                "unit_price": line["unit_price"],
                "line_total": round(line_total, 2),
                "special_instructions": line["special_instructions"] or "",
                "allergens": line["allergens"],
            })

        # Calculations: Subtotal, 12% Government VAT included/standard, Grand Total
        vat_amount = round(subtotal * 0.12, 2)
        grand_total = round(subtotal, 2)  # Standard campus dining retail includes tax in menu prices

        return {
            "status": "success",
            "success": True,
            "session_id": session_id,
            "table_no": table_no,
            "fsm_state": status,
            "item_count": total_qty,
            "items": items,
            "line_items": items,
            "subtotal": round(subtotal, 2),
            "vat_12_pct_breakdown": vat_amount,
            "grand_total": grand_total,
            "currency": "Philippine Pesos (pesos, ₱)",
        }


def confirm_order_and_deduct_stock(session_id: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Transactional order confirmation:
    1. Validates all line items against current inventory.
    2. Atomically decrements menu_items stock_qty in SQLite.
    3. Sets table_sessions status to 'STATE_CONFIRMED'.
    4. Generates an immutable digital checkout receipt.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN TRANSACTION")

            cursor.execute("SELECT table_no, status FROM table_sessions WHERE session_id = ?", (session_id,))
            session = cursor.fetchone()
            if not session:
                # Self-healing: Check if order items exist for this session in order_line_items
                cursor.execute("SELECT COUNT(*) as cnt FROM order_line_items WHERE session_id = ?", (session_id,))
                cnt_row = cursor.fetchone()
                if cnt_row and cnt_row["cnt"] > 0:
                    cursor.execute(
                        "INSERT OR REPLACE INTO table_sessions (session_id, table_no, status, updated_at) VALUES (?, 1, 'STATE_REVIEW', CURRENT_TIMESTAMP)",
                        (session_id,),
                    )
                    cursor.execute("SELECT table_no, status FROM table_sessions WHERE session_id = ?", (session_id,))
                    session = cursor.fetchone()
                else:
                    conn.rollback()
                    return {"status": "error", "success": False, "error": f"Session {session_id} not found.", "message": f"Session {session_id} not found."}

            # Retrieve order line items
            cursor.execute(
                """
                SELECT oli.item_id, oli.quantity, oli.special_instructions, oli.unit_price,
                       mi.name, mi.stock_qty
                FROM order_line_items oli
                JOIN menu_items mi ON oli.item_id = mi.id
                WHERE oli.session_id = ?
                """,
                (session_id,),
            )
            items = cursor.fetchall()
            if not items:
                conn.rollback()
                return {"status": "error", "success": False, "error": "Cannot confirm empty order. Cart has no items.", "message": "Cannot confirm empty order. Cart has no items."}

            # Check stock for all items
            for it in items:
                if it["stock_qty"] < it["quantity"]:
                    conn.rollback()
                    return {
                        "status": "stock_error",
                        "success": False,
                        "error": f"Transactional failure: '{it['name']}' only has {it['stock_qty']} in stock, but {it['quantity']} was ordered.",
                        "message": f"Transactional failure: '{it['name']}' only has {it['stock_qty']} in stock, but {it['quantity']} was ordered.",
                    }

            # Atomically decrement stock
            for it in items:
                cursor.execute(
                    "UPDATE menu_items SET stock_qty = stock_qty - ? WHERE id = ?",
                    (it["quantity"], it["item_id"]),
                )

            # Update session state to STATE_CONFIRMED
            cursor.execute(
                "UPDATE table_sessions SET status = 'STATE_CONFIRMED', updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,),
            )

            conn.commit()

            # Compile final receipt
            tab = view_current_order(session_id, db_path=db_path)
            receipt_id = f"PB-{session['table_no']}-{str(uuid.uuid4())[:6].upper()}"

            # Automatically dispatch to Kitchen Display System (KDS)
            create_kitchen_ticket(
                table_no=session["table_no"],
                session_id=session_id,
                items=tab["line_items"],
                grand_total=tab["grand_total"],
                payment_method="CASH",
                payment_ref="COUNTER",
                ticket_id=receipt_id,
                db_path=db_path
            )

            return {
                "status": "success",
                "success": True,
                "receipt_id": receipt_id,
                "table_no": session["table_no"],
                "session_id": session_id,
                "fsm_state": "STATE_CONFIRMED",
                "message": "Order successfully finalized and confirmed! Kitchen ticket dispatched.",
                "itemized_receipt": tab["line_items"],
                "subtotal": tab["subtotal"],
                "vat_included": tab["vat_12_pct_breakdown"],
                "grand_total": tab["grand_total"],
                "currency": "Philippine Pesos (pesos, ₱)",
            }
        except Exception as e:
            conn.rollback()
            return {"status": "error", "success": False, "error": f"Database transaction aborted: {str(e)}", "message": f"Database transaction aborted: {str(e)}"}


# =====================================================================
# KITCHEN DISPLAY SYSTEM (KDS) & INVENTORY ANALYTICS EXTENSIONS
# =====================================================================

def create_kitchen_ticket(
    table_no: int,
    session_id: str,
    items: List[Dict[str, Any]],
    grand_total: Optional[float] = None,
    subtotal: Optional[float] = None,
    payment_method: str = "CASH",
    payment_ref: str = "",
    ticket_id: Optional[str] = None,
    db_path: str = DB_PATH
) -> Dict[str, Any]:
    """Inserts an active kitchen ticket into the kitchen fulfillment queue."""
    t_id = ticket_id or f"PB-{table_no}-{str(uuid.uuid4())[:6].upper()}"
    items_serialized = json.dumps(items)
    effective_total = subtotal if subtotal is not None else (grand_total if grand_total is not None else 0.0)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO kitchen_tickets 
            (ticket_id, table_no, session_id, items_json, subtotal, payment_method, payment_ref, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'PREPARING', CURRENT_TIMESTAMP)
            """,
            (t_id, table_no, session_id, items_serialized, effective_total, payment_method, payment_ref),
        )
        conn.commit()
    return {
        "status": "success",
        "success": True,
        "ticket_id": t_id,
        "table_no": table_no,
        "payment_method": payment_method,
        "payment_ref": payment_ref,
        "kitchen_status": "PREPARING"
    }


def get_kitchen_tickets(status: Optional[str] = None, status_filter: Optional[str] = None, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves kitchen tickets with parsed item arrays."""
    effective_status = status or status_filter
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if effective_status and effective_status.upper() != "ALL":
            cursor.execute(
                "SELECT * FROM kitchen_tickets WHERE status = ? ORDER BY id DESC",
                (effective_status.upper(),)
            )
        else:
            cursor.execute("SELECT * FROM kitchen_tickets ORDER BY id DESC")
        rows = cursor.fetchall()

    results = []
    for r in rows:
        ticket = dict(r)
        try:
            ticket["items"] = json.loads(ticket.get("items_json") or "[]")
        except Exception:
            ticket["items"] = []
        results.append(ticket)
    return results


def update_kitchen_ticket_status(ticket_id: str, new_status: str, db_path: str = DB_PATH) -> bool:
    """Updates the status of a kitchen ticket ('PREPARING', 'READY', 'SERVED')."""
    valid = ["PREPARING", "READY", "SERVED"]
    clean_status = new_status.upper().strip()
    if clean_status not in valid:
        return False
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE kitchen_tickets SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE ticket_id = ?",
            (clean_status, ticket_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_analytics_summary(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Aggregates business intelligence metrics: total sales, order volume, and inventory health."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. Total revenue & count from kitchen tickets
        cursor.execute("SELECT COUNT(*) as count, COALESCE(SUM(subtotal), 0) as total_rev FROM kitchen_tickets")
        sales_row = cursor.fetchone()
        total_orders = sales_row["count"] if sales_row else 0
        total_revenue = sales_row["total_rev"] if sales_row else 0.0

        # 2. Tickets by status
        cursor.execute("SELECT status, COUNT(*) as count FROM kitchen_tickets GROUP BY status")
        status_counts = {r["status"]: r["count"] for r in cursor.fetchall()}

        # 3. Inventory overview: low stock (<= 5)
        cursor.execute("SELECT name, category, price, stock_qty FROM menu_items ORDER BY stock_qty ASC")
        all_items = [dict(r) for r in cursor.fetchall()]
        low_stock_items = [item for item in all_items if item["stock_qty"] <= 5]

        # 4. Categories sales count
        cursor.execute("SELECT category, COUNT(*) as item_count, AVG(price) as avg_price FROM menu_items GROUP BY category")
        categories_stat = [dict(r) for r in cursor.fetchall()]

    return {
        "status": "success",
        "total_revenue": round(total_revenue, 2),
        "total_orders": total_orders,
        "status_breakdown": status_counts,
        "low_stock_count": len(low_stock_items),
        "low_stock_items": low_stock_items,
        "all_inventory": all_items,
        "category_summary": categories_stat
    }


def restock_menu_item(item_name: str, qty_to_add: int = 10, db_path: str = DB_PATH) -> Dict[str, Any]:
    """Adds stock quantity to a specific menu item."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        item = _find_menu_item(item_name, conn)
        if not item:
            return {"status": "error", "message": f"Item '{item_name}' not found."}
        
        cursor.execute(
            "UPDATE menu_items SET stock_qty = stock_qty + ? WHERE id = ?",
            (qty_to_add, item["id"])
        )
        conn.commit()
        cursor.execute("SELECT stock_qty FROM menu_items WHERE id = ?", (item["id"],))
        updated_stock = cursor.fetchone()["stock_qty"]
    return {
        "status": "success",
        "item_name": item["name"],
        "added_qty": qty_to_add,
        "new_stock": updated_stock,
        "message": f"Successfully added {qty_to_add} stock to {item['name']}. Current stock: {updated_stock}."
    }


def restock_all_items(target_min_qty: int = 20, db_path: str = DB_PATH) -> Dict[str, Any]:
    """Restocks any item below target_min_qty up to target_min_qty."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE menu_items SET stock_qty = ? WHERE stock_qty < ?",
            (target_min_qty, target_min_qty)
        )
        affected = cursor.rowcount
        conn.commit()
    return {
        "status": "success",
        "affected_items": affected,
        "new_minimum": target_min_qty,
        "message": f"Restocked {affected} items to at least {target_min_qty} units."
    }


# Rubric-Mandated Aliases (Lesson 7 Final Project Specification)
def get_cart_summary(session_id: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    """Calculates and returns the active cart summary, itemized line items, and totals."""
    return view_current_order(session_id, db_path=db_path)


def add_item_to_order_tool(session_id: str, item_name: str, qty: int = 1, instructions: str = "", db_path: str = DB_PATH) -> Dict[str, Any]:
    """Applies agentic tool call to add item to table order tab with inventory guardrails."""
    return add_item_to_order(session_id, item_name, qty=qty, instructions=instructions, db_path=db_path)


update_ticket_status = update_kitchen_ticket_status

