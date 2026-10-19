"""
init_db.py
Database Initialization Script for Perpetual Bistro
Course: BSCS 3112 - Artificial Intelligence (Lesson 7 Final Project)
University of Perpetual Help System DALTA

Executes init_restaurant_db.sql to populate restaurant.db with tables:
- menu_items (21 items including campus combos and stock counts)
- table_sessions (FSM conversational state machine tracking)
- order_line_items & cart_items (active order tabs and modifications)
- kitchen_tickets (live KDS fulfillment queue)
"""

import os
import sys
import sqlite3

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "restaurant.db")
SQL_PATH = os.path.join(BASE_DIR, "init_restaurant_db.sql")


def init_database(db_path: str = DB_PATH, sql_path: str = SQL_PATH, reset: bool = False) -> None:
    """Initializes SQLite database using SQL schema file."""
    if reset and os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"[INIT] Existing database removed for fresh start: {db_path}")
        except PermissionError:
            print(f"[INIT] Database file active in server process. Executing clean table drop and reseed.")

    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"SQL script not found at {sql_path}")

    print(f"[INIT] Reading SQL schema from: {sql_path}")
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_script = f.read()

    print(f"[INIT] Connecting to SQLite at: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        if reset:
            cursor.execute("DROP VIEW IF EXISTS cart_items")
            cursor.execute("DROP TABLE IF EXISTS order_line_items")
            cursor.execute("DROP TABLE IF EXISTS kitchen_tickets")
            cursor.execute("DROP TABLE IF EXISTS table_sessions")
            cursor.execute("DROP TABLE IF EXISTS menu_items")
            conn.commit()
        cursor.executescript(sql_script)
        conn.commit()

        # Audit verification
        cursor.execute("SELECT COUNT(*) as count FROM menu_items")
        item_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as oos_count FROM menu_items WHERE stock_qty = 0")
        oos_count = cursor.fetchone()["oos_count"]

        cursor.execute("SELECT COUNT(DISTINCT category) as cat_count FROM menu_items")
        cat_count = cursor.fetchone()["cat_count"]

        print(f"\n==============================================================")
        print(f" [SUCCESS] Perpetual Bistro Database Initialized Successfully")
        print(f"==============================================================")
        print(f" • Database File:      {db_path}")
        print(f" • Total Menu Items:   {item_count}")
        print(f" • Active Categories:  {cat_count} (Burgers, Pastas, Sides, Beverages, Desserts, Combos)")
        print(f" • Out-of-Stock Items: {oos_count} (Prepared for edge-case defense)")
        print(f" • State Engine Table: table_sessions (Ready for FSM)")
        print(f" • Active Cart Table:  order_line_items / cart_items (Ready)")
        print(f" • Live KDS Table:     kitchen_tickets (Ready)")
        print(f"==============================================================\n")
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Database initialization failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    reset_db = "--reset" in sys.argv
    init_database(reset=reset_db)
