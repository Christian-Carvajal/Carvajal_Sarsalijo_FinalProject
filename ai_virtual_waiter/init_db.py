"""
init_db.py
Database Initialization Script for Perpetual Bistro
Course: BSCS 3112 - Artificial Intelligence (Lesson 7 Final Project)
University of Perpetual Help System DALTA

Executes restaurant_db.sql to populate restaurant.db with tables:
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
_CANDIDATE_SQL = os.path.join(BASE_DIR, "restaurant_db.sql")
SQL_PATH = _CANDIDATE_SQL if os.path.exists(_CANDIDATE_SQL) else os.path.join(BASE_DIR, "init_restaurant_db.sql")


def init_database(db_path: str = DB_PATH, sql_path: str = SQL_PATH, reset: bool = False) -> None:
    """Initializes SQLite database using SQL schema file."""
    if reset and os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"[INIT] Existing database removed for fresh start: {db_path}")
        except PermissionError:
            print(f"[INIT] Database file active in server process. Executing clean table drop and reseed.")

    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"SQL initialization schema not found at: {sql_path}")

    print(f"[INIT] Reading SQL schema from: {sql_path}")
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_script = f.read()

    print(f"[INIT] Connecting to SQLite at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.executescript(sql_script)
        conn.commit()

        # Audit checks
        cursor.execute("SELECT COUNT(*) FROM menu_items;")
        total_items = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT category) FROM menu_items;")
        total_cats = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM menu_items WHERE stock_qty = 0;")
        oos_items = cursor.fetchone()[0]

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]

        print(f"\n==============================================================")
        print(f" [SUCCESS] Perpetual Bistro Database Initialized Successfully")
        print(f"==============================================================")
        print(f" • Database File:      {db_path}")
        print(f" • Total Menu Items:   {total_items}")
        print(f" • Active Categories:  {total_cats} (Burgers, Pastas, Sides, Beverages, Desserts, Combos)")
        print(f" • Out-of-Stock Items: {oos_items} (Prepared for edge-case defense)")
        print(f" • State Engine Table: table_sessions (Ready for FSM)")
        print(f" • Active Cart Table:  order_line_items / cart_items (Ready)")
        print(f" • Live KDS Table:     kitchen_tickets (Ready)")
        print(f"==============================================================\n")

    except sqlite3.Error as e:
        print(f"[INIT ERROR] Failed to initialize SQLite database: {e}")
        conn.rollback()
        raise e
    finally:
        conn.close()


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    init_database(reset=reset_flag)
