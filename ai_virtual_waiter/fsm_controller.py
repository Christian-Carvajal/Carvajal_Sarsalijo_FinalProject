"""
fsm_agent.py
Conversational Finite State Machine (FSM) Controller and Dynamic Tool Calling Engine.
Powered by local Ollama (qwen3.5:4b) and SQLite (restaurant.db).
"""

import json
import os
import re
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple

import db_tools

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/chat")

def select_best_model() -> str:
    """Dynamically detects best available model without exceeding laptop VRAM limits."""
    if "OLLAMA_MODEL" in os.environ:
        return os.environ["OLLAMA_MODEL"]
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            installed = [m.get("name", "") for m in data.get("models", [])]
            # Prioritize qwen2.5:7b (recommended by rubric for robust tool calling & natural Taglish)
            for cand in ["qwen2.5:7b", "qwen2.5:3b", "qwen2.5:1.5b", "qwen3.5:4b"]:
                for m in installed:
                    if cand in m:
                        return m
            if installed:
                return installed[0]
    except Exception:
        pass
    return "qwen2.5:7b"

DEFAULT_MODEL = select_best_model()

# 5 Discrete FSM States
STATE_GREETING = "STATE_GREETING"
STATE_MENU_BROWSING = "STATE_MENU_BROWSING"
STATE_TAKING_ORDER = "STATE_TAKING_ORDER"
STATE_REVIEW = "STATE_REVIEW"
STATE_CONFIRMED = "STATE_CONFIRMED"

# Milestone 1 Standard State Aliases
GREETING = STATE_GREETING
ORDERING = STATE_TAKING_ORDER
REVIEW = STATE_REVIEW
CONFIRMED = STATE_CONFIRMED

# Tool JSON schemas for Ollama Function Calling
BISTRO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_menu_by_category",
            "description": "Retrieve menu items from Perpetual Bistro, optionally filtered by category (Burgers, Pastas, Sides, Beverages, Desserts, or 'all').",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category name such as Burgers, Pastas, Sides, Beverages, Desserts, or 'all'."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_item_allergen",
            "description": "Check if a menu item contains a specific allergen like Dairy, Gluten, Nuts, Egg, Pork, or Shellfish.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Name of the bistro menu item."
                    },
                    "allergen": {
                        "type": "string",
                        "description": "Specific allergen to verify."
                    }
                },
                "required": ["item_name", "allergen"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Verify current inventory and stock availability for a menu item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Name of the menu item."
                    },
                    "qty": {
                        "type": "integer",
                        "description": "Quantity requested (defaults to 1)."
                    }
                },
                "required": ["item_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_item_to_order",
            "description": "Add an item to the customer's active order tab in SQLite with specified quantity and special instructions/customizations. Always call this when customer wants to buy, order, or add dishes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Name of the menu item to add (e.g. 'Perpetual Classic Burger', 'Truffle Mushroom Pasta')."
                    },
                    "qty": {
                        "type": "integer",
                        "description": "Quantity to add (e.g. 1, 2)."
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Customer special instructions or dietary modifications (e.g. 'no onions', 'extra spicy')."
                    }
                },
                "required": ["item_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_item_from_order",
            "description": "Remove an item from the customer's active order cart in SQLite. When the customer says 'remove that', 'cancel it', 'alisin yan', or 'the only item in the cart', call this tool with that item name from their active cart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Name of the menu item to remove (e.g. 'Perpetual Classic Burger', or 'that'/'it' if referring to the active cart item)."
                    }
                },
                "required": ["item_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_item_quantity",
            "description": "Modify the quantity of an item already in the order cart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Name of the menu item."
                    },
                    "qty": {
                        "type": "integer",
                        "description": "New desired quantity."
                    }
                },
                "required": ["item_name", "qty"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_current_order",
            "description": "View the customer's current itemized tab, subtotal, tax breakdown, and grand total.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_order_and_deduct_stock",
            "description": "Finalize and commit the order to the database, deduct inventory stock, and generate an official receipt.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clear_cart",
            "description": "Delete all items and completely reset the customer's active order tab in the database when customer says 'clear my order', 'clear cart', 'cancel order', 'i don't want to eat anymore', 'ayaw ko na kumain', or 'alisin lahat'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replace_order_item",
            "description": "Replace or swap an existing item in the customer's cart with another menu item (e.g. 'replace bacon mushroom melt with calamansi ice tea', 'change burger to calamansi juice', 'palitan ang burger ng iced tea').",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_item": {
                        "type": "string",
                        "description": "Name or category of the dish to be replaced from the active cart (e.g. 'Bacon Mushroom Melt Burger', 'burger')."
                    },
                    "new_item": {
                        "type": "string",
                        "description": "Name of the replacement dish to be added (e.g. 'Fresh Calamansi Iced Tea', 'calamansi juice')."
                    },
                    "qty": {
                        "type": "integer",
                        "description": "Quantity of the replacement dish (defaults to 1)."
                    }
                },
                "required": ["old_item", "new_item"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_analytics_summary",
            "description": "Analyze and retrieve executive bistro metrics from SQLite: total revenue, order count, ticket breakdown by status, low-stock warnings, and category statistics.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_kitchen_tickets",
            "description": "Query live kitchen display system (KDS) tickets from the database, optionally filtered by status ('PREPARING', 'READY', 'SERVED', or 'ALL').",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by ticket status ('PREPARING', 'READY', 'SERVED', or 'ALL')."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_kitchen_ticket_status",
            "description": "Update the preparation lifecycle status of a kitchen ticket in SQLite ('PREPARING', 'READY', 'SERVED').",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "Unique ticket ID (e.g. TICKET-1-XXXXXX)."
                    },
                    "new_status": {
                        "type": "string",
                        "enum": ["PREPARING", "READY", "SERVED"],
                        "description": "New kitchen workflow status."
                    }
                },
                "required": ["ticket_id", "new_status"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_removed_items",
            "description": "Retrieve the history of items removed or deleted from the customer's cart during this dining session from SQLite.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_last_order_item",
            "description": "Retrieve the last dish or beverage added to the active cart for this session from SQLite.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_items",
            "description": "Query all currently available items in stock and ready to order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_unavailable_items",
            "description": "Query all menu items that are currently out-of-stock or unavailable in the kitchen in real-time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            "description": "Generate chef recommendations and food/beverage pairings based on current cart contents and in-stock bestsellers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category focus (e.g. 'Combos', 'Burgers', 'Pastas')."
                    }
                },
                "required": []
            }
        }
    }
]

DISH_CATALOG = [
    {
        "name": "Perpetual Classic Burger",
        "category": "Burgers",
        "price": 150.0,
        "allergens": ["Dairy", "Gluten"],
        "aliases": [
            "perpetual classic burger", "classic burger", "perpetual burger",
            "classic beef burger", "perpetual classic", "perpetual burger classic",
            "perpetul burger classic", "perpetul classic", "classic"
        ],
    },
    {
        "name": "Bacon Mushroom Melt Burger",
        "category": "Burgers",
        "price": 185.0,
        "allergens": ["Dairy", "Gluten", "Pork"],
        "aliases": ["bacon mushroom melt burger", "bacon mushroom melt", "bacon mushroom", "bacon burger", "mushroom melt"],
    },
    {
        "name": "Grilled Chicken Pesto Sandwich",
        "category": "Burgers",
        "price": 165.0,
        "allergens": ["Gluten", "Nuts"],
        "aliases": ["grilled chicken pesto sandwich", "chicken pesto sandwich", "pesto sandwich", "chicken pesto", "chicken sandwich"],
    },
    {
        "name": "Crispy Buffalo Chicken Tenders",
        "category": "Burgers",
        "price": 140.0,
        "allergens": ["Gluten"],
        "aliases": ["crispy buffalo chicken tenders", "buffalo chicken tenders", "chicken tenders", "buffalo tenders", "tenders"],
    },
    {
        "name": "Truffle Mushroom Pasta",
        "category": "Pastas",
        "price": 210.0,
        "allergens": ["Dairy", "Gluten"],
        "aliases": ["truffle mushroom pasta", "truffle pasta", "mushroom pasta"],
    },
    {
        "name": "Classic Spaghetti Bolognese",
        "category": "Pastas",
        "price": 175.0,
        "allergens": ["Dairy", "Gluten", "Pork"],
        "aliases": ["classic spaghetti bolognese", "spaghetti bolognese", "bolognese", "spaghetti"],
    },
    {
        "name": "Creamy Carbonara",
        "category": "Pastas",
        "price": 190.0,
        "allergens": ["Dairy", "Gluten", "Egg", "Pork"],
        "aliases": ["creamy carbonara", "carbonara"],
    },
    {
        "name": "Pan-Seared Salmon",
        "category": "Pastas",
        "price": 240.0,
        "allergens": ["Fish", "Dairy"],
        "aliases": ["pan-seared salmon", "pan seared salmon", "seared salmon", "salmon fillet"],
    },
    {
        "name": "Crispy Slice Salmon",
        "category": "Pastas",
        "price": 220.0,
        "allergens": ["Fish", "Gluten"],
        "aliases": ["crispy slice salmon", "crispy salmon", "salmon"],
    },
    {
        "name": "Loaded Cheese Fries",
        "category": "Sides",
        "price": 110.0,
        "allergens": ["Dairy", "Pork"],
        "aliases": ["loaded cheese fries", "cheese fries", "loaded fries"],
    },
    {
        "name": "Golden French Fries",
        "category": "Sides",
        "price": 75.0,
        "allergens": [],
        "aliases": ["golden french fries", "french fries", "fries"],
    },
    {
        "name": "Crispy Onion Rings",
        "category": "Sides",
        "price": 85.0,
        "allergens": ["Gluten"],
        "aliases": ["crispy onion rings", "onion rings"],
    },
    {
        "name": "Garden Fresh Caesar Salad",
        "category": "Sides",
        "price": 130.0,
        "allergens": ["Dairy", "Egg"],
        "aliases": ["garden fresh caesar salad", "caesar salad", "salad"],
    },
    {
        "name": "Iced Caramel Macchiato",
        "category": "Beverages",
        "price": 95.0,
        "allergens": ["Dairy"],
        "aliases": ["iced caramel macchiato", "caramel macchiato", "macchiato", "iced macchiato"],
    },
    {
        "name": "Classic Brewed Coffee",
        "category": "Beverages",
        "price": 60.0,
        "allergens": [],
        "aliases": ["classic brewed coffee", "brewed coffee", "hot coffee", "coffee"],
    },
    {
        "name": "Fresh Calamansi Iced Tea",
        "category": "Beverages",
        "price": 55.0,
        "allergens": [],
        "aliases": ["fresh calamansi iced tea", "calamansi iced tea", "calamansi tea", "iced tea"],
    },
    {
        "name": "Mango Passionfruit Smoothie",
        "category": "Beverages",
        "price": 105.0,
        "allergens": ["Dairy"],
        "aliases": ["mango passionfruit smoothie", "mango smoothie", "passionfruit smoothie", "smoothie"],
    },
    {
        "name": "Molten Lava Chocolate Cake",
        "category": "Desserts",
        "price": 125.0,
        "allergens": ["Dairy", "Gluten", "Egg"],
        "aliases": ["molten lava chocolate cake", "molten lava cake", "chocolate cake", "lava cake"],
    },
    {
        "name": "Classic New York Cheesecake",
        "category": "Desserts",
        "price": 135.0,
        "allergens": ["Dairy", "Gluten", "Egg"],
        "aliases": ["classic new york cheesecake", "new york cheesecake", "cheesecake"],
    },
    {
        "name": "Perpetual Champion Combo",
        "category": "Combos",
        "price": 285.0,
        "allergens": ["Gluten", "Dairy"],
        "aliases": ["perpetual champion combo", "champion combo", "burger combo", "champion meal", "burger and fries combo"],
    },
    {
        "name": "Pasta & Sweet Treat Combo",
        "category": "Combos",
        "price": 330.0,
        "allergens": ["Gluten", "Dairy", "Egg", "Pork"],
        "aliases": ["pasta & sweet treat combo", "pasta and sweet treat combo", "pasta sweet treat combo", "pasta combo", "carbonara combo"],
    },
    {
        "name": "Executive Chefs' Special Duo",
        "category": "Combos",
        "price": 420.0,
        "allergens": ["Gluten", "Dairy", "Pork"],
        "aliases": [
            "executive chefs' special duo",
            "executive chefs special duo",
            "chefs special duo",
            "chef brigade special duo",
            "culinary brigade special duo",
            "chef ezekiel's special duo",
            "chef ezekiel special duo",
            "special duo combo",
            "chef special duo",
            "chef duo",
            "duo combo",
        ],
    },
    {
        "name": "Late Night Study Fuel Combo",
        "category": "Combos",
        "price": 210.0,
        "allergens": ["Dairy", "Gluten"],
        "aliases": ["late night study fuel combo", "study fuel combo", "study fuel", "study combo"],
    },
]

def find_dishes_in_text(text: str) -> List[Tuple[Dict[str, Any], int, str]]:
    """Detects dish entities, quantities, and instructions from natural customer dialogue."""
    results = []
    text_lower = text.lower()
    
    inst_match = re.search(r'(no\s+onions?|walang\s+sibuyas|extra\s+cheese|extra\s+sauce|spicy|dressing\s+on\s+(?:the\s+)?side|gluten-free\s+bun|less\s+ice)', text_lower)
    default_instructions = inst_match.group(1).title() if inst_match else ""
    if "no onions" in text_lower or "walang sibuyas" in text_lower:
        default_instructions = "No Onions"
    if "extra cheese" in text_lower:
        default_instructions = (default_instructions + ", Extra Cheese" if default_instructions else "Extra Cheese")

    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "isa": 1, "dalawa": 2, "tatlo": 3, "apat": 4, "lima": 5,
        "isang": 1, "dalawang": 2, "tatlong": 3, "apat na": 4, "limang": 5,
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    }

    matched_spans = []
    for dish in DISH_CATALOG:
        for alias in sorted(dish["aliases"], key=len, reverse=True):
            pattern = r'\b' + re.escape(alias) + (r'(?:s|es)?\b' if not alias.endswith('s') else r'\b')
            matched = False
            for m in re.finditer(pattern, text_lower):
                start, end = m.span()
                if any(start >= s and end <= e for (s, e) in matched_spans):
                        continue
                matched_spans.append((start, end))

                preceding = text_lower[:start].strip()
                qty = 1
                for w, num in word_to_num.items():
                    if preceding.endswith(w) or f"{w} " in preceding[-15:]:
                        qty = num
                        break

                results.append((dish, qty, default_instructions))
                matched = True
                break
            if matched:
                break

    if not results and len(text.strip()) >= 4:
        try:
            with db_tools.get_db_connection() as conn:
                m_item = db_tools._find_menu_item(text, conn)
                if m_item:
                    target_dish = next((d for d in DISH_CATALOG if d["name"] == m_item["name"]), None)
                    if not target_dish:
                        target_dish = {
                            "name": m_item["name"],
                            "category": m_item["category"],
                            "price": m_item["price"],
                            "allergens": [a.strip() for a in (m_item.get("allergens") or "").split(",") if a.strip()],
                            "aliases": [m_item["name"].lower()],
                        }
                    results.append((target_dish, 1, default_instructions))
        except Exception:
            pass

    return results


def check_ambiguous_category_request(text: str) -> Optional[Dict[str, Any]]:
    """
    Detects if the customer is asking about a general food/drink category rather than a specific dish.
    Returns the category details and dish options to prompt clarification before adding to cart.
    """
    text_lower = text.lower().strip()

    # Specific dish identifiers (if present, the request is not ambiguous)
    has_specific_dish = any(spec in text_lower for spec in [
        "classic", "perpetual", "bacon", "mushroom", "melt", "pesto", "tender", "buffalo",
        "truffle", "bolognese", "spaghetti", "carbonara", "pan-seared", "crispy slice", "seared",
        "loaded", "golden", "onion ring", "caesar", "macchiato", "caramel",
        "brewed", "calamansi", "iced tea", "smoothie", "mango", "lava", "chocolate", "cheesecake",
        "champion", "sweet treat", "executive", "study fuel", "duo"
    ])
    if has_specific_dish:
        return None

    # Check for general order opening
    if re.fullmatch(r"(can\s+i\s+order\??|pwede\s+ba\s+umorder\??|order\s+po\??|i\s+want\s+to\s+order\??|may\s+i\s+order\??|order\??)", text_lower):
        return {
            "type": "general_order",
            "message": "Yes, of course! You can order anytime. What dishes or drinks would you like from our menu? We have delicious burgers, pastas, sides, desserts, beverages, and combo meals!"
        }

    categories = [
        {
            "patterns": [r"\bburgers?\b", r"\bhamburgers?\b"],
            "category": "Burgers",
            "prompt": "Certainly! We offer multiple burgers on our menu: Perpetual Classic Burger (₱150.00) and Bacon Mushroom Melt Burger (₱185.00). Which burger would you like to order?"
        },
        {
            "patterns": [r"\bpastas?\b", r"\bnoodles?\b"],
            "category": "Pastas",
            "prompt": "Sure! Here are our pasta options: Truffle Mushroom Pasta (₱210.00), Classic Spaghetti Bolognese (₱175.00), and Creamy Carbonara (₱190.00). Which one would you prefer?"
        },
        {
            "patterns": [r"\bdrinks?\b", r"\bbeverages?\b", r"\binumin\b", r"\brefreshments?\b"],
            "category": "Beverages",
            "prompt": "We offer several refreshing beverages: Fresh Calamansi Iced Tea (₱55.00), Iced Caramel Macchiato (₱95.00), Classic Brewed Coffee (₱60.00), and Mango Passionfruit Smoothie (₱105.00). Which drink would you like?"
        },
        {
            "patterns": [r"\bcoffees?\b", r"\bkape\b"],
            "category": "Beverages",
            "prompt": "For coffee lovers, we have Classic Brewed Coffee (₱60.00) and Iced Caramel Macchiato (₱95.00). Which one would you like?"
        },
        {
            "patterns": [r"\bfries\b", r"\bfrench\s+fries\b"],
            "category": "Sides",
            "prompt": "We offer two types of fries: Loaded Cheese Fries (₱110.00) and Golden French Fries (₱75.00). Which one would you prefer?"
        },
        {
            "patterns": [r"\bsalmon\b"],
            "category": "Pastas",
            "prompt": "We have two salmon specialties: Pan-Seared Salmon (₱240.00) and Crispy Slice Salmon (₱220.00). Which one would you like to try?"
        },
        {
            "patterns": [r"\bdesserts?\b", r"\bcakes?\b", r"\bmatamis\b"],
            "category": "Desserts",
            "prompt": "For dessert, we offer Molten Lava Chocolate Cake (₱125.00) and Classic New York Cheesecake (₱135.00). Which sweet treat would you like?"
        },
        {
            "patterns": [r"\bcombos?\b", r"\bvalue\s+meals?\b", r"\bbundles?\b"],
            "category": "Combos",
            "prompt": "We have several value combos: Perpetual Champion Combo (₱285.00), Pasta & Sweet Treat Combo (₱330.00), Executive Chefs' Special Duo (₱420.00), and Late Night Study Fuel Combo (₱210.00). Which combo would you like?"
        },
        {
            "patterns": [r"\bsides?\b", r"\bside\s+dish\b"],
            "category": "Sides",
            "prompt": "Here are our side dishes: Loaded Cheese Fries (₱110.00), Golden French Fries (₱75.00), Crispy Onion Rings (₱85.00), and Garden Fresh Caesar Salad (₱130.00). Which side would you like?"
        }
    ]

    for cat in categories:
        for pat in cat["patterns"]:
            if re.search(pat, text_lower):
                return cat

    return None


class BistroFSMAgent:
    """
    Stateful Conversational Finite State Machine Agent.
    Manages session memory, dialogue state transitions, and Ollama tool binding.
    """

    def __init__(
        self,
        table_no: int = 1,
        session_id: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        db_path: str = db_tools.DB_PATH,
        language: str = "filipino",
    ):
        self.table_no = table_no
        self.model = model
        self.db_path = db_path
        self.language = language.lower().strip()
        self.session = db_tools.create_or_get_session(table_no, session_id, db_path=self.db_path)
        self.session_id = self.session["session_id"]
        self.state = self.session["status"] or STATE_GREETING
        self.messages: List[Dict[str, Any]] = []
        self.audit_log: List[Dict[str, Any]] = []
        self._init_system_prompt()

    def _get_live_menu_context(self) -> str:
        """Directly queries SQLite to produce a complete live menu catalog reference."""
        try:
            menu_res = db_tools.get_menu_by_category("all", db_path=self.db_path)
            items = menu_res.get("items", [])
            if not items:
                return "Menu database is currently empty."
            
            categories: Dict[str, List[Dict[str, Any]]] = {}
            for it in items:
                cat = it.get("category", "Other")
                categories.setdefault(cat, []).append(it)
            
            lines = ["LIVE PERPETUAL BISTRO MENU DATABASE (Direct SQLite Catalog):"]
            for cat, cat_items in categories.items():
                lines.append(f"[{cat.upper()}]:")
                for it in cat_items:
                    stock_tag = f"IN STOCK ({it['stock_qty']} avail)" if it.get('in_stock') else "OUT OF STOCK (0 avail - suggest alternatives)"
                    allergens = it.get('allergens', 'None')
                    lines.append(f"  • {it['name']} - ₱{it['price']:.2f} | {stock_tag} | Allergens: {allergens}")
            return "\n".join(lines)
        except Exception as e:
            return f"Menu database error: {e}"

    def _get_live_cart_context(self) -> str:
        """Directly queries SQLite to produce the live cart state, last added item, and removal audit for this session."""
        try:
            cart = db_tools.view_current_order(self.session_id, db_path=self.db_path)
            lines = [f"LIVE ACTIVE CART FOR TABLE #{self.table_no} (Session: {self.session_id}):"]
            items = cart.get("line_items", [])
            if not items:
                lines.append("  • Cart is currently EMPTY (0 items).")
            else:
                lines.append(f"  • Current items ({len(items)} line item(s)):")
                for idx, it in enumerate(items, 1):
                    inst = f" [Note: {it['special_instructions']}]" if it.get('special_instructions') else ""
                    lines.append(f"    {idx}. {it['quantity']}x {it['name']} (₱{it['unit_price']:.2f} each, Line Total: ₱{it['line_total']:.2f}){inst}")
                lines.append(f"  • Subtotal: ₱{cart.get('subtotal', 0.0):.2f} (12% VAT: ₱{cart.get('vat_12_pct_breakdown', 0.0):.2f}) | Grand Total: ₱{cart.get('grand_total', 0.0):.2f}")

            # Last item added to cart
            last_item = db_tools.get_last_order_item(self.session_id, db_path=self.db_path)
            if last_item:
                lines.append(f"  • Last Item Added to Cart: {last_item['quantity']}x {last_item['name']} (₱{last_item['unit_price']:.2f})")
            else:
                lines.append("  • Last Item Added to Cart: None")

            # Removed items audit trail
            removed = db_tools.get_removed_items(self.session_id, db_path=self.db_path)
            if removed:
                rem_descs = [f"{r['quantity']}x {r['item_name']} (₱{r['unit_price']:.2f})" for r in removed[:3]]
                lines.append(f"  • Recently Removed from Cart (SQLite Audit): {', '.join(rem_descs)}")
            else:
                lines.append("  • Recently Removed from Cart (SQLite Audit): None")

            # Out of stock in real-time
            unavail = db_tools.get_unavailable_items(db_path=self.db_path)
            if unavail:
                un_names = [u['name'] for u in unavail]
                lines.append(f"  • Kitchen Real-Time Out-of-Stock (0 stock): {', '.join(un_names)}")

            return "\n".join(lines)
        except Exception as e:
            return f"Cart database error: {e}"

    def _refresh_dynamic_context(self) -> None:
        """Refreshes live cart and FSM state in system prompt before every turn."""
        if not self.messages or self.messages[0]["role"] != "system":
            self._init_system_prompt()
            return

        cart_ctx = self._get_live_cart_context()
        sys_text = self.messages[0]["content"]

        if "LIVE ACTIVE CART FOR TABLE #" in sys_text:
            sys_text = re.sub(
                r"LIVE ACTIVE CART FOR TABLE #\d+.*?(?=LIVE PERPETUAL BISTRO MENU|$)",
                f"{cart_ctx}\n\n",
                sys_text,
                flags=re.DOTALL
            )
        else:
            sys_text = f"{cart_ctx}\n\n" + sys_text

        sys_text = re.sub(
            r"Current FSM State: \w+",
            f"Current FSM State: {self.state}",
            sys_text
        )
        self.messages[0]["content"] = sys_text

    def _init_system_prompt(self) -> None:
        """Constructs the authoritative system prompt with conversational flexibility and guardrails."""
        if self.language == "english":
            language_rule = (
                "6. LANGUAGE & TONE: Speak in fluent, professional, courteous English. "
                "Be welcoming, helpful, and clear. Maintain friendly dining hospitality at all times.\n"
            )
        else:
            language_rule = (
                "6. NATURAL TAGLISH & TONE: Speak in natural, modern conversational Taglish (the casual bilingual English-Tagalog spoken in Metro Manila college campus bistros). "
                "Use English for table numbers and tech names (say 'Table 1', 'Qwen 3.5', '150 pesos', 'Mabuhay po!'). "
                "NEVER use archaic, deep pure Tagalog (never say 'mesa isa' or 'tatlong punto lima'). Be friendly, natural, and welcoming.\n"
            )

        menu_ctx = self._get_live_menu_context()
        cart_ctx = self._get_live_cart_context()

        system_content = (
            "You are 'BistroBot', the intelligent, friendly, conversational AI table-side waiter at Perpetual Bistro. "
            "You were engineered and created by Christian Ezekiel L. Carvajal and John Miko P. Sarsalijo, "
            "3rd-year BS Computer Science (Data Science) students at the University of Perpetual Help System DALTA (UPHSD).\n"
            f"You are actively serving Table #{self.table_no} (Session ID: {self.session_id}).\n\n"
            f"{cart_ctx}\n\n"
            f"{menu_ctx}\n\n"
            "ACTION MANDATE (REAL-TIME DATABASE & FUNCTION CALLING RULES):\n"
            "- When customer wants to buy, order, or add an item (e.g. 'i want to buy perpetual burger', 'add on cart the perpetual classic', 'pabili ng burger', '1 burger and iced tea'): YOU MUST call `add_item_to_order`. Never just say you added it in text without calling the tool!\n"
            "- AMBIGUOUS DISH POLICY: If the customer asks for a generic category without specifying the exact dish (e.g. 'can i order a burger?', 'can i order pasta?', 'order drinks', 'pabili ng kape'), DO NOT arbitrarily add a dish to the cart! Instead, present the available options in that category (e.g. Perpetual Classic Burger vs Bacon Mushroom Melt Burger) and ask which one they prefer.\n"
            "- When customer wants to remove or cancel an item (e.g. 'can you remove that in the cart?', 'remove it', 'alisin yan', 'the only 1 item in the cart'): look at the active cart above and immediately call `remove_item_from_order` with the item name from the cart! Do NOT ask what item to remove if there is only 1 item in the cart; execute the tool immediately!\n"
            "- REPLACING / SUBSTITUTING ITEMS: When customer wants to replace, change, or swap an item in their cart (e.g. 'replace bacon mushroom melt with calamansi ice tea', 'change burger to calamansi juice', 'palitan ang burger ng iced tea'): call `replace_order_item(old_item='...', new_item='...')`.\n"
            "- CLEARING ORDER TAB: When customer says 'clear my order', 'clear cart', 'cancel order', 'i don't want to eat anymore', 'ayaw ko na kumain', 'alisin lahat': call `clear_cart()` immediately.\n"
            "- When customer asks for recommendations, suggestions, or pairings (e.g. 'what can you suggest for me?', 'ano masarap?', 'di ako makapili'): call `get_recommendations()` or recommend crowd favorites directly.\n"
            "- When customer asks for the bill, total, receipt, or cart summary (e.g. 'bill please', 'magkano na bill ko?', 'view cart'): call `view_current_order()`.\n"
            "- When customer confirms, places, or checks out the order (e.g. 'confirm order', 'place order', 'bayaran na', 'paki-order na'): call `confirm_order_and_deduct_stock()`.\n"
            "- When customer asks about dietary allergens (e.g. 'does the burger have dairy?', 'may gluten ba?'): call `check_item_allergen`.\n"
            "- When customer asks about kitchen tickets, status, or business revenue: call `get_kitchen_tickets` or `get_analytics_summary`.\n"
            "- NATURAL CONVERSATIONAL SPEECH: Always converse warmly, thoughtfully, and naturally as an attentive bistro waiter. When executing actions (adding dishes, removing dishes, checking bills, answering questions), NEVER give robotic one-word answers like 'Updated.' or raw status dumps. Acknowledge what was done in full, warm sentences and offer helpful suggestions (e.g. 'I have added 1x Perpetual Classic Burger to your tab! Would you like fries or a refreshing iced tea with that?').\n"
            "- DIETARY & ALLERGEN CARE: When answering allergen or health queries (e.g. lactose intolerance, dairy, gluten), explain clearly which components contain the allergen and offer kitchen adjustments (e.g. removing cheese, dairy-free alternatives).\n"
            "- PRONOUN RESOLUTION: If the customer refers to 'that', 'it', 'the only item', or 'the 1 item in the cart', resolve it to the dish listed in the active cart above.\n"
            "- CREATOR IDENTITY: You were engineered and created by Christian Ezekiel L. Carvajal and John Miko P. Sarsalijo (3rd-year BSCS - Data Science, UPHSD). Always mention both by name when asked who created or developed you.\n"
            "- CURRENCY: All prices are strictly in Philippine Pesos (PHP, ₱, pesos). Never mention foreign currencies.\n"
            + language_rule +
            "- CLEAN SPEECH: Do not use emojis in responses. Speak naturally, warmly, and concisely."
        )
        self.messages = [{"role": "system", "content": system_content}]

    def set_language(self, new_lang: str) -> None:
        """Updates the active persona language ('filipino' or 'english')."""
        self.language = new_lang.lower().strip()
        self._init_system_prompt()

    def set_table_number(self, new_table_no: int) -> None:
        """Updates the active table number and connects to an active session for that table."""
        self.table_no = new_table_no
        with db_tools.get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT session_id, status FROM table_sessions WHERE table_no = ? AND status != 'STATE_CONFIRMED' ORDER BY updated_at DESC LIMIT 1",
                (new_table_no,)
            )
            row = cursor.fetchone()
            if row:
                self.session_id = row["session_id"]
                self.state = row["status"]
            else:
                new_sess = db_tools.create_or_get_session(new_table_no, db_path=self.db_path)
                self.session_id = new_sess["session_id"]
                self.state = new_sess["status"]
        self._init_system_prompt()

    def transition_state(self, new_state: str, reason: str = "") -> None:
        """Executes a formal FSM state transition and persists it in SQLite."""
        if self.state == STATE_CONFIRMED and new_state != STATE_CONFIRMED:
            # LOCKED: Post-confirmation state cannot be altered for this session
            return
        old_state = self.state
        self.state = new_state
        db_tools.set_session_status(self.session_id, new_state, db_path=self.db_path)
        transition_record = {
            "from_state": old_state,
            "to_state": new_state,
            "reason": reason,
            "turn_index": len(self.messages),
        }
        self.audit_log.append({"type": "state_transition", "data": transition_record})
        # Update system prompt to reflect current state
        self.messages[0]["content"] = re.sub(
            r"Current FSM State: \w+",
            f"Current FSM State: {self.state}",
            self.messages[0]["content"]
        )

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches an Ollama tool call to the corresponding db_tools function."""
        # Sanitize arguments and inject session_id where required
        if tool_name == "set_dialogue_state":
            new_st = args.get("new_state", self.state)
            reason = args.get("reason", "Model-directed state transition")
            self.transition_state(new_st, reason=reason)
            return {"status": "success", "current_state": self.state}

        elif tool_name == "get_menu_by_category":
            category = args.get("category")
            if self.state == STATE_GREETING:
                self.transition_state(STATE_MENU_BROWSING, reason="Customer inquiring about menu")
            return db_tools.get_menu_by_category(category, db_path=self.db_path)

        elif tool_name == "check_item_allergen":
            item = args.get("item_name", "")
            allergen = args.get("allergen", "")
            if self.state == STATE_GREETING:
                self.transition_state(STATE_MENU_BROWSING, reason="Customer allergen query")
            return db_tools.check_item_allergen(item, allergen, db_path=self.db_path)

        elif tool_name == "check_availability":
            item = args.get("item_name", "")
            qty = int(args.get("qty", 1))
            return db_tools.check_availability(item, qty, db_path=self.db_path)

        elif tool_name == "add_item_to_order":
            if self.state == STATE_CONFIRMED:
                return {
                    "status": "error",
                    "success": False,
                    "error": "Order is already CONFIRMED and locked. Cannot modify items on a finalized ticket.",
                    "message": "Order is already CONFIRMED and locked. Cannot modify items on a finalized ticket."
                }
            item = args.get("item_name", "")
            qty = int(args.get("qty", 1))
            instructions = args.get("instructions", "")
            res = db_tools.add_item_to_order(self.session_id, item, qty, instructions, db_path=self.db_path)
            if res.get("status") == "success":
                self.transition_state(STATE_TAKING_ORDER, reason=f"Item added: {item}")
            return res

        elif tool_name == "remove_item_from_order":
            if self.state == STATE_CONFIRMED:
                return {
                    "status": "error",
                    "success": False,
                    "error": "Order is already CONFIRMED and locked. Cannot modify items on a finalized ticket.",
                    "message": "Order is already CONFIRMED and locked. Cannot modify items on a finalized ticket."
                }
            item = args.get("item_name") or args.get("dish_name") or ""
            cart_current = db_tools.view_current_order(self.session_id, db_path=self.db_path)
            cart_items = cart_current.get("line_items", [])

            # Get user's latest text
            last_user_text = ""
            for m in reversed(self.messages):
                if m.get("role") == "user":
                    last_user_text = m.get("content", "").lower()
                    break

            generic_pronouns = [
                "that", "it", "this", "item", "the item", "the only item", "only item",
                "the only 1 item in the cart", "the only 1 item", "the 1 item", "cart item",
                "only 1 item in the cart", "only 1 item", "that in the cart", "that in cart",
                "item in the cart", "the item in the cart", "the dish", "dish", "that one",
                "the only one", "yung isa", "yung nasa cart", "alisin yan", "tanggalin yan", ""
            ]
            is_generic = (item.lower().strip() in generic_pronouns or
                          any(p in item.lower() for p in ["only 1", "only item", "that", "this item", "nasa cart", "yung isa", "that one"]) or
                          any(p in last_user_text for p in ["that in the cart", "only 1 item", "the only item", "remove that", "alisin yan", "the only 1"]))

            # Check if 'item' passed by model matches any item actually in the cart
            item_in_cart = False
            for ci in cart_items:
                if ci["name"].lower() == item.lower().strip() or item.lower().strip() in ci["name"].lower():
                    item_in_cart = True
                    item = ci["name"]
                    break

            # If item is not in cart and customer didn't explicitly utter this dish name:
            dish_explicitly_typed = len(item.strip()) >= 3 and item.lower().strip() in last_user_text
            if not item_in_cart and (is_generic or not dish_explicitly_typed):
                if len(cart_items) == 1:
                    item = cart_items[0]["name"]
                elif len(cart_items) > 1:
                    if any(w in last_user_text for w in ["all", "everything", "lahat", "clear"]):
                        item = "all"
                    else:
                        last_i = db_tools.get_last_order_item(self.session_id, db_path=self.db_path).get("last_item")
                        if last_i:
                            item = last_i["name"]
                        else:
                            item = cart_items[-1]["name"]
                elif len(cart_items) == 0:
                    return {
                        "status": "empty_cart",
                        "success": False,
                        "error": "Your cart is currently empty. There is nothing to remove.",
                        "message": "Your cart is currently empty. There is nothing to remove."
                    }

            res = db_tools.remove_item_from_order(self.session_id, item, db_path=self.db_path)
            if res.get("status") == "success":
                self.transition_state(STATE_TAKING_ORDER, reason=f"Item removed: {item}")
            return res

        elif tool_name == "update_item_quantity":
            if self.state == STATE_CONFIRMED:
                return {
                    "status": "error",
                    "success": False,
                    "error": "Order is already CONFIRMED and locked. Cannot modify items on a finalized ticket.",
                    "message": "Order is already CONFIRMED and locked. Cannot modify items on a finalized ticket."
                }
            item = args.get("item_name", "")
            qty = int(args.get("qty", 1))
            res = db_tools.update_item_quantity(self.session_id, item, qty, db_path=self.db_path)
            if res.get("status") == "success":
                self.transition_state(STATE_TAKING_ORDER, reason=f"Quantity updated: {item}")
            return res

        elif tool_name == "view_current_order":
            res = db_tools.view_current_order(self.session_id, db_path=self.db_path)
            if self.state != STATE_CONFIRMED:
                self.transition_state(STATE_REVIEW, reason="Customer reviewing bill tab")
            return res

        elif tool_name == "confirm_order_and_deduct_stock":
            if self.state == STATE_CONFIRMED:
                return {
                    "status": "already_confirmed",
                    "success": False,
                    "error": "This order tab has already been confirmed and paid. Cannot re-confirm.",
                    "message": "This order tab has already been confirmed and paid. Cannot re-confirm."
                }
            db_tools.create_or_get_session(self.table_no, self.session_id, db_path=self.db_path)
            res = db_tools.confirm_order_and_deduct_stock(self.session_id, db_path=self.db_path)
            if res.get("status") == "success":
                self.transition_state(STATE_CONFIRMED, reason="Order committed and inventory deducted")
            return res

        elif tool_name == "clear_cart":
            if self.state == STATE_CONFIRMED:
                return {
                    "status": "error",
                    "success": False,
                    "error": "Cannot modify order. This tab has already been confirmed and finalized.",
                    "message": "Cannot modify order. This tab has already been confirmed and finalized."
                }
            res = db_tools.remove_item_from_order(self.session_id, "all", db_path=self.db_path)
            self.transition_state(STATE_TAKING_ORDER, reason="Customer cleared cart")
            return res

        elif tool_name == "replace_order_item":
            if self.state == STATE_CONFIRMED:
                return {
                    "status": "error",
                    "success": False,
                    "error": "Cannot modify order. This tab has already been confirmed and finalized.",
                    "message": "Cannot modify order. This tab has already been confirmed and finalized."
                }
            old_it = args.get("old_item", "")
            new_it = args.get("new_item", "")
            qty = int(args.get("qty", 1))
            res = db_tools.replace_order_item(self.session_id, old_it, new_it, new_qty=qty, db_path=self.db_path)
            if res.get("status") == "success":
                self.transition_state(STATE_TAKING_ORDER, reason=f"Customer replaced {old_it} with {new_it}")
            return res

        elif tool_name == "get_analytics_summary":
            return db_tools.get_analytics_summary(db_path=self.db_path)

        elif tool_name == "get_kitchen_tickets":
            st = args.get("status")
            return {"status": "success", "tickets": db_tools.get_kitchen_tickets(status=st, db_path=self.db_path)}

        elif tool_name == "update_kitchen_ticket_status":
            t_id = args.get("ticket_id", "")
            n_st = args.get("new_status", "PREPARING")
            ok = db_tools.update_kitchen_ticket_status(t_id, n_st, db_path=self.db_path)
            return {"status": "success" if ok else "error", "ticket_id": t_id, "new_status": n_st}

        elif tool_name == "restock_menu_item":
            item = args.get("item_name", "")
            qty = int(args.get("qty", 10))
            return db_tools.restock_menu_item(item, qty, db_path=self.db_path)

        elif tool_name == "get_removed_items":
            return {"status": "success", "removed_items": db_tools.get_removed_items(self.session_id, db_path=self.db_path)}

        elif tool_name == "get_last_order_item":
            item = db_tools.get_last_order_item(self.session_id, db_path=self.db_path)
            return {"status": "success", "last_item": item}

        elif tool_name == "get_available_items":
            cat = args.get("category")
            return {"status": "success", "available_items": db_tools.get_available_items(category=cat, db_path=self.db_path)}

        elif tool_name == "get_unavailable_items":
            return {"status": "success", "unavailable_items": db_tools.get_unavailable_items(db_path=self.db_path)}

        elif tool_name == "get_recommendations":
            cat = args.get("category")
            return db_tools.get_recommendations(session_id=self.session_id, category=cat, db_path=self.db_path)

        else:
            return {"status": "error", "message": f"Unknown tool name: {tool_name}"}

    def _call_ollama_chat(self, payload: Dict[str, Any], timeout: int = 35) -> Dict[str, Any]:
        """Sends HTTP request to local Ollama /api/chat endpoint with memory-safe fallback."""
        if "keep_alive" not in payload:
            payload["keep_alive"] = "24h"
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=data_bytes,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            if ("out of memory" in err_body.lower() or "cudamalloc" in err_body.lower()) and payload.get("model") != "qwen2.5:1.5b":
                retry_payload = dict(payload)
                retry_payload["model"] = "qwen2.5:1.5b"
                if "options" in retry_payload:
                    retry_payload["options"]["num_ctx"] = 2048
                try:
                    retry_bytes = json.dumps(retry_payload).encode("utf-8")
                    retry_req = urllib.request.Request(OLLAMA_API_URL, data=retry_bytes, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(retry_req, timeout=timeout) as resp:
                        self.model = "qwen2.5:1.5b"
                        return json.loads(resp.read().decode("utf-8"))
                except Exception:
                    pass
            raise RuntimeError(f"Ollama HTTP error {e.code}: {err_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama connection error at {OLLAMA_API_URL}: {str(e)}")

    def _ensure_critical_tool_execution(self, user_input: str, turn_tool_calls: List[Dict[str, Any]]) -> str:
        """
        Safety guard: If customer clearly requested ordering, reviewing, or confirming,
        guarantee the respective tool call is executed even if the LLM only responded in free text.
        """
        text_lower = user_input.lower()
        tools_executed = {tc["tool"] for tc in turn_tool_calls}
        extra_messages = []

        # Ambiguous Category & General Order Guard (Ask first before adding):
        ambig = check_ambiguous_category_request(user_input)
        if ambig:
            if ambig.get("type") == "general_order":
                return ambig["message"]
            cat_name = ambig.get("category")
            if cat_name and "get_menu_by_category" not in tools_executed:
                m_res = self.execute_tool("get_menu_by_category", {"category": cat_name})
                turn_tool_calls.append({"tool": "get_menu_by_category", "arguments": {"category": cat_name}, "result": m_res})
                tools_executed.add("get_menu_by_category")
                if self.state != STATE_CONFIRMED:
                    self.transition_state(STATE_MENU_BROWSING, reason=f"Customer inquiring about {cat_name} options")
                return ambig["prompt"]

        # Creator Identity Guard (Defense Standard):
        creator_keywords = ["who created you", "who made you", "who programmed you", "who developed you", "who are your creators", "sino gumawa sa'yo", "sino gumawa sayo", "sino gumawa"]
        if any(w in text_lower for w in creator_keywords):
            return "I was created and engineered by Christian Ezekiel L. Carvajal and John Miko P. Sarsalijo, 3rd-year BS Computer Science (Data Science) students at the University of Perpetual Help System DALTA (UPHSD), for our AI Finals Project (Perpetual Bistro)!"

        # Audit Inquiries: What item was removed?
        rem_keywords = ["what item was removed", "what was removed", "what did you remove", "what did i remove", "ano tinanggal", "anong tinanggal", "tinanggal ko", "tinanggal kanina", "removed item", "which item was removed"]
        if any(w in text_lower for w in rem_keywords) and "get_removed_items" not in tools_executed:
            rem_res = self.execute_tool("get_removed_items", {})
            turn_tool_calls.append({"tool": "get_removed_items", "arguments": {}, "result": rem_res})
            tools_executed.add("get_removed_items")
            rem_list = rem_res.get("removed_items", [])
            if rem_list:
                latest = rem_list[0]
                extra_messages.append(f"The last item removed from your cart was {latest['quantity']}x {latest['item_name']} (₱{latest['unit_price']:.2f}).")
            else:
                extra_messages.append("No items have been removed from your cart during this session.")

        # Audit Inquiries: Last order item
        last_keywords = ["last order", "last item", "last dish", "huling order", "huling in-order", "huling dinagdag", "what did i just add", "what was my last"]
        if any(w in text_lower for w in last_keywords) and "get_last_order_item" not in tools_executed:
            last_res = self.execute_tool("get_last_order_item", {})
            turn_tool_calls.append({"tool": "get_last_order_item", "arguments": {}, "result": last_res})
            tools_executed.add("get_last_order_item")
            last_i = last_res.get("last_item")
            if last_i:
                extra_messages.append(f"Your last added item was {last_i['quantity']}x {last_i['name']} (₱{last_i['unit_price']:.2f}).")
            else:
                extra_messages.append("You haven't added any items to your order tab yet.")

        # Clear Cart Safety Guard
        is_clear_intent = bool(re.search(
            r"\b(clear\s+(?:my\s+)?order|clear\s+(?:my\s+)?cart|clear\s+tab|cancel\s+(?:my\s+)?order|cancel\s+all|don't\s+want\s+to\s+eat|dont\s+want\s+to\s+eat|ayaw\s+ko\s+nang?\s+kumain|ayaw\s+na\s+kumain|reset\s+cart|reset\s+order|alisin\s+lahat|tanggalin\s+lahat)\b",
            text_lower
        ))
        if is_clear_intent and "clear_cart" not in tools_executed:
            c_res = self.execute_tool("clear_cart", {})
            turn_tool_calls.append({"tool": "clear_cart", "arguments": {}, "result": c_res})
            tools_executed.add("clear_cart")
            return "I have cleared all items from your order tab. Your cart is now completely empty."

        # Replace Item Safety Guard
        is_replace_intent = bool(re.search(r"\b(replace|swap|substitute|palitan|change)\b", text_lower)) and bool(re.search(r"\b(with|to|for|ng|into)\b", text_lower))
        if is_replace_intent and "replace_order_item" not in tools_executed:
            rep_match = re.search(r"(?:replace|swap|substitute|palitan|change)\s+(.*?)\s+(?:with|to|for|ng|into)\s+(.*?)(?:\s+(?:instead|cause|kasi|because|please|po)|\s*$)", text_lower)
            raw_old = rep_match.group(1).strip() if rep_match else "burger"
            raw_new = rep_match.group(2).strip() if rep_match else text_lower
            rep_res = self.execute_tool("replace_order_item", {"old_item": raw_old, "new_item": raw_new, "qty": 1})
            turn_tool_calls.append({"tool": "replace_order_item", "arguments": {"old_item": raw_old, "new_item": raw_new, "qty": 1}, "result": rep_res})
            tools_executed.add("replace_order_item")
            if rep_res.get("status") == "success":
                return f"I've replaced {rep_res.get('old_item')} with {rep_res.get('new_item')} in your order tab! Your new order total is ₱{rep_res.get('new_order_total', 0):.2f}."
            return rep_res.get("message", "Unable to replace item.")

        has_allergen_words = any(w in text_lower for w in ["allergen", "allergy", "allergies", "lactose", "dairy", "gluten", "wheat", "nut", "nuts", "egg", "pork", "shellfish"])
        has_question_words = bool(re.search(r"\b(does|is|are|can|what|anong|ano|meron|mayroon|how much|magkano|ingredients|contain)\b", text_lower))
        has_explicit_order_verb = bool(re.search(r"\b(order|add|pabili|buy|get|take|pakidagdag|isama|paki-order|i-order|dagdag|bilhin)\b", text_lower))
        has_remove_verb = bool(re.search(r"\b(remove|delete|cancel|alisin|tanggalin|bawas|ibawas|alising|tanggal)\b", text_lower))
        has_update_verb = bool(re.search(r"\b(change|update|gawing|palitan|make\s+it|set\s+to|increase|decrease)\b", text_lower))
        is_remove_intent = has_remove_verb or bool(re.search(r"\b(only\s+1\s+item|only\s+item|remove\s+that|remove\s+it|delete\s+that|cancel\s+that|alisin\s+yan|tanggalin\s+yan|tanggal\s+yan)\b", text_lower))

        # 1. Allergen Tool Safety Guard: If user asked about allergens and tool wasn't executed:
        if has_allergen_words and "check_item_allergen" not in tools_executed:
            found_dishes_a = find_dishes_in_text(user_input)
            target_dish = found_dishes_a[0][0]["name"] if found_dishes_a else "Perpetual Classic Burger"
            target_allergen = "Dairy"
            for a in ["Gluten", "Nuts", "Egg", "Pork", "Shellfish", "Dairy"]:
                if a.lower() in text_lower:
                    target_allergen = a
                    break
            a_res = self.execute_tool("check_item_allergen", {"item_name": target_dish, "allergen": target_allergen})
            turn_tool_calls.append({"tool": "check_item_allergen", "arguments": {"item_name": target_dish, "allergen": target_allergen}, "result": a_res})
            tools_executed.add("check_item_allergen")

        # 2. Remove Item Safety Guard (with contextual pronoun resolution from SQLite cart)
        if is_remove_intent and "remove_item_from_order" not in tools_executed:
            found_dishes_r = find_dishes_in_text(user_input)
            if found_dishes_r:
                for dish, qty, instructions in found_dishes_r:
                    r_res = self.execute_tool("remove_item_from_order", {"item_name": dish["name"]})
                    turn_tool_calls.append({"tool": "remove_item_from_order", "arguments": {"item_name": dish["name"]}, "result": r_res})
                    tools_executed.add("remove_item_from_order")
                    if r_res.get("status") == "success":
                        extra_messages.append(r_res.get("message", f"Removed {dish['name']} from tab."))
            else:
                cart_current = db_tools.view_current_order(self.session_id, db_path=self.db_path)
                cart_items = cart_current.get("line_items", [])
                if len(cart_items) == 1:
                    target_name = cart_items[0]["name"]
                    r_res = self.execute_tool("remove_item_from_order", {"item_name": target_name})
                    turn_tool_calls.append({"tool": "remove_item_from_order", "arguments": {"item_name": target_name}, "result": r_res})
                    tools_executed.add("remove_item_from_order")
                    if r_res.get("status") == "success":
                        extra_messages.append(f"Removed {target_name} from your order tab. Your cart is now empty.")
                elif len(cart_items) > 1 and any(w in text_lower for w in ["all", "everything", "lahat", "clear", "clear cart"]):
                    for it in cart_items:
                        r_res = self.execute_tool("remove_item_from_order", {"item_name": it["name"]})
                        turn_tool_calls.append({"tool": "remove_item_from_order", "arguments": {"item_name": it["name"]}, "result": r_res})
                    tools_executed.add("remove_item_from_order")
                    extra_messages.append("Cleared all items from your order tab.")

        # 3. Update Item Quantity Safety Guard:
        if has_update_verb and not is_remove_intent and "update_item_quantity" not in tools_executed:
            found_dishes_u = find_dishes_in_text(user_input)
            for dish, qty, instructions in found_dishes_u:
                u_res = self.execute_tool("update_item_quantity", {"item_name": dish["name"], "qty": qty})
                turn_tool_calls.append({"tool": "update_item_quantity", "arguments": {"item_name": dish["name"], "qty": qty}, "result": u_res})
                tools_executed.add("update_item_quantity")
                if u_res.get("status") == "success":
                    extra_messages.append(u_res.get("message", f"Updated {dish['name']} to {qty}."))

        # 4. Add Item(s) to Order ONLY IF NOT allergen query AND NOT removing AND NOT updating AND NOT already added:
        if "add_item_to_order" not in tools_executed and "update_item_quantity" not in tools_executed and not has_allergen_words and not is_remove_intent and not has_update_verb and (has_explicit_order_verb or not has_question_words):
            found_dishes = find_dishes_in_text(user_input)
            if found_dishes:
                added_dish_names = {(tc.get("arguments", {}).get("item_name") or tc.get("arguments", {}).get("dish_name")) for tc in turn_tool_calls if tc["tool"] == "add_item_to_order"}
                for dish, qty, instructions in found_dishes:
                    if dish["name"] not in added_dish_names:
                        res = self.execute_tool("add_item_to_order", {
                            "item_name": dish["name"],
                            "qty": qty,
                            "instructions": instructions
                        })
                        turn_tool_calls.append({
                            "tool": "add_item_to_order",
                            "arguments": {"item_name": dish["name"], "qty": qty, "instructions": instructions},
                            "result": res
                        })
                        tools_executed.add("add_item_to_order")
                        if res.get("status") == "success":
                            extra_messages.append(res.get("message", f"Added {qty}x {dish['name']}"))
                        elif res.get("status") == "out_of_stock":
                            extra_messages.append(f"{res.get('message')} {res.get('suggestion', '')}".strip())

        # 5. Order Confirmation / Checkout (if user asked to confirm or place order)
        confirm_keywords = [
            "confirm", "place my order", "place order", "place the order", "place it",
            "place this", "place na", "paki-place", "paki place", "finalize", "checkout",
            "bayaran", "i-confirm", "kumpirmahin", "pakiconfirm", "order na po", "complete order"
        ]
        if any(w in text_lower for w in confirm_keywords):
            if "confirm_order_and_deduct_stock" not in tools_executed:
                res = self.execute_tool("confirm_order_and_deduct_stock", {})
                turn_tool_calls.append({"tool": "confirm_order_and_deduct_stock", "arguments": {}, "result": res})
                tools_executed.add("confirm_order_and_deduct_stock")
                if res.get("status") == "success":
                    extra_messages.append(
                        f"Receipt #{res.get('receipt_id')} generated for Table #{res.get('table_no')}. "
                        f"Order routed to kitchen display! Grand Total: ₱{res.get('grand_total'):.2f}."
                    )
                elif res.get("message") == "Cannot confirm empty order.":
                    extra_messages.append("Your order tab is currently empty. Please select items from our menu first.")

        # 6. View Current Bill / Tab (only if explicitly inquiring about bill/cart and NOT ordering/removing/confirming)
        elif not is_remove_intent and not has_explicit_order_verb and not any(w in text_lower for w in ["add", "buy", "order", "pabili"]) and re.search(r"\b(bill|tab|magkano|bayarin|summary|review|view cart|show cart|check cart|tingnan ang cart)\b", text_lower):
            if "view_current_order" not in tools_executed and "confirm_order_and_deduct_stock" not in tools_executed:
                res = self.execute_tool("view_current_order", {})
                turn_tool_calls.append({"tool": "view_current_order", "arguments": {}, "result": res})

        # 4. Menu & Category Browsing Guard (if customer asks what else is sold / available)
        browse_menu_keywords = [
            "ano pang", "ano pa", "ibang tinda", "iba ninyong tinda", "iba niyong tinda",
            "anong tinda", "anong meron", "ano ang menu", "what else do you sell", "what other items",
            "what do you sell", "what's on the menu", "menu options", "food choices"
        ]
        if any(w in text_lower for w in browse_menu_keywords) and not tools_executed:
            m_res = self.execute_tool("get_menu_by_category", {"category": "all"})
            turn_tool_calls.append({"tool": "get_menu_by_category", "arguments": {"category": "all"}, "result": m_res})
            tools_executed.add("get_menu_by_category")
            if self.state != STATE_CONFIRMED:
                self.transition_state(STATE_MENU_BROWSING, reason="Customer menu inquiry")

        # 7. Recommendations Guard (if customer asks for recommendations/suggestions)
        rec_keywords = ["suggest", "recommend", "rekomenda", "masarap", "best seller", "bestseller", "cant pick", "can't pick", "di makapili", "hindi makapili"]
        if any(w in text_lower for w in rec_keywords) and "get_recommendations" not in tools_executed:
            rec_res = self.execute_tool("get_recommendations", {})
            turn_tool_calls.append({"tool": "get_recommendations", "arguments": {}, "result": rec_res})
            tools_executed.add("get_recommendations")
            recs = rec_res.get("bestsellers", []) or rec_res.get("recommended_items", [])
            rec_str = ", ".join([f"{r['name']} (₱{r['price']:.2f})" for r in recs[:3]])
            pairings = rec_res.get("pairing_suggestions", [])
            pair_str = f" Pairings: {'; '.join(pairings[:2])}." if pairings else ""
            extra_messages.append(f"Recommended crowd-favorites: {rec_str}.{pair_str} Which one would you like to try?")

        return " ".join(extra_messages)

    def _fallback_dialogue_turn(self, user_input: str, error_reason: str = "") -> Dict[str, Any]:
        """
        High-reliability automated bistro agent fallback.
        Ensures conversational responsiveness, state machine progression,
        and SQLite tool execution when Ollama is unavailable, out of memory, or slow.
        """
        text_lower = user_input.lower().strip()
        turn_tool_calls: List[Dict[str, Any]] = []
        is_filipino = self.language == "filipino"

        confirm_keywords = [
            "confirm", "place my order", "place order", "place the order", "place it",
            "place this", "place na", "paki-place", "paki place", "finalize", "checkout",
            "bayaran", "i-confirm", "kumpirmahin", "pakiconfirm", "order na po", "complete order"
        ]
        has_confirm_intent = any(w in text_lower for w in confirm_keywords)
        found_dishes = find_dishes_in_text(user_input)

        has_greeting = bool(re.search(r"\b(hi|hello|mabuhay|kamusta|good morning|good afternoon|good evening)\b", text_lower))
        has_browse_query = bool(re.search(r"\b(what|anong|ano|meron|show|list|menu|options|choices|categories|tignan|tingnan|available|mayroon)\b", text_lower))
        has_order_verb = bool(re.search(r"\b(order|add|buy|get|take|pabili|hingi|gusto|pakidagdag|isama)\b", text_lower))
        has_remove_verb = bool(re.search(r"\b(remove|delete|cancel|alisin|tanggalin|bawas|ibawas|alising|tanggal)\b", text_lower))
        has_update_verb = bool(re.search(r"\b(change|update|gawing|palitan|make\s+it|set\s+to|increase|decrease)\b", text_lower))
        is_remove_intent = has_remove_verb or bool(re.search(r"\b(only\s+1\s+item|only\s+item|remove\s+that|remove\s+it|delete\s+that|cancel\s+that|alisin\s+yan|tanggalin\s+yan|tanggal\s+yan)\b", text_lower))

        # Post-Confirmation State Lock Guard: Block any modification/cancellation once confirmed
        if self.state == STATE_CONFIRMED:
            if found_dishes or has_confirm_intent or any(w in text_lower for w in ["cancel", "kansela", "bawiin", "dagdag", "palitan", "alisin", "tanggalin", "remove", "change", "modify", "edit"]):
                if is_filipino:
                    resp_text = (
                        f"Paumanhin po! Ang inyong order para sa Table #{self.table_no} ay kumpirmado na at kasalukuyan nang inihahanda ng ating chef sa kusina. "
                        "Hindi na po ito maaaring baguhin o kanselahin. Maaari po kayong magbukas ng bagong order tab kung nais ninyong magdagdag ng ibang putahe!"
                    )
                else:
                    resp_text = (
                        f"I apologize, but your order for Table #{self.table_no} has already been confirmed and sent to the kitchen for preparation. "
                        "Orders cannot be modified or canceled once confirmed. Feel free to open a new order session if you would like to order additional items!"
                    )
                cart_state = db_tools.view_current_order(self.session_id, db_path=self.db_path)
                turn_data = {
                    "turn_index": len(self.messages),
                    "user_input": user_input,
                    "assistant_response": resp_text,
                    "current_state": STATE_CONFIRMED,
                    "fsm_state": STATE_CONFIRMED,
                    "tool_calls": [],
                    "engine": "fallback_deterministic",
                    "reason": "post_confirmation_state_lock",
                    "cart": cart_state,
                }
                self.audit_log.append({"type": "dialogue_turn", "data": turn_data})
                return turn_data

        is_clear_intent = bool(re.search(
            r"\b(clear\s+(?:my\s+)?order|clear\s+(?:my\s+)?cart|clear\s+tab|cancel\s+(?:my\s+)?order|cancel\s+all|don't\s+want\s+to\s+eat|dont\s+want\s+to\s+eat|ayaw\s+ko\s+nang?\s+kumain|ayaw\s+na\s+kumain|reset\s+cart|reset\s+order|alisin\s+lahat|tanggalin\s+lahat)\b",
            text_lower
        ))
        is_replace_intent = bool(re.search(r"\b(replace|swap|substitute|palitan|change)\b", text_lower)) and bool(re.search(r"\b(with|to|for|ng|into)\b", text_lower))

        # Intent: Clear Order Tab Completely
        if is_clear_intent:
            c_res = self.execute_tool("clear_cart", {})
            turn_tool_calls.append({"tool": "clear_cart", "arguments": {}, "result": c_res})
            if is_filipino:
                resp_text = "Naalis ko na po ang lahat ng putahe sa inyong order tab. Wala na pong laman ang inyong cart. Sabihan niyo lang po ako kung nais ninyong mag-order ulit!"
            else:
                resp_text = "I have cleared all items from your order tab. Your cart is now completely empty. Feel free to let me know whenever you'd like to order again!"
            cart_state = db_tools.view_current_order(self.session_id, db_path=self.db_path)
            turn_data = {
                "turn_index": len(self.messages),
                "user_input": user_input,
                "assistant_response": resp_text,
                "current_state": self.state,
                "fsm_state": self.state,
                "tool_calls": turn_tool_calls,
                "engine": "fallback_deterministic",
                "reason": "clear_cart_intent",
                "cart": cart_state,
            }
            self.audit_log.append({"type": "dialogue_turn", "data": turn_data})
            return turn_data

        # Intent: Replace / Substitute Item in Cart
        elif is_replace_intent:
            rep_match = re.search(r"(?:replace|swap|substitute|palitan|change)\s+(.*?)\s+(?:with|to|for|ng|into)\s+(.*?)(?:\s+(?:instead|cause|kasi|because|please|po)|\s*$)", text_lower)
            raw_old = rep_match.group(1).strip() if rep_match else "burger"
            raw_new = rep_match.group(2).strip() if rep_match else text_lower
            rep_res = self.execute_tool("replace_order_item", {"old_item": raw_old, "new_item": raw_new, "qty": 1})
            turn_tool_calls.append({"tool": "replace_order_item", "arguments": {"old_item": raw_old, "new_item": raw_new, "qty": 1}, "result": rep_res})
            if rep_res.get("status") == "success":
                if is_filipino:
                    resp_text = f"Napalitan ko na po ang {rep_res.get('old_item')} ng {rep_res.get('new_item')} sa inyong order tab! Ang inyong bagong total ay ₱{rep_res.get('new_order_total', 0):.2f}."
                else:
                    resp_text = f"I've replaced {rep_res.get('old_item')} with {rep_res.get('new_item')} in your order tab! Your new order total is ₱{rep_res.get('new_order_total', 0):.2f}. Would you like to add anything else?"
            else:
                resp_text = rep_res.get("message", "Unable to replace item.")
            cart_state = db_tools.view_current_order(self.session_id, db_path=self.db_path)
            turn_data = {
                "turn_index": len(self.messages),
                "user_input": user_input,
                "assistant_response": resp_text,
                "current_state": self.state,
                "fsm_state": self.state,
                "tool_calls": turn_tool_calls,
                "engine": "fallback_deterministic",
                "reason": "replace_item_intent",
                "cart": cart_state,
            }
            self.audit_log.append({"type": "dialogue_turn", "data": turn_data})
            return turn_data

        # 0. Intent: Greeting (if customer greets without dishes or explicit ordering verbs)
        elif has_greeting and not has_order_verb and not (found_dishes and not has_browse_query):
            if is_filipino:
                resp_text = (
                    f"Mabuhay! Ako si BistroBot, ang inyong AI dining assistant para sa Table #{self.table_no}. "
                    "Ano po ang gusto ninyong i-order ngayon? Pwede kayong magtanong tungkol sa aming mga burgers, pastas, inumin, o mag-place ng order!"
                )
            else:
                resp_text = (
                    f"Mabuhay and welcome to Perpetual Bistro! I am BistroBot, your virtual dining assistant at Table #{self.table_no}. "
                    "How may I serve you today? Feel free to ask about our burgers, pastas, chef specials, or place your order!"
                )
            if self.state == STATE_CONFIRMED:
                pass
            else:
                self.transition_state(STATE_GREETING, reason="Customer greeting")
            turn_data = {
                "turn_index": len(self.messages),
                "user_input": user_input,
                "assistant_response": resp_text,
                "current_state": self.state,
                "fsm_state": self.state,
                "tool_calls": [],
                "engine": "fallback_deterministic",
                "reason": "greeting_intent",
            }
            self.audit_log.append({"type": "dialogue_turn", "data": turn_data})
            return turn_data

        # 1. Intent: Allergen & Dietary Query (Priority 1: Never add to cart!)
        # Clear Cart Safety Guard
        is_clear_intent = bool(re.search(
            r"\b(clear\s+(?:my\s+)?order|clear\s+(?:my\s+)?cart|clear\s+tab|cancel\s+(?:my\s+)?order|cancel\s+all|don't\s+want\s+to\s+eat|dont\s+want\s+to\s+eat|ayaw\s+ko\s+nang?\s+kumain|ayaw\s+na\s+kumain|reset\s+cart|reset\s+order|alisin\s+lahat|tanggalin\s+lahat)\b",
            text_lower
        ))
        if is_clear_intent and "clear_cart" not in tools_executed:
            c_res = self.execute_tool("clear_cart", {})
            turn_tool_calls.append({"tool": "clear_cart", "arguments": {}, "result": c_res})
            tools_executed.add("clear_cart")
            return "I have cleared all items from your order tab. Your cart is now completely empty."

        # Replace Item Safety Guard
        is_replace_intent = bool(re.search(r"\b(replace|swap|substitute|palitan|change)\b", text_lower)) and bool(re.search(r"\b(with|to|for|ng|into)\b", text_lower))
        if is_replace_intent and "replace_order_item" not in tools_executed:
            rep_match = re.search(r"(?:replace|swap|substitute|palitan|change)\s+(.*?)\s+(?:with|to|for|ng|into)\s+(.*?)(?:\s+(?:instead|cause|kasi|because|please|po)|\s*$)", text_lower)
            raw_old = rep_match.group(1).strip() if rep_match else "burger"
            raw_new = rep_match.group(2).strip() if rep_match else text_lower
            rep_res = self.execute_tool("replace_order_item", {"old_item": raw_old, "new_item": raw_new, "qty": 1})
            turn_tool_calls.append({"tool": "replace_order_item", "arguments": {"old_item": raw_old, "new_item": raw_new, "qty": 1}, "result": rep_res})
            tools_executed.add("replace_order_item")
            if rep_res.get("status") == "success":
                return f"I've replaced {rep_res.get('old_item')} with {rep_res.get('new_item')} in your order tab! Your new order total is ₱{rep_res.get('new_order_total', 0):.2f}."
            return rep_res.get("message", "Unable to replace item.")

        has_allergen_words = any(w in text_lower for w in ["allergen", "allergy", "allergies", "lactose", "dairy", "gluten", "wheat", "nut", "nuts", "egg", "pork", "shellfish"])
        if has_allergen_words:
            target_dish = found_dishes[0][0]["name"] if found_dishes else "Perpetual Classic Burger"
            target_allergen = "Dairy"
            for a in ["Gluten", "Nuts", "Egg", "Pork", "Shellfish", "Dairy"]:
                if a.lower() in text_lower:
                    target_allergen = a
                    break

            res = self.execute_tool("check_item_allergen", {"item_name": target_dish, "allergen": target_allergen})
            turn_tool_calls.append({"tool": "check_item_allergen", "arguments": {"item_name": target_dish, "allergen": target_allergen}, "result": res})

            has_allergen = res.get("contains_allergen", False)
            all_decl = res.get("declared_allergens")
            all_list = res.get("all_item_allergens", [])
            allergen_str = all_decl if all_decl else (", ".join(all_list) if all_list else "None")

            if is_filipino:
                if has_allergen:
                    resp_text = (
                        f"Opo, ang {target_dish} ay may {target_allergen} (Kabuuang allergens: {allergen_str}). "
                        f"Kung may lactose intolerance o dietary restriction po kayo, maaari po nating ipatanggal ang cheese o i-customize ang inyong order!"
                    )
                else:
                    resp_text = f"Wala pong {target_allergen} ang {target_dish}. Ligtas po ito para sa inyo! (Nakatala allergens: {allergen_str})."
            else:
                if has_allergen:
                    resp_text = (
                        f"Yes, the {target_dish} contains {target_allergen} (Allergens: {allergen_str}). "
                        f"If you have mild lactose intolerance, we can gladly prepare it without cheese for you!"
                    )
                else:
                    resp_text = f"Good news! The {target_dish} is free of {target_allergen} (Allergens: {allergen_str})."

            if self.state != STATE_CONFIRMED:
                self.transition_state(STATE_MENU_BROWSING, reason="Customer allergen query")

            cart_state = db_tools.view_current_order(self.session_id, db_path=self.db_path)
            turn_data = {
                "turn_index": len(self.messages),
                "user_input": user_input,
                "assistant_response": resp_text,
                "current_state": self.state,
                "fsm_state": self.state,
                "tool_calls": turn_tool_calls,
                "engine": "fallback_deterministic",
                "reason": "allergen_inquiry",
                "cart": cart_state,
            }
            self.audit_log.append({"type": "dialogue_turn", "data": turn_data})
            return turn_data

        # Intent: Remove Item from Order (with contextual pronoun resolution)
        elif is_remove_intent:
            cart_current = db_tools.view_current_order(self.session_id, db_path=self.db_path)
            cart_items = cart_current.get("line_items", [])
            msgs = []
            if found_dishes:
                for dish, qty, instructions in found_dishes:
                    res = self.execute_tool("remove_item_from_order", {"item_name": dish["name"]})
                    turn_tool_calls.append({"tool": "remove_item_from_order", "arguments": {"item_name": dish["name"]}, "result": res})
                    if res.get("status") == "success":
                        if is_filipino:
                            msgs.append(f"Naalis ko na po ang {dish['name']} sa inyong order tab.")
                        else:
                            msgs.append(f"Removed {dish['name']} from your order tab.")
                    else:
                        msgs.append(res.get("message", f"Could not remove {dish['name']}."))
            elif len(cart_items) == 1:
                target_name = cart_items[0]["name"]
                res = self.execute_tool("remove_item_from_order", {"item_name": target_name})
                turn_tool_calls.append({"tool": "remove_item_from_order", "arguments": {"item_name": target_name}, "result": res})
                if res.get("status") == "success":
                    if is_filipino:
                        msgs.append(f"Naalis ko na po ang {target_name} sa inyong order tab. Wala na pong laman ang inyong cart.")
                    else:
                        msgs.append(f"Removed {target_name} from your order tab. Your cart is now empty.")
                else:
                    msgs.append(res.get("message", f"Could not remove {target_name}."))
            elif len(cart_items) > 1 and any(w in text_lower for w in ["all", "everything", "lahat", "clear", "clear cart"]):
                for it in cart_items:
                    res = self.execute_tool("remove_item_from_order", {"item_name": it["name"]})
                    turn_tool_calls.append({"tool": "remove_item_from_order", "arguments": {"item_name": it["name"]}, "result": res})
                if is_filipino:
                    msgs.append("Naalis ko na po ang lahat ng putahe sa inyong cart.")
                else:
                    msgs.append("Cleared all items from your order tab.")
            elif not cart_items:
                if is_filipino:
                    msgs.append("Wala pa pong laman ang inyong cart. Walang kailangang alisin.")
                else:
                    msgs.append("Your cart is currently empty. There is nothing to remove.")
            else:
                items_str = ", ".join([f"{it['quantity']}x {it['name']}" for it in cart_items])
                if is_filipino:
                    msgs.append(f"Mayroon po kayong {items_str} sa inyong cart. Alin po sa mga ito ang nais ninyong ipatanggal?")
                else:
                    msgs.append(f"You have {items_str} in your cart. Which one would you like me to remove?")
            resp_text = "\n".join(msgs)

        # Intent: Update Item Quantity in Order
        elif found_dishes and has_update_verb:
            msgs = []
            for dish, qty, instructions in found_dishes:
                res = self.execute_tool("update_item_quantity", {"item_name": dish["name"], "qty": qty})
                turn_tool_calls.append({"tool": "update_item_quantity", "arguments": {"item_name": dish["name"], "qty": qty}, "result": res})
                if res.get("status") == "success":
                    if is_filipino:
                        msgs.append(f"Na-update ko na po ang dami ng {dish['name']} sa {qty}.")
                    else:
                        msgs.append(f"Updated quantity of {dish['name']} to {qty}.")
                else:
                    msgs.append(res.get("message", f"Could not update {dish['name']}."))
            resp_text = "\n".join(msgs)

        # 2. Intent: Ordering Dishes (with optional immediate checkout in the same utterance)
        elif found_dishes and not has_remove_verb and not has_update_verb and (has_order_verb or not has_browse_query):
            added_msgs = []
            for dish, qty, instructions in found_dishes:
                res = self.execute_tool("add_item_to_order", {
                    "item_name": dish["name"],
                    "qty": qty,
                    "instructions": instructions
                })
                turn_tool_calls.append({
                    "tool": "add_item_to_order",
                    "arguments": {"item_name": dish["name"], "qty": qty, "instructions": instructions},
                    "result": res
                })
                if res.get("status") == "out_of_stock":
                    alts = res.get("recommended_alternatives", [])
                    alt_str = ", ".join([f"{a['name']} (₱{a['price']:.2f})" for a in alts]) if alts else "other items from our menu"
                    if is_filipino:
                        added_msgs.append(
                            f"Paumanhin po! Ang {dish['name']} ay kasalukuyang out-of-stock sa aming kusina ({res.get('stock_available', 0)} available). "
                            f"Nais niyo po bang subukan ang aming {alt_str} bilang alternatibo?"
                        )
                    else:
                        added_msgs.append(
                            f"I apologize, but {dish['name']} is currently out of stock ({res.get('stock_available', 0)} available). "
                            f"Would you like to try our {alt_str} instead?"
                        )
                elif res.get("status") == "stock_limit_reached":
                    if is_filipino:
                        added_msgs.append(f"Paumanhin po! {res.get('message')}")
                    else:
                        added_msgs.append(f"I apologize! {res.get('message')}")
                elif res.get("status") == "error":
                    added_msgs.append(res.get("message", "Cannot modify order."))
                else:
                    inst_note = f" ({instructions})" if instructions else ""
                    if is_filipino:
                        added_msgs.append(f"Nai-dagdag ko na po ang {qty}x {dish['name']}{inst_note} sa inyong order tab.")
                    else:
                        added_msgs.append(f"Added {qty}x {dish['name']}{inst_note} to your tab.")

            # If the customer also said "place order" / "confirm" in the same voice command:
            if has_confirm_intent:
                c_res = self.execute_tool("confirm_order_and_deduct_stock", {})
                turn_tool_calls.append({"tool": "confirm_order_and_deduct_stock", "arguments": {}, "result": c_res})
                if c_res.get("status") == "success":
                    if is_filipino:
                        added_msgs.append(
                            f"\nAt na-confirm na po ang inyong order! (Receipt #{c_res.get('receipt_id')}) para sa Table #{c_res.get('table_no')}.\n"
                            f"Naipadala na po ang order ticket sa kitchen display at inihahanda na ng aming chef.\n"
                            f"Subtotal: ₱{c_res.get('subtotal', 0.0):.2f} | 12% VAT: ₱{c_res.get('vat_included', 0.0):.2f}\n"
                            f"Kabuuang Bayarin: ₱{c_res.get('grand_total', 0.0):.2f}.\n"
                            f"Ihahatid po ito sa inyong lamesa sa lalong madaling panahon!"
                        )
                    else:
                        added_msgs.append(
                            f"\nAnd your order has been confirmed! (Receipt #{c_res.get('receipt_id')}) for Table #{c_res.get('table_no')}.\n"
                            f"Your order ticket is routed to the kitchen display and is being prepared.\n"
                            f"Subtotal: ₱{c_res.get('subtotal', 0.0):.2f} | 12% VAT: ₱{c_res.get('vat_included', 0.0):.2f}\n"
                            f"Grand Total: ₱{c_res.get('grand_total', 0.0):.2f}.\n"
                            f"Your meal will be served to your table shortly!"
                        )
            else:
                if is_filipino:
                    added_msgs.append("May nais pa po ba kayong idagdag, o nais niyo na pong i-confirm and place order?")
                else:
                    added_msgs.append("Would you like to add anything else, or confirm and place your order?")

            resp_text = "\n".join(added_msgs)

        # 2. Intent: Standalone Order Confirmation / Checkout (dishes already in cart)
        elif has_confirm_intent:
            res = self.execute_tool("confirm_order_and_deduct_stock", {})
            turn_tool_calls.append({"tool": "confirm_order_and_deduct_stock", "arguments": {}, "result": res})
            if res.get("status") == "success":
                if is_filipino:
                    resp_text = (
                        f"Maraming salamat po! Kumpirmado na ang inyong order (Receipt #{res.get('receipt_id')}) para sa Table #{res.get('table_no')}.\n"
                        f"Naipadala na po ang order ticket sa kitchen display at inihahanda na ng aming chef.\n"
                        f"Subtotal: ₱{res.get('subtotal', 0.0):.2f} | 12% VAT: ₱{res.get('vat_included', 0.0):.2f}\n"
                        f"Kabuuang Bayarin: ₱{res.get('grand_total', 0.0):.2f}.\n"
                        f"Ihahatid po ito sa inyong lamesa sa lalong madaling panahon!"
                    )
                else:
                    resp_text = (
                        f"Thank you! Your order has been confirmed (Receipt #{res.get('receipt_id')}) for Table #{res.get('table_no')}.\n"
                        f"Your order ticket is routed to the kitchen display and is being prepared.\n"
                        f"Subtotal: ₱{res.get('subtotal', 0.0):.2f} | 12% VAT: ₱{res.get('vat_included', 0.0):.2f}\n"
                        f"Grand Total: ₱{res.get('grand_total', 0.0):.2f}.\n"
                        f"Your meal will be served to your table shortly!"
                    )
            else:
                if is_filipino:
                    resp_text = "Wala pa pong laman ang inyong order tab. Ano po ang nais ninyong i-order mula sa ating menu?"
                else:
                    resp_text = "Your order tab is currently empty. What dishes would you like to order today?"

        # Intent: Inquire about Removed Items (Audit memory from SQLite)
        elif any(w in text_lower for w in ["what item was removed", "what was removed", "what did you remove", "what did i remove", "ano tinanggal", "anong tinanggal", "tinanggal ko", "tinanggal kanina", "removed item", "which item was removed"]):
            res = self.execute_tool("get_removed_items", {})
            turn_tool_calls.append({"tool": "get_removed_items", "arguments": {}, "result": res})
            rem_list = res.get("removed_items", [])
            if rem_list:
                latest = rem_list[0]
                if is_filipino:
                    resp_text = f"Ang pinakahuling putahe na naalis sa inyong cart ay ang {latest['quantity']}x {latest['item_name']} (₱{latest['unit_price']:.2f})."
                else:
                    resp_text = f"The item that was removed from your cart was {latest['quantity']}x {latest['item_name']} (₱{latest['unit_price']:.2f})."
            else:
                resp_text = "Wala pa pong naitalang tinanggal na putahe sa inyong cart para sa session na ito." if is_filipino else "No items have been removed from your cart during this dining session."

        # Intent: Inquire about Last Order / Last Added Item
        elif any(w in text_lower for w in ["last order", "last item", "last dish", "huling order", "huling in-order", "huling dinagdag", "what did i just add", "what was my last", "huling order ko"]):
            res = self.execute_tool("get_last_order_item", {})
            turn_tool_calls.append({"tool": "get_last_order_item", "arguments": {}, "result": res})
            last_item = res.get("last_item")
            if last_item:
                note = f" (Note: {last_item['special_instructions']})" if last_item.get("special_instructions") else ""
                if is_filipino:
                    resp_text = f"Ang inyong pinakahuling dinagdag sa cart ay {last_item['quantity']}x {last_item['name']} (₱{last_item['unit_price']:.2f} bawat isa){note}."
                else:
                    resp_text = f"Your last added item was {last_item['quantity']}x {last_item['name']} (₱{last_item['unit_price']:.2f} each){note}."
            else:
                resp_text = "Wala pa pong laman ang inyong order tab." if is_filipino else "You haven't added any items to your order tab yet."

        # Intent: Inquire about Unavailable / Out of Stock Items (Real-time SQL check)
        elif any(w in text_lower for w in ["unavailable", "out of stock", "ubos", "hindi available", "not available", "sold out", "wala na", "ano yung ubos"]):
            res = self.execute_tool("get_unavailable_items", {})
            turn_tool_calls.append({"tool": "get_unavailable_items", "arguments": {}, "result": res})
            un_items = res.get("unavailable_items", [])
            if un_items:
                un_desc = ", ".join([f"{it['name']} ({it['category']})" for it in un_items])
                if is_filipino:
                    resp_text = f"Kasalukuyang out-of-stock sa ating kusina ang mga sumusunod: {un_desc}. Lahat ng iba pang putahe ay handang ihanda para sa inyo!"
                else:
                    resp_text = f"Currently out of stock in our kitchen: {un_desc}. All other menu dishes are freshly in stock and ready to order!"
            else:
                resp_text = "Lahat po ng putahe sa ating menu ay available sa kusina ngayon!" if is_filipino else "All items on our bistro menu are currently in stock and available!"

        # Intent: Inquire about Available Items (Real-time SQL check)
        elif any(w in text_lower for w in ["available", "in stock", "anong meron", "mayroon kayo"]) and not any(w in text_lower for w in ["not", "un", "hindi", "out of", "ubos"]):
            res = self.execute_tool("get_available_items", {})
            turn_tool_calls.append({"tool": "get_available_items", "arguments": {}, "result": res})
            avail_items = res.get("available_items", [])
            sample = ", ".join([f"{it['name']} (₱{it['price']:.2f})" for it in avail_items[:6]])
            if is_filipino:
                resp_text = f"Narito po ang ilan sa ating mga available na putahe sa kusina: {sample}. Nais niyo po ba ng burger, pasta, o combo meal?"
            else:
                resp_text = f"Here are some of our popular available dishes in the kitchen: {sample}. Would you like burgers, pastas, sides, or beverages?"

        # 3. Intent: View Bill / Cart Review
        elif not is_remove_intent and not has_order_verb and not any(w in text_lower for w in ["add", "buy", "order", "pabili", "dagdag", "isama"]) and re.search(r"\b(bill|tab|total|magkano|bayarin|summary|review|view cart|show cart|tingnan ang cart)\b", text_lower):
            res = self.execute_tool("view_current_order", {})
            turn_tool_calls.append({"tool": "view_current_order", "arguments": {}, "result": res})
            items = res.get("line_items", [])
            if items:
                items_str = "\n".join([f"- {it['quantity']}x {it['name']} (₱{it['line_total']:.2f})" for it in items])
                if is_filipino:
                    resp_text = (
                        f"Narito po ang inyong kasalukuyang tab para sa Table #{self.table_no}:\n{items_str}\n\n"
                        f"Subtotal: ₱{res.get('subtotal', 0.0):.2f} (kasama ang 12% VAT: ₱{res.get('vat_12_pct_breakdown', 0.0):.2f})\n"
                        f"Kabuuang Halaga: ₱{res.get('grand_total', 0.0):.2f}.\n"
                        f"Nais niyo na po bang i-confirm ang inyong order?"
                    )
                else:
                    resp_text = (
                        f"Here is your current order tab for Table #{self.table_no}:\n{items_str}\n\n"
                        f"Subtotal: ₱{res.get('subtotal', 0.0):.2f} (12% VAT included: ₱{res.get('vat_12_pct_breakdown', 0.0):.2f})\n"
                        f"Grand Total: ₱{res.get('grand_total', 0.0):.2f}.\n"
                        f"Would you like to confirm and place this order?"
                    )
            else:
                resp_text = "Your cart is currently empty." if not is_filipino else "Wala pa pong laman ang inyong cart."

        # 4. Intent: Category / Menu Inquiry
        elif any(w in text_lower for w in ["burger", "burgers", "pasta", "pastas", "side", "sides", "drink", "drinks", "beverage", "dessert", "desserts", "combo", "combos", "value meal", "menu", "tinda", "sell", "offer", "other items", "pagkain", "ibang tinda", "ano pa", "ano pang"]):
            cat = "all"
            for c in ["Combos", "Burgers", "Pastas", "Sides", "Beverages", "Desserts"]:
                if c.lower() in text_lower or (c == "Combos" and "value meal" in text_lower):
                    cat = c
                    break
            res = self.execute_tool("get_menu_by_category", {"category": cat})
            turn_tool_calls.append({"tool": "get_menu_by_category", "arguments": {"category": cat}, "result": res})
            items = res.get("items", [])
            items_desc = ", ".join([f"{it['name']} (₱{it['price']:.2f})" for it in items[:5]])
            if is_filipino:
                resp_text = f"Narito po ang mga putahe sa {cat}: {items_desc}. Ano po ang nais ninyong subukan?"
            else:
                resp_text = f"Here are the items available under {cat}: {items_desc}. Which one would you like to try?"

        # 6. Intent: Recommendations & Bestsellers
        elif any(w in text_lower for w in ["recommend", "rekomenda", "best seller", "bestseller", "specialty", "masarap", "lunch", "popular", "can't pick", "cant pick", "di makapili", "hindi makapili", "suggest"]):
            res = self.execute_tool("get_recommendations", {})
            turn_tool_calls.append({"tool": "get_recommendations", "arguments": {}, "result": res})
            pairings = res.get("pairing_suggestions", [])
            pair_tag = (" Maaari niyo ring subukan ang aming pairing: " + " o ".join(pairings[:2]) + ".") if pairings else ""
            pair_eng = (" We also suggest pairing: " + " or ".join(pairings[:2]) + ".") if pairings else ""
            if is_filipino:
                resp_text = (
                    "Mabuhay po! Kung hindi pa kayo makapili, narito ang aming crowd-favorite bestsellers: "
                    "Perpetual Classic Burger (₱150.00), Truffle Mushroom Pasta (₱210.00), at Loaded Cheese Fries (₱110.00)! "
                    "Para naman sa pinaka-sulit na bundle, subukan ang Perpetual Champion Combo (₱285.00, tipid ng ₱30)!"
                    f"{pair_tag} Alin po ang nais ninyong i-try?"
                )
            else:
                resp_text = (
                    "Welcome! If you're having trouble deciding, our top bestsellers are the "
                    "Perpetual Classic Burger (₱150.00), Truffle Mushroom Pasta (₱210.00), and Loaded Cheese Fries (₱110.00)! "
                    "For the best value, check out our Perpetual Champion Combo (₱285.00, save ₱30)!"
                    f"{pair_eng} Which one would you like to try?"
                )

        # Intent: Analytics & Sales Query (Full Database Analysis Authority)
        elif any(w in text_lower for w in ["sales", "revenue", "analytics", "kita", "benta", "how much money", "order count", "orders today", "business"]):
            res = self.execute_tool("get_analytics_summary", {})
            turn_tool_calls.append({"tool": "get_analytics_summary", "arguments": {}, "result": res})
            rev = res.get("total_revenue", 0.0)
            cnt = res.get("total_orders", 0)
            low = res.get("low_stock_count", 0)
            if is_filipino:
                resp_text = (
                    f"Narito po ang live business analytics mula sa ating SQLite database:\n"
                    f"• Kabuuang Benta (Revenue): ₱{rev:.2f}\n"
                    f"• Kabuuang Orders: {cnt} ticket(s)\n"
                    f"• Low Stock Items: {low} item(s) ang mababa na ang imbentaryo."
                )
            else:
                resp_text = (
                    f"Here is the live bistro business analytics directly from SQLite:\n"
                    f"• Total Revenue: ₱{rev:.2f}\n"
                    f"• Total Orders: {cnt} ticket(s)\n"
                    f"• Low Stock Inventory: {low} item(s) below threshold."
                )

        # Intent: Kitchen Display / Tickets Query (Full Database Read Authority)
        elif any(w in text_lower for w in ["kitchen", "ticket", "tickets", "kds", "kusina", "preparing", "kitchen display"]):
            res = self.execute_tool("get_kitchen_tickets", {})
            turn_tool_calls.append({"tool": "get_kitchen_tickets", "arguments": {}, "result": res})
            tickets = res.get("tickets", [])
            if is_filipino:
                resp_text = f"Kasalukuyang may {len(tickets)} kitchen ticket(s) na nakatala sa ating database."
            else:
                resp_text = f"There are currently {len(tickets)} kitchen ticket(s) recorded in the live database."

        # 7. Intent: Creator & Developer Inquiry
        elif any(w in text_lower for w in ["who created", "who made", "who built", "who programmed", "who developed", "creators", "developer", "developers", "gumawa sa'yo", "gumawa sayo", "gumawa sa iyo", "sino gumawa"]):
            if is_filipino:
                resp_text = (
                    "Ako po ay buong-pusong binuo at ginawa nina Christian Ezekiel L. Carvajal at John Miko P. Sarsalijo, "
                    "mga mag-aaral sa 3rd year ng BS Computer Science - Data Science sa University of Perpetual Help System DALTA, "
                    "para sa kanilang AI Finals Project (Perpetual Bistro)! Paano ko po kayo matutulungan sa inyong dining experience ngayon?"
                )
            else:
                resp_text = (
                    "I was created and engineered by Christian Ezekiel L. Carvajal and John Miko P. Sarsalijo, "
                    "3rd-year BS Computer Science - Data Science students at the University of Perpetual Help System DALTA, "
                    "for their AI Finals Project! How can I assist you with your dining experience today?"
                )

        # 8. Intent: Greeting
        elif any(w in text_lower for w in ["hi", "hello", "mabuhay", "kamusta", "good morning", "good afternoon", "good evening"]):
            if is_filipino:
                resp_text = (
                    f"Mabuhay! Ako si BistroBot, ang inyong AI dining assistant para sa Table #{self.table_no}. "
                    "Ano po ang gusto ninyong i-order ngayon? Pwede kayong magtanong tungkol sa aming mga burgers, pastas, inumin, o mag-place ng order!"
                )
            else:
                resp_text = (
                    f"Mabuhay and welcome to Perpetual Bistro! I am BistroBot, your AI waiter for Table #{self.table_no}. "
                    "How may I serve you today? Feel free to ask about our menu categories, allergen details, or place an order!"
                )

        # Default Friendly Response
        else:
            if is_filipino:
                resp_text = (
                    f"Nandito po ako para tulungan kayo sa Table #{self.table_no}! "
                    "Maaari kayong mag-order ng burger, pasta, inumin, magtanong ng allergens, o tingnan ang inyong bill."
                )
            else:
                resp_text = (
                    f"I am here to assist you at Table #{self.table_no}! "
                    "Feel free to browse our menu, ask about dietary allergens, add dishes to your tab, or review your bill."
                )

        cart_state = db_tools.view_current_order(self.session_id, db_path=self.db_path)
        turn_data = {
            "session_id": self.session_id,
            "table_no": self.table_no,
            "fsm_state": self.state,
            "user_input": user_input,
            "assistant_response": resp_text,
            "thinking": f"[Automated Dialogue Engine Active | FSM State: {self.state}]",
            "tool_calls": turn_tool_calls,
            "cart": cart_state,
        }
        self.audit_log.append({"type": "dialogue_turn", "data": turn_data})
        return turn_data

    def process_turn(self, user_input: str) -> Dict[str, Any]:
        """
        Executes a conversational dialogue step with Ollama LLM and seamless automated fallback.
        """
        self._refresh_dynamic_context()
        self.messages.append({"role": "user", "content": user_input})
        if user_input.strip() == "Hello BistroBot":
            return self._fallback_dialogue_turn(user_input)

        turn_tool_calls: List[Dict[str, Any]] = []

        max_tool_iterations = 3
        iteration = 0
        final_content = ""
        model_thinking = ""
        executed_signatures: Set[str] = set()
        action_executed = False

        try:
            while iteration < max_tool_iterations:
                iteration += 1

                payload = {
                    "model": self.model,
                    "messages": self.messages,
                    "tools": BISTRO_TOOLS,
                    "stream": False,
                    "keep_alive": "24h",
                    "options": {
                        "temperature": 0.2,
                        "num_ctx": 2048,
                    }
                }

                resp = self._call_ollama_chat(payload, timeout=35)
                assistant_msg = resp.get("message", {})
                raw_content = assistant_msg.get("content", "")
                thinking = assistant_msg.get("thinking", "")
                if thinking:
                    model_thinking += ("\n" + thinking if model_thinking else thinking)

                tool_calls = assistant_msg.get("tool_calls", [])

                if not tool_calls and "```json" in raw_content and '"name":' in raw_content:
                    try:
                        match = re.search(r"```json\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
                        if match:
                            parsed = json.loads(match.group(1))
                            if "name" in parsed and "arguments" in parsed:
                                tool_calls = [{"function": parsed}]
                    except Exception:
                        pass

                text_lower = user_input.lower()
                is_clear_intent = bool(re.search(
                    r"\b(clear\s+(?:my\s+)?order|clear\s+(?:my\s+)?cart|clear\s+tab|cancel\s+(?:my\s+)?order|cancel\s+all|don't\s+want\s+to\s+eat|dont\s+want\s+to\s+eat|ayaw\s+ko\s+nang?\s+kumain|ayaw\s+na\s+kumain|reset\s+cart|reset\s+order|alisin\s+lahat|tanggalin\s+lahat)\b",
                    text_lower
                ))
                is_replace_intent = bool(re.search(r"\b(replace|swap|substitute|palitan|change)\b", text_lower)) and bool(re.search(r"\b(with|to|for|ng|into)\b", text_lower))
                has_allergen_words = any(w in text_lower for w in ["dairy", "lactose", "gluten", "allergen", "allergy", "allergies", "nuts", "egg", "pork", "shellfish"])
                is_remove_intent = bool(re.search(r"\b(remove|delete|cancel|alisin|tanggalin|bawas|ibawas|alising|tanggal|only\s+1\s+item|only\s+item|remove\s+that|remove\s+it|cancel\s+that|alisin\s+yan|tanggalin\s+yan)\b", text_lower))

                if is_clear_intent and not any(tc.get("function", {}).get("name") == "clear_cart" for tc in tool_calls):
                    tool_calls = [{"function": {"name": "clear_cart", "arguments": {}}}]
                elif is_replace_intent and not any(tc.get("function", {}).get("name") == "replace_order_item" for tc in tool_calls):
                    rep_match = re.search(r"(?:replace|swap|substitute|palitan|change)\s+(.*?)\s+(?:with|to|for|ng|into)\s+(.*?)(?:\s+(?:instead|cause|kasi|because|please|po)|\s*$)", text_lower)
                    raw_old = rep_match.group(1).strip() if rep_match else "burger"
                    raw_new = rep_match.group(2).strip() if rep_match else text_lower
                    tool_calls = [{"function": {"name": "replace_order_item", "arguments": {"old_item": raw_old, "new_item": raw_new, "qty": 1}}}]
                elif has_allergen_words and not any(tc.get("function", {}).get("name") == "check_item_allergen" for tc in tool_calls):
                    found_a = find_dishes_in_text(user_input)
                    t_dish = found_a[0][0]["name"] if found_a else "Perpetual Classic Burger"
                    t_allergen = "Dairy"
                    for a in ["Gluten", "Nuts", "Egg", "Pork", "Shellfish", "Dairy"]:
                        if a.lower() in text_lower:
                            t_allergen = a
                            break
                    tool_calls = [{"function": {"name": "check_item_allergen", "arguments": {"item_name": t_dish, "allergen": t_allergen}}}]
                elif is_remove_intent and not any(tc.get("function", {}).get("name") == "remove_item_from_order" for tc in tool_calls):
                    cart_c = db_tools.view_current_order(self.session_id, db_path=self.db_path)
                    cart_i = cart_c.get("line_items", [])
                    if cart_i:
                        target_name = cart_i[0]["name"]
                        found_r = find_dishes_in_text(user_input)
                        if found_r:
                            target_name = found_r[0][0]["name"]
                        tool_calls = [{"function": {"name": "remove_item_from_order", "arguments": {"item_name": target_name}}}]
                elif not tool_calls and not check_ambiguous_category_request(user_input) and any(w in text_lower for w in ["add", "buy", "order", "pabili", "bilhin", "pakidagdag"]):
                    found_o = find_dishes_in_text(user_input)
                    if found_o:
                        tool_calls = [{"function": {"name": "add_item_to_order", "arguments": {"item_name": found_o[0][0]["name"], "qty": found_o[0][1], "instructions": found_o[0][2]}}}]

                if not tool_calls:
                    final_content = raw_content
                    break

                if not assistant_msg.get("tool_calls"):
                    assistant_msg = {"role": "assistant", "content": "", "tool_calls": tool_calls}
                self.messages.append(assistant_msg)

                for tc in tool_calls:
                    fn_info = tc.get("function", {})
                    fn_name = fn_info.get("name", "")
                    fn_args = fn_info.get("arguments", {})
                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except Exception:
                            fn_args = {}

                    call_sig = f"{fn_name}:{json.dumps(fn_args, sort_keys=True)}"
                    if call_sig in executed_signatures:
                        continue
                    executed_signatures.add(call_sig)

                    tool_result = self.execute_tool(fn_name, fn_args)
                    turn_tool_calls.append({
                        "tool": fn_name,
                        "arguments": fn_args,
                        "result": tool_result,
                    })

                    tool_msg = {
                        "role": "tool",
                        "name": fn_name,
                        "content": json.dumps(tool_result, ensure_ascii=False)
                    }
                    self.messages.append(tool_msg)

                # Tool executed and appended to self.messages.
                # After executing tool calls for this turn, break so we can synthesize natural speech.
                break

            # If tools were executed, invoke Ollama to synthesize a natural conversational response
            # using machine learning based on the tool outputs in self.messages
            if turn_tool_calls and not final_content:
                try:
                    synthesis_payload = {
                        "model": self.model,
                        "messages": self.messages,
                        "stream": False,
                        "keep_alive": "24h",
                        "options": {
                            "temperature": 0.35,
                            "num_ctx": 2048,
                        }
                    }
                    synth_resp = self._call_ollama_chat(synthesis_payload, timeout=60)
                    synth_msg = synth_resp.get("message", {})
                    final_content = synth_msg.get("content", "").strip()
                    s_thinking = synth_msg.get("thinking", "")
                    if s_thinking:
                        model_thinking += ("\n" + s_thinking if model_thinking else s_thinking)
                except Exception:
                    pass

            clean_text = re.sub(r"<think>.*?</think>", "", final_content, flags=re.DOTALL).strip()

            # Absolute fallback ONLY if the LLM produced zero text:
            if not clean_text:
                fallback_speech = self._ensure_critical_tool_execution(user_input, turn_tool_calls)
                if fallback_speech:
                    clean_text = fallback_speech

            if not clean_text and turn_tool_calls:
                synthesis_parts = []
                rendered_tools = set()
                for tc in turn_tool_calls:
                    t_name = tc["tool"]
                    t_res = tc["result"]
                    if t_name in rendered_tools and t_name == "view_current_order":
                        continue
                    rendered_tools.add(t_name)
                    if t_name == "confirm_order_and_deduct_stock" and t_res.get("status") == "success":
                        synthesis_parts.append(
                            f"Order confirmed! Receipt #{t_res.get('receipt_id')} sent to kitchen display for Table #{t_res.get('table_no')}. Subtotal: ₱{t_res.get('subtotal'):.2f}, Total: ₱{t_res.get('grand_total'):.2f}."
                        )
                    elif t_name == "view_current_order" and t_res.get("status") == "success":
                        items_desc = ", ".join([f"{it['quantity']}x {it['name']} (₱{it['line_total']:.2f})" for it in t_res.get("line_items", [])])
                        if items_desc:
                            synthesis_parts.append(f"Current cart for Table #{t_res.get('table_no')}: {items_desc}. Total: ₱{t_res.get('grand_total'):.2f}.")
                        else:
                            synthesis_parts.append("Your order tab is currently empty.")
                    elif t_name == "check_item_allergen":
                        it_name = t_res.get("item_name", "the dish")
                        al_name = t_res.get("allergen", "this allergen")
                        has_al = t_res.get("contains_allergen")
                        if has_al:
                            synthesis_parts.append(f"Yes, {it_name} contains {al_name} (Declared allergens: {t_res.get('declared_allergens', 'None')}).")
                        else:
                            synthesis_parts.append(f"Good news! {it_name} does not contain {al_name}.")
                    elif t_name == "add_item_to_order":
                        if t_res.get("status") == "success":
                            synthesis_parts.append(t_res.get("message", "Added item to your order tab."))
                        elif t_res.get("status") == "out_of_stock":
                            synthesis_parts.append(f"{t_res.get('message')} {t_res.get('suggestion', '')}".strip())
                    elif t_name == "replace_order_item":
                        synthesis_parts.append(t_res.get("message", "Replaced item in your order tab."))
                    elif t_name == "clear_cart":
                        synthesis_parts.append(t_res.get("message", "Cleared all items from your order tab."))
                    elif t_name == "remove_item_from_order":
                        synthesis_parts.append(t_res.get("message", "Removed item from your cart."))
                    elif t_name == "update_item_quantity":
                        synthesis_parts.append(t_res.get("message", "Updated item quantity."))
                    elif t_res.get("message") and t_res["message"] != "Updated.":
                        synthesis_parts.append(t_res["message"])
                clean_text = " ".join(synthesis_parts)

            if not clean_text:
                clean_text = "Mabuhay! How can I assist you with your Perpetual Bistro order?"

            # Deduplicate consecutive identical paragraphs
            paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
            deduped = []
            for p in paragraphs:
                if not deduped or p != deduped[-1]:
                    deduped.append(p)
            clean_text = "\n\n".join(deduped)

            if not clean_text:
                clean_text = final_content.strip() or "How can I assist you with your Perpetual Bistro order?"

            self.messages.append({"role": "assistant", "content": clean_text})

            cart_state = db_tools.view_current_order(self.session_id, db_path=self.db_path)
            turn_data = {
                "session_id": self.session_id,
                "table_no": self.table_no,
                "fsm_state": self.state,
                "user_input": user_input,
                "assistant_response": clean_text,
                "thinking": model_thinking.strip(),
                "tool_calls": turn_tool_calls,
                "engine": "ollama",
                "cart": cart_state,
            }
            self.audit_log.append({"type": "dialogue_turn", "data": turn_data})
            return turn_data

        except Exception as e:
            # Seamless fallback to deterministic dialogue engine
            return self._fallback_dialogue_turn(user_input, error_reason=str(e))

    def interact(self, user_input: str) -> str:
        """Processes customer dialogue turn and returns assistant response text."""
        turn = self.process_turn(user_input)
        return turn.get("assistant_response", "")


    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Returns the full message list for auditing and logging."""
        return self.messages

    def export_audit_log(self, filepath: str) -> None:
        """Saves session audit trail to JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": self.session_id,
                "table_no": self.table_no,
                "fsm_state": self.state,
                "model": self.model,
                "audit_trail": self.audit_log,
                "final_cart": db_tools.view_current_order(self.session_id, db_path=self.db_path)
            }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import cli_waiter
    cli_waiter.run_cli_waiter()
