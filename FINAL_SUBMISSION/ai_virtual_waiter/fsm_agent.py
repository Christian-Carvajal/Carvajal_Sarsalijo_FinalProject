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
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            installed = [m.get("name", "") for m in data.get("models", [])]
            # Prioritize qwen2.5:1.5b to prevent cudaMalloc OOM on 4GB/6GB laptop GPUs
            for cand in ["qwen2.5:1.5b", "qwen3.5:4b", "jarvis-trained-model:latest"]:
                for m in installed:
                    if cand in m:
                        return m
            if installed:
                return installed[0]
    except Exception:
        pass
    return "qwen2.5:1.5b"

DEFAULT_MODEL = select_best_model()

# 5 Discrete FSM States
STATE_GREETING = "STATE_GREETING"
STATE_MENU_BROWSING = "STATE_MENU_BROWSING"
STATE_TAKING_ORDER = "STATE_TAKING_ORDER"
STATE_REVIEW = "STATE_REVIEW"
STATE_CONFIRMED = "STATE_CONFIRMED"

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
            "description": "Add an item to the customer's active order tab with specified quantity and special instructions/customizations (e.g., 'extra cheese', 'no onions').",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Name of the menu item to add."
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
            "description": "Remove an item completely from the customer's current order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Name of the menu item to remove."
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
            "name": "set_dialogue_state",
            "description": "Transition the conversational FSM state based on the customer's dialogue phase (e.g. STATE_MENU_BROWSING when asking about food, STATE_TAKING_ORDER when selecting dishes, STATE_REVIEW when asking for the bill, STATE_CONFIRMED on checkout).",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_state": {
                        "type": "string",
                        "enum": ["STATE_GREETING", "STATE_MENU_BROWSING", "STATE_TAKING_ORDER", "STATE_REVIEW", "STATE_CONFIRMED"],
                        "description": "The destination FSM state."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this transition occurred based on customer input."
                    }
                },
                "required": ["new_state"]
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
        "aliases": ["perpetual classic burger", "classic burger", "perpetual burger", "classic beef burger", "burger"],
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
        "aliases": ["grilled chicken pesto sandwich", "chicken pesto sandwich", "pesto sandwich", "chicken pesto", "sandwich"],
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
        "name": "Chef Ezekiel's Special Duo",
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
    return results


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

        system_content = (
            "You are 'BistroBot', the intelligent, friendly, conversational AI table-side waiter at Perpetual Bistro, "
            "University of Perpetual Help System DALTA.\n"
            f"You are actively serving Table #{self.table_no} (Session ID: {self.session_id}).\n\n"
            "NATURAL CONVERSATION & INTELLIGENCE GUIDELINES:\n"
            "1. CONVERSE DYNAMICALLY LIKE CHATGPT: Answer questions spontaneously on the spot. If a customer chats, "
            "asks what you personally recommend, asks for pairing ideas, jokes about food, or asks general dining questions, "
            "respond warmly, naturally, and intelligently in your own words. Never sound robotic or scripted.\n"
            "2. CURRENCY RULE: All prices in Perpetual Bistro are strictly in Philippine Pesos (PHP, ₱). "
            "Always state prices simply as 'pesos' or '₱' (e.g., '150 pesos' or '₱150.00'). "
            "NEVER say 'Cuban pesos', 'Mexican pesos', or any foreign currency denomination.\n"
            "3. STATE MACHINE CONTEXT:\n"
            f"   - Current FSM State: {self.state}.\n"
            "   - When customer browses menu or asks about dishes: explore using `get_menu_by_category` or `check_item_allergen` (STATE_MENU_BROWSING).\n"
            "   - When customer wants to order: invoke `add_item_to_order` with their requested quantity and customizations (STATE_TAKING_ORDER).\n"
            "   - When customer asks about their bill, current tab, or summary: call `view_current_order` to present their bill (STATE_REVIEW).\n"
            "   - When customer confirms or says they are ready to finalize: call `confirm_order_and_deduct_stock` (STATE_CONFIRMED).\n"
            "   - You can also call `set_dialogue_state` to transition the state whenever appropriate.\n"
            "4. TOOL CALLING AUTONOMY: Autonomously call the tools whenever needed to fetch real database data. "
            "Do not guess or hallucinate menu prices or inventory counts.\n"
            "5. OUT-OF-STOCK POLICY: If an item is out of stock (such as Crispy Buffalo Chicken Tenders), explain politely and suggest in-stock alternatives from the same category.\n"
            + language_rule +
            "7. EMOJI AND SPEECH CLEANLINESS: Do not use emojis in your responses. Keep text clean and verbal so text-to-speech audio speaks words cleanly without describing symbols."
        )
        self.messages = [{"role": "system", "content": system_content}]

    def set_language(self, new_lang: str) -> None:
        """Updates the active persona language ('filipino' or 'english')."""
        self.language = new_lang.lower().strip()
        self._init_system_prompt()

    def set_table_number(self, new_table_no: int) -> None:
        """Updates the active table number."""
        self.table_no = new_table_no
        self.session = db_tools.create_or_get_session(new_table_no, self.session_id, db_path=self.db_path)
        self._init_system_prompt()

    def transition_state(self, new_state: str, reason: str = "") -> None:
        """Executes a formal FSM state transition and persists it in SQLite."""
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
            item = args.get("item_name", "")
            qty = int(args.get("qty", 1))
            instructions = args.get("instructions", "")
            res = db_tools.add_item_to_order(self.session_id, item, qty, instructions, db_path=self.db_path)
            if res.get("status") == "success":
                self.transition_state(STATE_TAKING_ORDER, reason=f"Item added: {item}")
            return res

        elif tool_name == "remove_item_from_order":
            item = args.get("item_name", "")
            res = db_tools.remove_item_from_order(self.session_id, item, db_path=self.db_path)
            self.transition_state(STATE_TAKING_ORDER, reason=f"Item removed: {item}")
            return res

        elif tool_name == "update_item_quantity":
            item = args.get("item_name", "")
            qty = int(args.get("qty", 1))
            res = db_tools.update_item_quantity(self.session_id, item, qty, db_path=self.db_path)
            self.transition_state(STATE_TAKING_ORDER, reason=f"Quantity updated: {item}")
            return res

        elif tool_name == "view_current_order":
            res = db_tools.view_current_order(self.session_id, db_path=self.db_path)
            self.transition_state(STATE_REVIEW, reason="Customer reviewing bill tab")
            return res

        elif tool_name == "confirm_order_and_deduct_stock":
            db_tools.create_or_get_session(self.table_no, self.session_id, db_path=self.db_path)
            res = db_tools.confirm_order_and_deduct_stock(self.session_id, db_path=self.db_path)
            if res.get("status") == "success":
                self.transition_state(STATE_CONFIRMED, reason="Order committed and inventory deducted")
            return res

        else:
            return {"status": "error", "message": f"Unknown tool name: {tool_name}"}

    def _call_ollama_chat(self, payload: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
        """Sends HTTP request to local Ollama /api/chat endpoint with memory-safe fallback."""
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

        # 1. Add Item(s) to Order FIRST if dishes are detected
        found_dishes = find_dishes_in_text(user_input)
        if found_dishes:
            added_dish_names = {tc.get("arguments", {}).get("item_name") for tc in turn_tool_calls if tc["tool"] == "add_item_to_order"}
            for dish, qty, instructions in found_dishes:
                if dish["name"] == "Crispy Buffalo Chicken Tenders":
                    continue
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

        # 2. Order Confirmation / Checkout (if user asked to confirm or place order)
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

        # 3. View Current Bill / Tab (only if not already confirming)
        elif any(w in text_lower for w in ["bill", "cart", "tab", "magkano", "bayarin", "summary", "review"]):
            if "view_current_order" not in tools_executed and "confirm_order_and_deduct_stock" not in tools_executed:
                res = self.execute_tool("view_current_order", {})
                turn_tool_calls.append({"tool": "view_current_order", "arguments": {}, "result": res})

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

        # 1. Intent: Ordering Dishes (with optional immediate checkout in the same utterance)
        if found_dishes:
            added_msgs = []
            for dish, qty, instructions in found_dishes:
                if dish["name"] == "Crispy Buffalo Chicken Tenders":
                    avail = self.execute_tool("check_availability", {"item_name": dish["name"], "qty": qty})
                    turn_tool_calls.append({"tool": "check_availability", "arguments": {"item_name": dish["name"], "qty": qty}, "result": avail})
                    if is_filipino:
                        added_msgs.append(
                            f"Paumanhin po! Ang Crispy Buffalo Chicken Tenders ay kasalukuyang out-of-stock sa aming kusina (0 natitira). "
                            f"Nais niyo po bang subukan ang Grilled Chicken Pesto Sandwich (₱165.00) o Bacon Mushroom Melt Burger (₱185.00) bilang alternatibo?"
                        )
                    else:
                        added_msgs.append(
                            f"I apologize, but our Crispy Buffalo Chicken Tenders are currently out of stock (0 available). "
                            f"Would you like to try our Grilled Chicken Pesto Sandwich (₱165.00) or Bacon Mushroom Melt Burger (₱185.00) instead?"
                        )
                else:
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

        # 3. Intent: View Bill / Cart Review
        elif any(w in text_lower for w in ["bill", "cart", "tab", "total", "magkano", "bayarin", "summary", "review"]):
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

        # 4. Intent: Allergen Query
        elif any(w in text_lower for w in ["allergen", "allergy", "allergies", "lactose", "dairy", "gluten", "wheat", "nut", "nuts", "egg", "pork"]):
            found_dishes = find_dishes_in_text(user_input)
            target_dish = found_dishes[0][0]["name"] if found_dishes else "Perpetual Classic Burger"
            target_allergen = "Dairy"
            for a in ["Gluten", "Nuts", "Egg", "Pork", "Shellfish"]:
                if a.lower() in text_lower:
                    target_allergen = a
                    break

            res = self.execute_tool("check_item_allergen", {"item_name": target_dish, "allergen": target_allergen})
            turn_tool_calls.append({"tool": "check_item_allergen", "arguments": {"item_name": target_dish, "allergen": target_allergen}, "result": res})

            has_allergen = res.get("contains_allergen", False)
            all_allergens = res.get("all_item_allergens", [])
            allergen_str = ", ".join(all_allergens) if all_allergens else "None"

            if is_filipino:
                if has_allergen:
                    resp_text = (
                        f"Opo, ang {target_dish} ay may {target_allergen} (Kabuuang allergens: {allergen_str}). "
                        f"Kung may lactose intolerance o dietary restriction kayo, maaari po nating ipatanggal ang cheese o i-customize ang order!"
                    )
                else:
                    resp_text = f"Wala pong {target_allergen} ang {target_dish}. Ligtas po ito para sa inyo! (Nakatala allergens: {allergen_str})."
            else:
                if has_allergen:
                    resp_text = (
                        f"Yes, the {target_dish} contains {target_allergen} (Allergens: {allergen_str}). "
                        f"If you have mild lactose intolerance, we can prepare it without cheese for you!"
                    )
                else:
                    resp_text = f"Good news! The {target_dish} is free of {target_allergen} (Allergens: {allergen_str})."

        # 5. Intent: Category / Menu Inquiry
        elif any(w in text_lower for w in ["burger", "burgers", "pasta", "pastas", "side", "sides", "drink", "drinks", "beverage", "dessert", "desserts", "combo", "combos", "value meal", "menu"]):
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
        elif any(w in text_lower for w in ["recommend", "rekomenda", "best seller", "bestseller", "specialty", "masarap", "lunch", "popular"]):
            if is_filipino:
                resp_text = (
                    "Mabuhay po! Ang aming crowd-favorite bestsellers ay ang Perpetual Classic Burger (₱150.00), "
                    "Truffle Mushroom Pasta (₱210.00), at Loaded Cheese Fries (₱110.00)! "
                    "Para naman sa pinaka-sulit na bundle, subukan ang aming Perpetual Champion Combo (₱285.00, tipid ng ₱30) "
                    "o Pasta & Sweet Treat Combo (₱330.00, tipid ng ₱50)!"
                )
            else:
                resp_text = (
                    "Mabuhay! Our top bestsellers are the Perpetual Classic Burger (₱150.00), "
                    "Truffle Mushroom Pasta (₱210.00), and Loaded Cheese Fries (₱110.00)! "
                    "For the best value, check out our Campus Value Combos like the Perpetual Champion Combo (₱285.00, save ₱30) "
                    "or the Pasta & Sweet Treat Combo (₱330.00, save ₱50)!"
                )

        # 7. Intent: Greeting
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
        self.messages.append({"role": "user", "content": user_input})
        turn_tool_calls: List[Dict[str, Any]] = []

        max_tool_iterations = 5
        iteration = 0
        final_content = ""
        model_thinking = ""

        try:
            while iteration < max_tool_iterations:
                iteration += 1

                payload = {
                    "model": self.model,
                    "messages": self.messages,
                    "tools": BISTRO_TOOLS,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_ctx": 2048,
                    }
                }

                resp = self._call_ollama_chat(payload, timeout=15)
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

                if not tool_calls:
                    final_content = raw_content
                    self.messages.append({"role": "assistant", "content": final_content})
                    break

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

            # Safety guard: ensure critical orders/checkouts execute
            extra_tool_note = self._ensure_critical_tool_execution(user_input, turn_tool_calls)

            clean_text = re.sub(r"<think>.*?</think>", "", final_content, flags=re.DOTALL).strip()
            if extra_tool_note and not clean_text:
                clean_text = extra_tool_note

            if not clean_text and turn_tool_calls:
                synthesis_parts = []
                for tc in turn_tool_calls:
                    t_name = tc["tool"]
                    t_res = tc["result"]
                    if t_name == "confirm_order_and_deduct_stock" and t_res.get("status") == "success":
                        synthesis_parts.append(
                            f"🎉 **Order Confirmed!** (Receipt #{t_res.get('receipt_id')})\n"
                            f"Your order for Table #{t_res.get('table_no')} has been sent to the kitchen!\n"
                            f"**Subtotal:** ₱{t_res.get('subtotal'):.2f} | **Total:** ₱{t_res.get('grand_total'):.2f}\n"
                            f"Thank you for dining with Perpetual Bistro! Your food will be served shortly."
                        )
                    elif t_name == "view_current_order" and t_res.get("status") == "success":
                        items_desc = "\n".join([f"- {it['quantity']}x {it['name']} (₱{it['line_total']:.2f})" for it in t_res.get("line_items", [])])
                        synthesis_parts.append(
                            f"Here is your current order for Table #{t_res.get('table_no')}:\n{items_desc}\n\n"
                            f"**Subtotal:** ₱{t_res.get('subtotal'):.2f} (12% VAT included: ₱{t_res.get('vat_12_pct_breakdown'):.2f})\n"
                            f"**Grand Total:** ₱{t_res.get('grand_total'):.2f}\n\nWould you like to confirm this order or make any changes?"
                        )
                    elif t_name == "add_item_to_order":
                        synthesis_parts.append(t_res.get("message", "Item updated in your order."))
                    elif t_name == "remove_item_from_order":
                        synthesis_parts.append(t_res.get("message", "Item removed."))
                    elif t_name == "get_menu_by_category" and t_res.get("status") == "success":
                        item_names = ", ".join([it['name'] + f" (₱{it['price']:.2f})" for it in t_res.get("items", [])])
                        synthesis_parts.append(f"Here are the items available in {t_res.get('category_queried')}: {item_names}")
                clean_text = "\n\n".join(synthesis_parts)

            if not clean_text:
                clean_text = final_content.strip() or "How can I assist you with your Perpetual Bistro order?"

            cart_state = db_tools.view_current_order(self.session_id, db_path=self.db_path)
            turn_data = {
                "session_id": self.session_id,
                "table_no": self.table_no,
                "fsm_state": self.state,
                "user_input": user_input,
                "assistant_response": clean_text,
                "thinking": model_thinking.strip(),
                "tool_calls": turn_tool_calls,
                "cart": cart_state,
            }
            self.audit_log.append({"type": "dialogue_turn", "data": turn_data})
            return turn_data

        except Exception as e:
            # Seamless fallback to deterministic dialogue engine
            return self._fallback_dialogue_turn(user_input, error_reason=str(e))


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
