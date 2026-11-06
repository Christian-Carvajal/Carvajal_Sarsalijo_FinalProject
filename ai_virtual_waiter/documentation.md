# Project Documentation: AI Virtual Waiter for Perpetual Bistro
## Multi-Modal Conversational Agent with Local LLM Tool Calling and Finite State Machine Control
**Course:** BSCS 3112 — Artificial Intelligence (Lab)  
**Institution:** University of Perpetual Help System DALTA — Las Piñas Campus  
**Instructor:** Prof. Rob Malitao  
**Authors / Team Members:**
- **Christian Ezekiel L. Carvajal** — BSCS - DS 3rd Year (Lead Systems Architect & Driver)
- **John Miko P. Sarsalijo** — BSCS - DS 3rd Year (Fullstack Lead & Navigator)
**Execution Window:** October 19 – November 6, 2026  
**Final Project Due Date:** November 6, 2026 | **Live Defense:** November 7, 2026

---

## 📹 Video Walkthrough Demonstration (Milestone 2 Requirement)
- **Demonstration Link:** `https://youtu.be/[SUBMIT_YOUR_3_MINUTE_WALKTHROUGH_VIDEO_LINK_HERE]` *(Student 3–5 Minute End-to-End Voice & Kiosk Recording Placeholder)*
- **Walkthrough Outline:**
  1. **Table Greeting & Session Init (0:00 - 0:45)**: Kiosk boots on Table #1, vocal welcome via Taglish TTS.
  2. **Menu Browsing & Allergen Inquiries (0:45 - 1:45)**: Dynamic querying of burger offerings and lactose/gluten allergen checks.
  3. **Multi-Turn Ordering & Customization (1:45 - 2:30)**: Voice-adding dishes with instructions (`Extra Cheese, No Onions`).
  4. **Programmatic Inventory Guardrail (2:30 - 3:15)**: Out-of-stock rejection on Crispy Buffalo Chicken Tenders with automated menu substitution recommendations.
  5. **Human-in-the-Loop Override & KDS Checkout (3:15 - 4:15)**: Manual quantity manipulation (+, -, Delete), transaction confirmation, digital receipt generation, and KDS fulfillment routing.

---

## 1. Executive Summary & Business Scenario

Perpetual Bistro, the primary on-campus dining facility at the University of Perpetual Help System DALTA (UPHSD), requires an autonomous, table-side digital ordering kiosk to streamline dining operations during peak campus meal hours. 

To guarantee absolute institutional privacy, zero cloud subscription overhead, and operational resilience against campus network outages, the system runs **100% locally on edge laptop hardware**. It synthesizes:
1. **Local Speech Processing (STT / TTS)**: Voice input capture and Taglish neural voice synthesis (`speech_engine.py`).
2. **Conversational State Machine (FSM)**: Strict, deterministic dialogue lifecycle enforcing business rules (`fsm_controller.py`).
3. **Agentic Tool Calling**: Dynamic binding between local Ollama (`qwen2.5`) reasoning and SQLite relational tools.
4. **Relational Data Integrity**: Real-time ACID transactions managing menu inventory, active cart tabs, and Kitchen Display System (KDS) routing (`db_tools.py` -> `restaurant.db`).

```
+-----------------------------------------------------------------------------------------+
|                                1. I/O SURFACE LAYER                                     |
|  [ Microphone STT ]  <--->  [ Web Kiosk & KDS UI ]  <--->  [ Speaker Neural TTS ]      |
+-----------------------------------------------------------------------------------------+
                                           │
                                           ▼
+-----------------------------------------------------------------------------------------+
|                             2. CORE AGENT ENGINE LAYER                                  |
|   +------------------------------------+      +-------------------------------------+   |
|   |       FSM State Controller         | <--> |        Local Ollama / Qwen 2.5      |   |
|   | (GREETING/ORDERING/REVIEW/CONFIRM) |      |      (Reasoning & Intent Parser)    |   |
|   +------------------------------------+      +-------------------------------------+   |
+-----------------------------------------------------------------------------------------+
                                           │
                                           ▼
+-----------------------------------------------------------------------------------------+
|                        3. RELATIONAL PERSISTENCE LAYER (SQLite)                         |
|   [ menu_items (21 items) ]   [ table_sessions ]   [ order_line_items ]  [ kitchen_tickets ] |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Layer-by-Layer System Architecture

### Layer 1: The I/O Surface Layer (`speech_engine.py` & `ui_dashboard/`)
This external boundary interacts directly with diner hardware at the dining table:
- **Speech-to-Text (STT)**: Microphone capture wrapper via `SpeechRecognition` or browser Web Speech API with silence threshold detection (auto-sends query after 1.8s of silence).
- **Text-to-Speech (TTS)**: Phonetic normalization engine transforming currency symbols (`₱150` -> `150 pesos`) and school acronyms (`UPHSD` -> `Perpetual Help System DALTA`) into natural Philippine English/Taglish audio.
- **Web Tablet Dashboard**: Built with responsive HTML5/CSS3 featuring the official University Maroon (`#7B1113`) & Gold (`#D4AF37`) palette, multi-view catalog, dish customization modal, live cart table, manual override controls (+, -, Delete), and Back-of-House Kitchen Display System (KDS).

### Layer 2: Core Virtual Waiter Agent Engine (`fsm_controller.py`)
This middle processing layer bridges natural language reasoning with programmatic guardrails:
- **Finite State Machine (FSM)**: Enforces the 5 discrete operational states:
  - `STATE_GREETING`: Welcome customer and bind table number (Default: Table 1).
  - `STATE_MENU_BROWSING`: Provide menu listings, recommendations, and allergen facts.
  - `STATE_TAKING_ORDER`: Parse dish entities, quantities, and customizations (`No Onions`, `Extra Cheese`).
  - `STATE_REVIEW`: Present itemized tab with subtotal and tax breakdown for verification.
  - `STATE_CONFIRMED`: Lock cart, generate receipt, update inventory, and route kitchen ticket.
- **Local Ollama / Qwen 2.5**: Formulates structured tool calls from unconstrained conversational input without cloud latency. Dynamic fallback logic prioritizes lightweight models (`qwen2.5:1.5b`) to eliminate VRAM OOM errors on 4GB/6GB student laptops, paired with a deterministic regex parser for offline edge robustness.

### Layer 3: Relational Persistence Layer (`db_tools.py` & `restaurant.db`)
Single source of truth implemented in SQLite:
- `menu_items`: 21 authentic canteen offerings across 6 categories (Burgers, Pastas, Sides, Beverages, Desserts, Combos) including stock quantities (`stock_qty`) and allergen flags.
- `table_sessions`: Tracks table identification, creation timestamps, and active FSM state.
- `order_line_items` (and `cart_items` view): Tracks active tab line items, quantities, unit prices, and custom instructions.
- `kitchen_tickets`: Manages kitchen fulfillment queue (`PREPARING` -> `READY` -> `SERVED`) and payment records (Cash, GCash, Maya, RFID).

---

## 3. Tandem Collaboration & Task Matrix (Driver / Navigator Rotation)

| Milestone / Subsystem | Driver (Primary Author) | Navigator (Logic & Review) | Deliverable Artifact |
| :--- | :--- | :--- | :--- |
| **Milestone 1: Database & Tool CRUD** | Christian Ezekiel L. Carvajal | John Miko P. Sarsalijo | `restaurant_db.sql`, `init_db.py`, `db_tools.py` |
| **Milestone 1: FSM State Machine** | John Miko P. Sarsalijo | Christian Ezekiel L. Carvajal | `fsm_controller.py` |
| **Milestone 1: Decoupled Speech I/O** | Christian Ezekiel L. Carvajal | John Miko P. Sarsalijo | `speech_engine.py`, `cli_waiter.py` |
| **Milestone 2: UI Dashboard & KDS** | John Miko P. Sarsalijo | Christian Ezekiel L. Carvajal | `ui_dashboard/`, `index.html` |
| **Milestone 2: Main Application Pipeline** | Christian Ezekiel L. Carvajal | John Miko P. Sarsalijo | `app.py` |
| **Milestone 2: Report & Mathematical Proofs** | Christian Ezekiel L. Carvajal | John Miko P. Sarsalijo | `Carvajal_Sarsalijo_FinalProject_Report.pdf` |

---

## 4. Live Defense Edge Case Scenarios (Faculty Panel Protocol)

The system is fortified against the two mandatory defense stress tests:

1. **Stress Test 1: Out-of-Stock Item Rejection**
   - *Scenario*: Diner requests `Crispy Buffalo Chicken Tenders` (pre-seeded with `stock_qty = 0`).
   - *System Behavior*: `db_tools.check_availability()` intercepts the request before insertion. The engine responds with a polite Filipino/Taglish apology, leaves the cart uncharged, and suggests in-stock alternatives (Grilled Chicken Pesto Sandwich or Bacon Mushroom Melt).
2. **Stress Test 2: Post-Confirmation Modification Lock**
   - *Scenario*: Diner attempts to cancel or add items after the order reaches `STATE_CONFIRMED`.
   - *System Behavior*: FSM rejects tool execution. The agent informs the diner that the transaction has already been committed to `restaurant.db` and routed to the kitchen display.
