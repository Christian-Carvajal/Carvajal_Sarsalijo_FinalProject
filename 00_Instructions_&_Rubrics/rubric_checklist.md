# Rubric Compliance Checklist: Lesson 7 Final Project
## AI Virtual Waiter for Perpetual Bistro Using Ollama and Qwen
**Course:** BSCS 3112 - Artificial Intelligence (Lab)  
**Instructor:** Prof. Rob Malitao  
**Students / Team Members:** Christian Ezekiel L. Carvajal & John Miko P. Sarsalijo (BSCS - DS 3rd Year)  
**Execution Window:** October 19 – November 6, 2026  
**Due Dates:** Milestone 1 (Oct 28), Milestone 2 (Nov 6), Live Defense (Nov 7)

---

## 1. Required Deliverables & File Architecture (Page 3)
- [ ] `ai_virtual_waiter/` root folder
  - [ ] `init_restaurant_db.sql`: SQLite DDL schema and authentic UPHSD Bistro seed dataset (menu_items, table_sessions, cart_items, kitchen_tickets)
  - [ ] `init_db.py`: Database initialization script executing `init_restaurant_db.sql` into `restaurant.db`
  - [ ] `db_tools.py`: SQLite CRUD tool functions (`get_menu_items`, `add_item_to_order_tool`, `get_cart_summary`, `confirm_order_and_deduct_stock`, inventory guards)
  - [ ] `fsm_controller.py`: Finite State Machine class enforcing unbypassable states (`GREETING`, `ORDERING`, `REVIEW`, `CONFIRMED`)
  - [ ] `speech_engine.py`: Decoupled STT and TTS speech wrappers (offline local STT + TTS adapters)
  - [ ] `app.py`: Main application entry point orchestrating UI, Speech Engine, FSM Controller, Ollama/Qwen, and SQLite Tools
  - [ ] `project_documentation.md`: System architecture, state transition diagrams, tandem pair collaboration log, and user guide
  - [ ] `ui_dashboard/`: Web or desktop UI
    - [ ] `templates/`: HTML templates (`index.html`)
    - [ ] `static/`: CSS styling, brand assets, photos (`campus_hero.jpg`, `chef_ezekiel.jpg`)

---

## 2. Assessment Rubric Breakdown (100 Points Total)

### A. System Architecture & Integration (30% Weight)
- [ ] **100% Local Execution**: Runs locally using Ollama (`qwen2.5`), Python 3.10+, SQLite 3, and local speech modules without cloud API dependencies.
- [ ] **Multi-Modal Synchronization**: Seamlessly integrates voice I/O (STT/TTS), FSM state machine, LLM tool calling, SQLite persistence, and UI dashboard.
- [ ] **Decoupled Architecture**: Hardware audio drivers (speech engine) and browser/desktop graphics are cleanly separated from core reasoning and state logic.
- [ ] **Zero Latency Spikes / Audio Buffering**: Speech processing and transitions execute with smooth latency and no unhandled crashes.

### B. Agentic Tool Calling & Data Integrity (25% Weight)
- [ ] **Structured Tool Calling**: Qwen 2.5 reliably receives JSON tool signatures and outputs structured tool calls without hallucinating dishes or prices.
- [ ] **Database Integrity**: Tool calls query and mutate `menu_items`, `cart_items`, and `table_sessions` in SQLite with ACID transactional safety (`BEGIN TRANSACTION ... COMMIT / ROLLBACK`).
- [ ] **Accurate Tab & Tax Calculations**: Line items, quantities, special instructions, subtotal, and grand totals calculate with 100% accuracy.
- [ ] **Multi-Item & Combo Handling**: Reliably handles complex multi-item orders, custom instructions, and campus combo packages.

### C. FSM State Management & Guardrails (20% Weight)
- [ ] **Unbypassable Flow**: Strictly enforces `GREETING` → `ORDERING` → `REVIEW` → `CONFIRMED`.
- [ ] **Illegal State Jump Prevention**: Prevents skipping directly from `GREETING` to `CONFIRMED`, and blocks checkout if the cart is empty.
- [ ] **Post-Confirmation Lock**: Completely locks cart modifications once in `CONFIRMED` state (rejection of edits/cancellations).
- [ ] **Programmatic Inventory Guardrails**: Hard code guards in `db_tools.py` block orders when `stock_qty = 0` or requested quantity exceeds available stock, proactively suggesting in-stock alternatives.
- [ ] **Out-of-Domain Guardrails**: Politely rejects off-topic queries (e.g., weather, homework, general knowledge) and guides customer back to dining.

### D. User Interface & Experience (UI/UX) (15% Weight)
- [ ] **Real-Time Order Cart**: Clearly displays itemized line items, quantities, modifications, unit prices, line totals, and grand total.
- [ ] **Live Dialogue & Voice Transcript**: Visual feed showing real-time customer speech and AI waiter responses.
- [ ] **Human-in-the-Loop Controls**: Interactive manual touch controls (+, -, Delete item) to modify cart alongside voice commands.
- [ ] **UPHSD Institutional Design**: Branded with official Perpetual Maroon (`#7B1113`) & Imperial Gold (`#D4AF37`) palette, Cameo Seal, and responsive layout bounds (`scrollWidth <= clientWidth`).

### E. Live Defense & Tandem Collaboration (10% Weight)
- [ ] **Tandem Pair Collaboration**: Clear division of labor between Christian Ezekiel L. Carvajal and John Miko P. Sarsalijo with balanced driver/navigator roles documented in `project_documentation.md`.
- [ ] **Offline Readiness**: System tested and ready to execute live on a local laptop without active internet connection.
- [ ] **Defense Stress Tests Prepared**:
  - [ ] Stress Test 1: Ordering an out-of-stock item (system intercepts, rejects commit, and offers alternatives).
  - [ ] Stress Test 2: Attempting to modify or cancel an order after reaching `CONFIRMED` state (system politely refuses and explains order has been routed to kitchen).
