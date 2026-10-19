# AI Virtual Waiter for Perpetual Bistro (Lesson 7 Final Project)
## Multi-Modal Conversational Agent with Local LLM Tool Calling and Finite State Machine Control

**Course:** BSCS 3112 — Artificial Intelligence (Lab)  
**Institution:** University of Perpetual Help System DALTA — Las Piñas Campus  
**Instructor:** Prof. Rob Malitao  
**Authors / Student Pair:**
- **Christian Ezekiel L. Carvajal** — BSCS - DS 3rd Year (Lead Systems Architect & Driver)
- **John Miko P. Sarsalijo** — BSCS - DS 3rd Year (Fullstack Lead & Driver)

**Execution Window:** October 19 – November 6, 2026 | **Live Defense:** November 7, 2026

---

## 1. Project Overview & Business Scenario

Perpetual Bistro, the primary on-campus dining facility at the University of Perpetual Help System DALTA (UPHSD), deploys an autonomous, table-side digital ordering kiosk to streamline dining operations during peak campus meal hours. 

To guarantee absolute institutional privacy, zero cloud API costs, and operational resilience against campus network outages, the system runs **100% locally on edge laptop hardware** synthesizing:
1. **Local Speech Processing (STT / TTS)**: Voice input capture and Taglish neural voice synthesis.
2. **Conversational State Machine (FSM)**: Strict, deterministic dialogue lifecycle enforcing business rules (`GREETING` → `ORDERING` → `REVIEW` → `CONFIRMED`).
3. **Agentic Tool Calling**: Dynamic binding between local Ollama (`qwen2.5`) reasoning and SQLite relational tools.
4. **Relational Data Integrity**: Real-time ACID transactions managing menu inventory, active cart tabs, and Kitchen Display System (KDS) routing.

---

## 2. Repository Structure

```
.
├── ai_virtual_waiter/              # Core Application Source Code
│   ├── app.py                      # FastAPI application server & REST routing
│   ├── db_tools.py                 # SQLite CRUD, transactions & inventory guardrails
│   ├── fsm_controller.py           # Finite State Machine controller (5 states)
│   ├── speech_engine.py            # Decoupled STT & Taglish TTS speech engine
│   ├── init_db.py                  # Database setup with safe table drop & reseed
│   ├── init_restaurant_db.sql      # DDL schema & 21 Perpetual Bistro menu seeds
│   ├── restaurant.db               # Operational SQLite database
│   ├── project_documentation.md    # Architecture, state diagrams & tandem log
│   └── ui_dashboard/               # Tablet Kiosk UI & KDS board
│       ├── static/                 # Brand assets & culinary team photo
│       └── templates/index.html    # Single-page ordering kiosk
├── 00_Instructions_&_Rubrics/      # Official professor briefs & rubrics
├── 01_Work_In_Progress/            # Intermediate development snapshots
├── 02_Tests_&_Drafts/              # TDD test suites (contract & defense stress tests)
├── FINAL_SUBMISSION/               # Isolated clean submission bundle
└── Carvajal_Sarsalijo_FinalProject.zip # Compressed MS Teams submission deliverable
```

---

## 3. Quick-Start Guide

### Prerequisites
1. Local Ollama installed and running:
   ```bash
   ollama pull qwen2.5:1.5b
   ollama run qwen2.5:1.5b
   ```
2. Python 3.10+ with required dependencies:
   ```bash
   pip install fastapi uvicorn edge-tts requests pydantic jinja2
   ```

### Running the Kiosk
1. Reset fresh test database:
   ```bash
   python ai_virtual_waiter/init_db.py --reset
   ```
2. Launch tablet kiosk server:
   ```bash
   python ai_virtual_waiter/app.py --port 8000
   ```
3. Open browser at [http://localhost:8000](http://localhost:8000).

### Running Verification Tests
```bash
# Automated Production Contract Tests (16/16 Passed)
python "02_Tests_&_Drafts/test_production_contracts.py"

# Panel Defense Stress Tests (4/4 Passed)
python "02_Tests_&_Drafts/test_edge_cases.py"
```
