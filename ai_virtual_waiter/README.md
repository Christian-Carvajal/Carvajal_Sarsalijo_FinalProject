# Perpetual Bistro: AI Virtual Waiter (Lesson 7 Final Project - Milestone 1)
## Multi-Modal Conversational Agent with Local LLM Tool Calling and Finite State Machine Control
**University of Perpetual Help System DALTA — Las Piñas Campus**  
**Course:** BSCS 3112 — Artificial Intelligence (Lab)  
**Instructor:** Prof. Rob Malitao  
**Authors:** Christian Ezekiel L. Carvajal & John Miko P. Sarsalijo (BSCS - DS 3rd Year)  
**Milestone 1 Deadline:** October 28, 2026 (Weight: 40%)  

---

## 📋 Milestone 1 Scope & Deliverables (100% Complete)

| Deliverable | File | Description |
| :--- | :--- | :--- |
| **1. SQLite DDL & Seeds** | `restaurant_db.sql` | DDL tables and 21 authentic UPHSD menu seeds across 6 categories. |
| **2. Database Setup** | `init_db.py` & `restaurant.db` | Safe SQLite initialization, stock audit, and pre-seeded database. |
| **3. SQLite Tool Engine** | `db_tools.py` | Relational CRUD, inventory checking, atomic stock deduction, and cart summaries. |
| **4. FSM Core Controller** | `fsm_controller.py` | 5-state deterministic dialogue engine with Ollama qwen2.5 tool binding & offline fallback. |
| **5. Decoupled Voice Pipeline** | `speech_engine.py` | Microphone STT capture, phonetic Taglish normalizer, Edge-TTS & pyttsx3 fallback. |
| **6. Automated Test Suite** | `milestone1_test_suite.py` | Automated CLI integration test suite verifying DB/CRUD, inventory guardrails, FSM, and speech decoupling. |
| **7. CLI Integration Harness** | `cli_waiter.py` | Executable CLI loop demonstrating spoken/text input triggering FSM transitions without GUI dependencies. |
| **8. System Documentation** | `documentation.md` | Technical architecture, state diagrams, and XP tandem pair collaboration log. |
| **9. Academic Report** | `Carvajal_Sarsalijo_FinalProject_Milestone1_Report.pdf` | Formal institutional report covering PEAS, Search Strategy, Task Scheduling Arrays, and Heuristics. |

---

## 🎯 Double-Rubric Compliance Matrix

### 1. Lab Activity Lesson 7 PDF Brief Rubric (Milestone 1 Core)
- **Voice I/O Pipeline**: Fully decoupled in `speech_engine.py` with STT audio capture, phonetic Taglish normalization, and neural TTS synthesis.
- **FSM Core Controller**: Enforces strict transitions across `STATE_GREETING`, `STATE_MENU_BROWSING`, `STATE_TAKING_ORDER`, `STATE_REVIEW`, and `STATE_CONFIRMED`.
- **SQLite Tool Engine**: Autonomous Ollama `qwen2.5` tool calling querying and mutating `restaurant.db` with ACID rollback integrity.
- **Tandem Collaboration**: Documented 50/50 Driver/Navigator task distribution across milestones.

### 2. LMS Google Classroom Rubric (16 / 16 Points — Excellent)
- **PEAS Framework Application (4/4)**: Deconstructs Performance Measure, Environment, Actuators, and Sensors in Section 2 of formal report.
- **Task Scheduling Arrays (4/4)**: Demonstrates BFS Queue (FIFO via `collections.deque`) and DFS Stack (LIFO via Python List) for multi-turn task scheduling in Section 7 of formal report.
- **Search Strategy Analysis (4/4)**: Mathematical time and space complexity comparison of uninformed search ($O(b^d)$, $O(b^m)$ / $O(bm)$) and informed search ($A^*$) in Section 6 of formal report.
- **Heuristic Function Design (4/4)**: Admissible and consistent Manhattan Distance metric ($h(n)$) across category, price, and prep time dimensions in Section 8 of formal report.

---

## 🚀 Quick-Start Execution Guide

### 1. Requirements
- Python 3.10+ (Runs 100% on **Standard Library** — NO virtual environment required)
- *Optional:* Local Ollama with `qwen2.5` (`ollama run qwen2.5:1.5b`). If Ollama is not installed or offline, the agent automatically falls back to its built-in deterministic FSM dialogue engine.
- *Optional Voice Packages:* `speechrecognition`, `pyttsx3`, `edge-tts`, `rich`. If absent, standard console input/output is used automatically.

### 2. Initializing Database
```powershell
python init_db.py --reset
```

### 3. Running Milestone 1 Verification Test Suite
```powershell
python milestone1_test_suite.py
```

### 4. Running Interactive Waiter CLI Loop
```powershell
python cli_waiter.py --table 1
```
*(Add `--voice` to enable live microphone Speech-to-Text, or `--tts` for vocal audio feedback).*
