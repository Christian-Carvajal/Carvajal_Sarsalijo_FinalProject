"""
app.py
Main Application Entry Point for Perpetual Bistro AI Virtual Waiter
Course: BSCS 3112 - Artificial Intelligence (Lesson 7 Final Project)
University of Perpetual Help System DALTA

Chains all system architectural layers:
- UI Surface Layer: Tabletop ordering kiosk & Kitchen Display System (ui_dashboard/)
- Decoupled Speech I/O: STT & TTS voice processing (speech_engine.py)
- Core Agent Engine: FSM State Controller & Ollama/Qwen 2.5 (fsm_controller.py)
- Relational Persistence: Transactional SQLite database CRUD engine (db_tools.py)
"""

import os
import sys
import json
import re
from typing import Optional, Dict, Any
from pydantic import BaseModel

# Add current folder to path
_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import edge_tts

import db_tools
from fsm_controller import (
    BistroFSMAgent,
    STATE_GREETING,
    STATE_MENU_BROWSING,
    STATE_TAKING_ORDER,
    STATE_REVIEW,
    STATE_CONFIRMED,
    DEFAULT_MODEL,
)
from speech_engine import speech_engine

# Initialize database
db_tools.init_db()

app = FastAPI(title="Perpetual Bistro AI Virtual Waiter System (Lesson 7 Final Project)")

# Resolve UI Assets Path (prioritize ui_dashboard/ according to Lesson 7 file spec)
UI_DASHBOARD_DIR = os.path.join(_CURRENT_DIR, "ui_dashboard")
STATIC_DIR = os.path.join(UI_DASHBOARD_DIR, "static") if os.path.isdir(os.path.join(UI_DASHBOARD_DIR, "static")) else os.path.join(_CURRENT_DIR, "static")
TEMPLATES_DIR = os.path.join(UI_DASHBOARD_DIR, "templates") if os.path.isdir(os.path.join(UI_DASHBOARD_DIR, "templates")) else os.path.join(_CURRENT_DIR, "templates")
TEMPLATES_INDEX = os.path.join(TEMPLATES_DIR, "index.html")

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Enable CORS for local tablet kiosks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active Agent Instance (Default Table #1)
CURRENT_TABLE = 1
active_agent = BistroFSMAgent(table_no=CURRENT_TABLE, model=DEFAULT_MODEL)


# Pydantic Schemas
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    table_no: Optional[int] = 1


class AddItemRequest(BaseModel):
    session_id: Optional[str] = None
    item_name: str
    quantity: Optional[int] = None
    qty: Optional[int] = None
    instructions: Optional[str] = ""

    def get_effective_qty(self) -> int:
        if self.quantity is not None:
            return self.quantity
        if self.qty is not None:
            return self.qty
        return 1


class UpdateItemRequest(BaseModel):
    session_id: Optional[str] = None
    item_name: str
    new_quantity: Optional[int] = None
    qty: Optional[int] = None

    def get_effective_qty(self) -> int:
        if self.new_quantity is not None:
            return self.new_quantity
        if self.qty is not None:
            return self.qty
        return 1


class RemoveItemRequest(BaseModel):
    session_id: Optional[str] = None
    item_name: str
    qty: Optional[int] = None


class CheckoutRequest(BaseModel):
    session_id: Optional[str] = None


class SwitchTableRequest(BaseModel):
    table_no: int


class PaymentCheckoutRequest(BaseModel):
    session_id: Optional[str] = None
    table_no: Optional[int] = None
    payment_method: str = "CASH"
    payment_ref: Optional[str] = ""
    student_id: Optional[str] = ""


class KdsStatusRequest(BaseModel):
    ticket_id: str
    new_status: Optional[str] = None
    status: Optional[str] = None

    def get_effective_status(self) -> str:
        return (self.new_status or self.status or "READY").upper()


class RestockRequest(BaseModel):
    item_name: Optional[str] = None
    qty: Optional[int] = 10
    restock_all: Optional[bool] = False
    target_min: Optional[int] = 20


# REST Endpoints
@app.get("/api/menu")
def get_menu(category: Optional[str] = None):
    """Returns the full restaurant menu or filters by category."""
    items = db_tools.get_menu_items(category)
    return {
        "status": "success",
        "category_queried": category or "all",
        "item_count": len(items),
        "items": items,
    }


@app.get("/api/cart")
def get_current_cart():
    """Retrieves active table cart summary, items, and tax breakdown."""
    summary = db_tools.get_cart_summary(active_agent.session_id)
    return summary


@app.get("/api/cart/{session_id}")
def get_cart_by_session(session_id: str):
    """Retrieves the active order tab, state, subtotal, and tax breakdown."""
    summary = db_tools.get_cart_summary(session_id)
    return {"status": "success", "session_id": session_id, "summary": summary}


@app.post("/api/cart/add")
def add_to_cart(req: AddItemRequest):
    """Direct UI Human-in-the-Loop item addition."""
    sid = req.session_id or active_agent.session_id
    effective_qty = req.get_effective_qty()
    session = db_tools.create_or_get_session(table_no=CURRENT_TABLE, session_id=sid)
    current_state = session.get("status", STATE_GREETING)

    if current_state == STATE_CONFIRMED:
        raise HTTPException(status_code=400, detail="Cannot modify order. This tab has already been confirmed and paid.")

    result = db_tools.add_item_to_order(
        session_id=sid,
        item_name=req.item_name,
        qty=effective_qty,
        instructions=req.instructions or "",
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to add item"))

    if current_state in (STATE_GREETING, STATE_MENU_BROWSING):
        db_tools.update_session_state(sid, STATE_TAKING_ORDER)
        active_agent.state = STATE_TAKING_ORDER

    updated_cart = db_tools.get_cart_summary(sid)
    return {
        "status": "success",
        "result": result,
        "cart": updated_cart,
        "fsm_state": active_agent.state,
        "session_id": sid,
    }


@app.post("/api/cart/update")
def update_cart_item(req: UpdateItemRequest):
    """Human-in-the-Loop cart quantity adjustment."""
    sid = req.session_id or active_agent.session_id
    effective_qty = req.get_effective_qty()
    session = db_tools.create_or_get_session(table_no=CURRENT_TABLE, session_id=sid)
    if session.get("status") == STATE_CONFIRMED:
        raise HTTPException(status_code=400, detail="Cannot edit. Order is already finalized.")

    result = db_tools.update_item_quantity(sid, req.item_name, effective_qty)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Update failed"))

    updated_cart = db_tools.get_cart_summary(sid)
    return {
        "status": "success",
        "result": result,
        "cart": updated_cart,
        "fsm_state": active_agent.state,
    }


@app.post("/api/cart/remove")
def remove_cart_item(req: RemoveItemRequest):
    """Human-in-the-Loop cart item deletion."""
    sid = req.session_id or active_agent.session_id
    session = db_tools.create_or_get_session(table_no=CURRENT_TABLE, session_id=sid)
    if session.get("status") == STATE_CONFIRMED:
        raise HTTPException(status_code=400, detail="Cannot remove. Order is already finalized.")

    result = db_tools.remove_item_from_order(sid, req.item_name)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Deletion failed"))

    updated_cart = db_tools.get_cart_summary(sid)
    return {
        "status": "success",
        "result": result,
        "cart": updated_cart,
        "fsm_state": active_agent.state,
    }


@app.post("/api/table/switch")
def switch_table(req: SwitchTableRequest):
    """Switches active tabletop terminal to a different table number."""
    global CURRENT_TABLE, active_agent
    if req.table_no < 1 or req.table_no > 50:
        raise HTTPException(status_code=400, detail="Table number must be between 1 and 50.")
    CURRENT_TABLE = req.table_no
    active_agent = BistroFSMAgent(table_no=CURRENT_TABLE, model=DEFAULT_MODEL)
    cart = db_tools.get_cart_summary(active_agent.session_id)
    return {
        "status": "success",
        "table_no": CURRENT_TABLE,
        "session_id": active_agent.session_id,
        "fsm_state": active_agent.state,
        "cart": cart,
        "message": f"Switched terminal session to Table #{CURRENT_TABLE}.",
    }


@app.post("/api/checkout")
def checkout(req: CheckoutRequest):
    """Commits order tab, transitions to STATE_CONFIRMED, and updates inventory stock."""
    session = db_tools.create_or_get_session(table_no=CURRENT_TABLE, session_id=req.session_id)
    if session.get("status") == STATE_CONFIRMED:
        return {"status": "already_confirmed", "message": "This order has already been finalized."}

    summary = db_tools.get_cart_summary(req.session_id)
    if summary["item_count"] == 0:
        raise HTTPException(status_code=400, detail="Your order cart is empty. Please add items before checking out.")

    receipt = db_tools.confirm_order_and_deduct_stock(req.session_id)
    if not receipt.get("success"):
        raise HTTPException(status_code=400, detail=receipt.get("error", "Checkout transaction failed"))

    db_tools.update_session_state(req.session_id, STATE_CONFIRMED)
    return {"status": "success", "receipt": receipt}


@app.post("/api/payment/checkout")
def payment_checkout(req: PaymentCheckoutRequest):
    """Multi-channel payment processor (Cash, GCash, Maya, RFID) generating kitchen tickets."""
    sid = req.session_id or active_agent.session_id
    t_no = req.table_no or active_agent.table_no or CURRENT_TABLE
    session = db_tools.create_or_get_session(table_no=t_no, session_id=sid)
    summary = db_tools.get_cart_summary(sid)

    if summary["item_count"] == 0:
        raise HTTPException(status_code=400, detail="Your cart is empty. Please select food items first.")

    receipt = db_tools.confirm_order_and_deduct_stock(sid)
    if not receipt.get("success"):
        raise HTTPException(status_code=400, detail=receipt.get("error", "Stock deduction failed"))

    db_tools.update_session_state(sid, STATE_CONFIRMED)
    active_agent.state = STATE_CONFIRMED

    # Route ticket to Kitchen Display System (KDS)
    ticket = db_tools.create_kitchen_ticket(
        table_no=t_no,
        session_id=sid,
        items=summary["items"],
        subtotal=summary["subtotal"],
        payment_method=req.payment_method,
        payment_ref=req.payment_ref or f"AUTO-{req.payment_method}",
    )

    current_cart = db_tools.get_cart_summary(sid)

    return {
        "status": "success",
        "payment_method": req.payment_method,
        "payment_ref": req.payment_ref,
        "ticket": ticket,
        "receipt": receipt,
        "cart": current_cart,
        "fsm_state": STATE_CONFIRMED,
        "message": f"Payment via {req.payment_method} approved! Ticket #{ticket['ticket_id']} dispatched to kitchen.",
    }


@app.get("/api/kds/tickets")
def get_kds_tickets(status: Optional[str] = None):
    """Back-of-House Kitchen Display System (KDS) live ticket queue."""
    tickets = db_tools.get_kitchen_tickets(status=status)
    return {"status": "success", "tickets": tickets}


@app.post("/api/kds/status")
def update_kds_status(req: KdsStatusRequest):
    """Updates kitchen preparation lifecycle status (PREPARING -> READY -> SERVED)."""
    status_to_apply = req.get_effective_status()
    success = db_tools.update_ticket_status(req.ticket_id, status_to_apply)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to update ticket {req.ticket_id}")
    return {"status": "success", "ticket_id": req.ticket_id, "new_status": status_to_apply}


@app.get("/api/admin/analytics")
def get_admin_analytics():
    """Returns store performance analytics, ticket volumes, and inventory levels."""
    analytics = db_tools.get_analytics_summary()
    return analytics


@app.post("/api/admin/restock")
def restock_inventory(req: RestockRequest):
    """Restocks either a single menu item or all items below target threshold."""
    if req.restock_all:
        res = db_tools.restock_all_items(target_min_qty=req.target_min or 20)
        return res
    elif req.item_name:
        res = db_tools.restock_menu_item(item_name=req.item_name, qty_to_add=req.qty or 10)
        if res.get("status") == "error":
            raise HTTPException(status_code=404, detail=res.get("message"))
        return res
    else:
        raise HTTPException(status_code=400, detail="Must provide item_name or set restock_all=true")


class LanguageRequest(BaseModel):
    language: str


class TtsRequest(BaseModel):
    text: str
    language: Optional[str] = "filipino"


@app.post("/api/chat")
def chat_interaction(req: ChatRequest):
    """Processes natural language dialogue or voice input through FSM Controller & Qwen."""
    global active_agent
    if req.table_no and req.table_no != active_agent.table_no:
        active_agent = BistroFSMAgent(table_no=req.table_no, model=DEFAULT_MODEL)

    if req.session_id and req.session_id != active_agent.session_id:
        active_agent.session_id = req.session_id

    turn_data = active_agent.process_turn(req.message)
    resp_text = turn_data.get("assistant_response", "")
    speech_text = speech_engine.normalize_taglish_text(resp_text)
    current_cart = db_tools.get_cart_summary(active_agent.session_id)

    return {
        "response": resp_text,
        "assistant_response": resp_text,
        "speech_text": speech_text,
        "tool_calls": turn_data.get("tool_calls", []),
        "fsm_state": active_agent.state,
        "session_id": active_agent.session_id,
        "table_no": active_agent.table_no,
        "cart": current_cart,
        "model_used": active_agent.model,
    }


@app.post("/api/language")
def set_language_endpoint(req: LanguageRequest):
    """Updates active conversational persona language ('filipino' or 'english')."""
    active_agent.set_language(req.language)
    return {
        "status": "success",
        "language": active_agent.language,
        "message": f"Language persona switched to {active_agent.language.capitalize()}.",
    }


@app.get("/api/voice/listen")
def voice_listen():
    """Captures microphone audio via SpeechEngine and transcribes to text string without hanging."""
    transcription = speech_engine.listen_from_microphone(timeout=5, allow_console_input=False)
    return {"status": "success", "transcription": transcription}


@app.post("/api/tts")
async def text_to_speech_post(req: TtsRequest):
    """High-fidelity Taglish Text-to-Speech audio synthesizer (POST JSON)."""
    clean_text = speech_engine.normalize_taglish_text(req.text)
    voice = "en-PH-JamesNeural" if (req.language or "").lower() == "filipino" else "en-US-JennyNeural"
    try:
        communicate = edge_tts.Communicate(clean_text, voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return Response(content=audio_data, media_type="audio/mpeg")
    except Exception:
        return Response(content=b"\xff\xfb\x90\x00", media_type="audio/mpeg")


@app.get("/api/tts")
async def text_to_speech_get(text: str):
    """High-fidelity Taglish Text-to-Speech audio synthesizer (GET query)."""
    return await text_to_speech_post(TtsRequest(text=text))


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the complete single-page interactive tablet dashboard."""
    if os.path.exists(TEMPLATES_INDEX):
        with open(TEMPLATES_INDEX, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=f"<h1>Perpetual Bistro Kiosk</h1><p>Template not found at {TEMPLATES_INDEX}</p>")


def find_available_port(preferred_port=8000, fallback_ports=(5000, 8001, 8080, 8888)):
    import socket
    for port in [preferred_port] + list(fallback_ports):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred_port


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Perpetual Bistro AI Virtual Waiter Application")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the dashboard on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    args = parser.parse_args()

    port = args.port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((args.host, port))
        except OSError:
            old_port = port
            port = find_available_port(preferred_port=8000)
            print(f"[Notice] Port {old_port} was occupied. Automatically selected available port {port}.")

    print(f"\n==================================================================")
    print(f"[Perpetual Bistro] Starting AI Virtual Waiter System (Lesson 7)")
    print(f"[Perpetual Bistro] Local Tablet URL: http://localhost:{port} or http://127.0.0.1:{port}")
    print(f"[Perpetual Bistro] Decoupled Speech I/O: Active (SpeechEngine)")
    print(f"[Perpetual Bistro] Core FSM Controller:  Active (fsm_controller.py)")
    print(f"[Perpetual Bistro] SQLite Tool Engine:   Active (restaurant.db)")
    print(f"==================================================================\n")
    import threading
    import webbrowser
    def _open_kiosk(url):
        import time
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=_open_kiosk, args=(f"http://127.0.0.1:{port}",), daemon=True).start()

    uvicorn.run("app:app", host=args.host, port=port, reload=True)
