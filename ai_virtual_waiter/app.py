"""
app.py
Main Application Entry Point for Perpetual Bistro AI Virtual Waiter (Milestone 2)
Course: BSCS 3112 - Artificial Intelligence (Lesson 7 Final Project)
University of Perpetual Help System DALTA — Las Piñas Campus

Chains all system architectural layers:
- UI Surface Layer: Tabletop ordering kiosk & Kitchen Display System (ui_dashboard/)
- Decoupled Speech I/O: STT & TTS voice processing (speech_engine.py)
- Core Agent Engine: FSM State Controller & Ollama/Qwen 2.5 (fsm_controller.py)
- Relational Persistence: Transactional SQLite database CRUD engine (db_tools.py)

Dual-Engine Architecture:
- Primary: FastAPI + Uvicorn (if installed)
- Resilient Zero-Dependency Fallback: Python Standard Library ThreadingHTTPServer (no venv required)
- Auto-Browser Launch: Automatically opens default web browser at http://localhost:8000 on startup
"""

import os
import sys
import json
import re
import mimetypes
import threading
import webbrowser
from typing import Optional, Dict, Any

# Ensure current folder is on sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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

# Active Agent Instance (Default Table #1)
CURRENT_TABLE = 1
active_agent = BistroFSMAgent(table_no=CURRENT_TABLE, model=DEFAULT_MODEL)

# Resolve UI Assets Path (prioritize ui_dashboard/ according to Lesson 7 file spec)
UI_DASHBOARD_DIR = os.path.join(BASE_DIR, "ui_dashboard")
STATIC_DIR = os.path.join(UI_DASHBOARD_DIR, "static") if os.path.isdir(os.path.join(UI_DASHBOARD_DIR, "static")) else os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(UI_DASHBOARD_DIR, "templates") if os.path.isdir(os.path.join(UI_DASHBOARD_DIR, "templates")) else os.path.join(BASE_DIR, "templates")
TEMPLATES_INDEX = os.path.join(TEMPLATES_DIR, "index.html")

os.makedirs(STATIC_DIR, exist_ok=True)


def find_available_port(preferred_port=8000, fallback_ports=(8001, 8080, 8888, 5000)):
    import socket
    for port in [preferred_port] + list(fallback_ports):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    return preferred_port


def sanitize_for_speech(t: str, language: str = "filipino") -> str:
    """Pre-cleans text for speech synthesis: tunes phonetics, strips emojis/markdown, and formats pesos."""
    if not t:
        return ""
    s = re.sub(r'₱\s*(\d+(?:\.\d+)?)(?:\s*pesos)?', r'\1 pesos', t, flags=re.IGNORECASE)
    s = re.sub(r'₱', ' pesos', s)
    s = re.sub(r'[*_#`~]', '', s)
    s = re.sub(r'[^\w\s.,!?-]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# =====================================================================
# FASTAPI ENGINE (Activates if FastAPI & Uvicorn are installed)
# =====================================================================
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

if HAS_FASTAPI:
    app = FastAPI(title="Perpetual Bistro AI Virtual Waiter System (Lesson 7 Final Project)")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ChatRequest(BaseModel):
        message: str
        table_no: Optional[int] = None

    class CartAddRequest(BaseModel):
        item_name: str
        qty: Optional[int] = 1
        instructions: Optional[str] = ""

    class CartUpdateRequest(BaseModel):
        item_name: str
        qty: int

    class SessionResetRequest(BaseModel):
        table_no: Optional[int] = 1

    class TableSwitchRequest(BaseModel):
        table_no: int

    class LanguageRequest(BaseModel):
        language: str

    class TTSRequest(BaseModel):
        text: str
        voice: Optional[str] = None
        language: Optional[str] = "filipino"

    class KdsStatusRequest(BaseModel):
        ticket_id: str
        status: str

    class PaymentCheckoutRequest(BaseModel):
        payment_method: str = "CASH"
        payment_ref: Optional[str] = ""
        student_id: Optional[str] = ""
        table_no: Optional[int] = None

    class RestockRequest(BaseModel):
        item_name: Optional[str] = None
        qty: Optional[int] = 10
        restock_all: Optional[bool] = False
        target_min: Optional[int] = 20

    @app.get("/api/session")
    def get_session_info():
        cart = db_tools.view_current_order(active_agent.session_id)
        return {
            "session_id": active_agent.session_id,
            "table_no": active_agent.table_no,
            "fsm_state": res.get("fsm_state", "STATE_CONFIRMED"),
            "language": getattr(active_agent, "language", "filipino"),
            "model": active_agent.model,
            "cart": cart,
        }

    @app.get("/api/menu")
    def get_menu():
        return db_tools.get_menu_by_category("all")

    @app.get("/api/cart")
    def get_cart():
        return db_tools.view_current_order(active_agent.session_id)

    @app.post("/api/language")
    def set_language_endpoint(req: LanguageRequest):
        global active_agent
        lang = req.language.lower().strip()
        if lang not in ["filipino", "english"]:
            raise HTTPException(status_code=400, detail="Language must be 'filipino' or 'english'.")
        active_agent.set_language(lang)
        return {"status": "success", "language": lang, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED")}

    @app.post("/api/chat")
    def process_chat(req: ChatRequest):
        global active_agent
        if req.table_no and req.table_no != active_agent.table_no:
            active_agent.set_table_number(req.table_no)
        turn_data = active_agent.process_turn(req.message)
        cart = db_tools.view_current_order(active_agent.session_id)
        return {
            "assistant_response": turn_data.get("assistant_response", ""),
            "response": turn_data.get("assistant_response", ""),
            "speech_text": turn_data.get("assistant_response", ""),
            "fsm_state": res.get("fsm_state", "STATE_CONFIRMED"),
            "tool_calls": turn_data.get("tool_calls", []),
            "engine": turn_data.get("engine", "ollama"),
            "cart": cart,
            "session_id": active_agent.session_id,
            "table_no": active_agent.table_no,
        }

    @app.post("/api/cart/add")
    def add_cart_item(req: CartAddRequest):
        global active_agent
        res = active_agent.execute_tool("add_item_to_order", {
            "item_name": req.item_name,
            "qty": req.qty or 1,
            "instructions": req.instructions or "",
        })
        cart = db_tools.view_current_order(active_agent.session_id)
        return {"result": res, "cart": cart, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED")}

    @app.post("/api/cart/update")
    def update_cart_item(req: CartUpdateRequest):
        res = db_tools.update_item_quantity(active_agent.session_id, req.item_name, req.qty)
        cart = db_tools.view_current_order(active_agent.session_id)
        return {"result": res, "cart": cart, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED")}

    @app.post("/api/cart/remove")
    def remove_cart_item(req: CartUpdateRequest):
        res = db_tools.remove_item_from_order(active_agent.session_id, req.item_name)
        cart = db_tools.view_current_order(active_agent.session_id)
        return {"result": res, "cart": cart, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED")}

    @app.post("/api/table/switch")
    def switch_table_endpoint(req: TableSwitchRequest):
        global active_agent
        if req.table_no != active_agent.table_no:
            active_agent.set_table_number(req.table_no)
        cart = db_tools.view_current_order(active_agent.session_id)
        return {"status": "success", "table_no": active_agent.table_no, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED"), "cart": cart}

    @app.post("/api/confirm")
    def confirm_order():
        global active_agent
        db_tools.create_or_get_session(active_agent.table_no, active_agent.session_id)
        res = active_agent.execute_tool("confirm_order_and_deduct_stock", {})
        cart = db_tools.view_current_order(active_agent.session_id)
        return {"result": res, "cart": cart, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED")}

    @app.get("/api/kds/tickets")
    def get_kds_tickets(status: Optional[str] = None):
        return {"status": "success", "tickets": db_tools.get_kitchen_tickets(status_filter=status)}

    @app.post("/api/kds/status")
    def update_kds_ticket_status(req: KdsStatusRequest):
        ok = db_tools.update_kitchen_ticket_status(req.ticket_id, req.status)
        if not ok:
            raise HTTPException(status_code=400, detail="Invalid ticket status.")
        return {"status": "success", "ticket_id": req.ticket_id, "new_status": req.status}

    @app.post("/api/payment/checkout")
    def process_payment_and_checkout(req: PaymentCheckoutRequest):
        global active_agent
        if req.table_no and req.table_no != active_agent.table_no:
            active_agent.set_table_number(req.table_no)
        res = active_agent.execute_tool("confirm_order_and_deduct_stock", {})
        if res.get("status") == "success":
            res["payment_method"] = req.payment_method
            res["payment_ref"] = req.payment_ref or ""
            db_tools.create_kitchen_ticket(
                table_no=active_agent.table_no,
                session_id=active_agent.session_id,
                items=res.get("itemized_receipt") or res.get("items", []),
                grand_total=res.get("grand_total") or 0.0,
                payment_method=req.payment_method,
                payment_ref=req.payment_ref or "",
                ticket_id=res.get("receipt_id"),
            )
            # Fresh session for the next order
            active_agent.session = db_tools.create_or_get_session(active_agent.table_no, session_id=None)
            active_agent.session_id = active_agent.session["session_id"]
            active_agent.state = "STATE_GREETING"
            active_agent._init_system_prompt()
        cart = db_tools.view_current_order(active_agent.session_id)
        return {
            "status": res.get("status", "error"),
            "success": res.get("status") == "success",
            "receipt": res,
            "result": res,
            "message": res.get("message", "Order confirmed successfully!"),
            "cart": cart,
            "fsm_state": res.get("fsm_state", "STATE_CONFIRMED"),
        }

    @app.get("/api/admin/analytics")
    def get_admin_analytics():
        return db_tools.get_analytics_summary()

    @app.post("/api/admin/restock")
    def restock_inventory(req: RestockRequest):
        if req.restock_all:
            res = db_tools.restock_all_items(target_min_qty=req.target_min or 20)
        else:
            res = db_tools.restock_menu_item(req.item_name or "Perpetual Classic Burger", qty_to_add=req.qty or 10)
        return {"result": res, "message": res.get("message", "Stock updated successfully.")}

    @app.post("/api/reset")
    def reset_session(req: SessionResetRequest):
        global active_agent
        table_num = req.table_no or 1
        current_lang = getattr(active_agent, "language", "filipino")
        active_agent = BistroFSMAgent(table_no=table_num, model=DEFAULT_MODEL, language=current_lang)
        return {"status": "success", "session_id": active_agent.session_id, "table_no": active_agent.table_no, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED")}

    @app.post("/api/tts")
    async def natural_tts_endpoint(req: TTSRequest):
        try:
            import edge_tts
            lang = (req.language or getattr(active_agent, "language", "filipino")).lower().strip()
            clean_text = sanitize_for_speech(req.text, language=lang)
            if not clean_text:
                return Response(content=b"", media_type="audio/mpeg")
            voice = req.voice or ("en-US-JennyNeural" if lang == "english" else "en-PH-RosaNeural")
            communicate = edge_tts.Communicate(clean_text, voice)
            chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return Response(content=b"".join(chunks), media_type="audio/mpeg")
        except Exception:
            return Response(content=b"", media_type="audio/mpeg")

    @app.get("/", response_class=HTMLResponse)
    def index_page():
        if os.path.exists(TEMPLATES_INDEX):
            with open(TEMPLATES_INDEX, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content=f"<h1>Perpetual Bistro Kiosk</h1><p>Template not found at {TEMPLATES_INDEX}</p>")


# =====================================================================
# ZERO-DEPENDENCY STANDARD LIBRARY HTTP SERVER
# (Runs if FastAPI/Uvicorn are absent — 100% Python Standard Library)
# =====================================================================
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.parse


class NativeKioskHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        global active_agent
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            if os.path.exists(TEMPLATES_INDEX):
                with open(TEMPLATES_INDEX, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        elif path.startswith("/static/"):
            rel_file = path[len("/static/"):].lstrip("/")
            file_path = os.path.join(STATIC_DIR, rel_file)
            if os.path.exists(file_path):
                ctype, _ = mimetypes.guess_type(file_path)
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", ctype or "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        elif path == "/api/session":
            cart = db_tools.view_current_order(active_agent.session_id)
            self._send_json({
                "session_id": active_agent.session_id,
                "table_no": active_agent.table_no,
                "fsm_state": res.get("fsm_state", "STATE_CONFIRMED"),
                "language": getattr(active_agent, "language", "filipino"),
                "model": active_agent.model,
                "cart": cart,
            })
            return

        elif path == "/api/menu":
            self._send_json(db_tools.get_menu_by_category("all"))
            return

        elif path == "/api/cart":
            self._send_json(db_tools.view_current_order(active_agent.session_id))
            return

        elif path == "/api/kds/tickets":
            query = urllib.parse.parse_qs(parsed.query)
            status_filter = query.get("status", [None])[0]
            self._send_json({"status": "success", "tickets": db_tools.get_kitchen_tickets(status_filter=status_filter)})
            return

        elif path == "/api/admin/analytics":
            self._send_json(db_tools.get_analytics_summary())
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        global active_agent
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        if path == "/api/chat":
            msg = body.get("message", "")
            t_no = body.get("table_no")
            if t_no and t_no != active_agent.table_no:
                active_agent.set_table_number(t_no)
            turn_data = active_agent.process_turn(msg)
            cart = db_tools.view_current_order(active_agent.session_id)
            self._send_json({
                "assistant_response": turn_data.get("assistant_response", ""),
                "response": turn_data.get("assistant_response", ""),
                "speech_text": turn_data.get("assistant_response", ""),
                "fsm_state": res.get("fsm_state", "STATE_CONFIRMED"),
                "tool_calls": turn_data.get("tool_calls", []),
                "engine": turn_data.get("engine", "ollama"),
                "cart": cart,
                "session_id": active_agent.session_id,
                "table_no": active_agent.table_no,
            })
            return

        elif path == "/api/cart/add":
            res = active_agent.execute_tool("add_item_to_order", {
                "item_name": body.get("item_name", ""),
                "qty": body.get("qty", 1),
                "instructions": body.get("instructions", ""),
            })
            cart = db_tools.view_current_order(active_agent.session_id)
            self._send_json({"result": res, "cart": cart, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED")})
            return

        elif path == "/api/cart/update":
            res = db_tools.update_item_quantity(active_agent.session_id, body.get("item_name", ""), body.get("qty", 1))
            cart = db_tools.view_current_order(active_agent.session_id)
            self._send_json({"result": res, "cart": cart, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED")})
            return

        elif path == "/api/cart/remove":
            res = db_tools.remove_item_from_order(active_agent.session_id, body.get("item_name", ""))
            cart = db_tools.view_current_order(active_agent.session_id)
            self._send_json({"result": res, "cart": cart, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED")})
            return

        elif path == "/api/confirm":
            db_tools.create_or_get_session(active_agent.table_no, active_agent.session_id)
            res = active_agent.execute_tool("confirm_order_and_deduct_stock", {})
            cart = db_tools.view_current_order(active_agent.session_id)
            self._send_json({"result": res, "cart": cart, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED")})
            return

        elif path == "/api/payment/checkout":
            table_no = body.get("table_no", active_agent.table_no)
            if table_no != active_agent.table_no:
                active_agent.set_table_number(table_no)
            res = active_agent.execute_tool("confirm_order_and_deduct_stock", {})
            p_method = body.get("payment_method", "CASH")
            p_ref = body.get("payment_ref", "")
            if res.get("status") == "success":
                res["payment_method"] = p_method
                res["payment_ref"] = p_ref
                db_tools.create_kitchen_ticket(
                    table_no=active_agent.table_no,
                    session_id=active_agent.session_id,
                    items=res.get("itemized_receipt") or res.get("items", []),
                    grand_total=res.get("grand_total") or 0.0,
                    payment_method=p_method,
                    payment_ref=p_ref,
                    ticket_id=res.get("receipt_id"),
                )
                # Fresh session for the next order
                active_agent.session = db_tools.create_or_get_session(active_agent.table_no, session_id=None)
                active_agent.session_id = active_agent.session["session_id"]
                active_agent.state = "STATE_GREETING"
                active_agent._init_system_prompt()
            cart = db_tools.view_current_order(active_agent.session_id)
            self._send_json({
                "status": res.get("status", "error"),
                "success": res.get("status") == "success",
                "receipt": res,
                "result": res,
                "message": res.get("message", "Order confirmed successfully!"),
                "cart": cart,
                "fsm_state": res.get("fsm_state", "STATE_CONFIRMED"),
            })
            return

        elif path == "/api/table/switch":
            t_num = body.get("table_no", 1)
            active_agent.set_table_number(t_num)
            cart = db_tools.view_current_order(active_agent.session_id)
            self._send_json({"status": "success", "table_no": active_agent.table_no, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED"), "cart": cart})
            return

        elif path == "/api/language":
            lang = body.get("language", "filipino").lower().strip()
            active_agent.set_language(lang)
            self._send_json({"status": "success", "language": lang, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED")})
            return

        elif path == "/api/kds/status":
            ok = db_tools.update_kitchen_ticket_status(body.get("ticket_id", ""), body.get("status", "PREPARING"))
            self._send_json({"status": "success" if ok else "error"})
            return

        elif path == "/api/admin/restock":
            if body.get("restock_all"):
                res = db_tools.restock_all_items(target_min_qty=body.get("target_min", 20))
            else:
                res = db_tools.restock_menu_item(body.get("item_name", "Perpetual Classic Burger"), qty_to_add=body.get("qty", 10))
            self._send_json({"result": res, "message": res.get("message", "Stock updated successfully.")})
            return

        elif path == "/api/reset":
            t_num = body.get("table_no", 1)
            active_agent = BistroFSMAgent(table_no=t_num, model=DEFAULT_MODEL, language=getattr(active_agent, "language", "filipino"))
            self._send_json({"status": "success", "session_id": active_agent.session_id, "table_no": active_agent.table_no, "fsm_state": res.get("fsm_state", "STATE_CONFIRMED")})
            return

        elif path == "/api/tts":
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        self.send_response(404)
        self.end_headers()


def start_server(port: int = 8000, host: str = "127.0.0.1", force_native: bool = False):
    available_port = find_available_port(preferred_port=port)
    if available_port != port:
        print(f"[Notice] Port {port} was in use. Automatically switched to port {available_port}.")
    port = available_port

    url = f"http://localhost:{port}"

    def _auto_open():
        import time
        time.sleep(1.2)
        try:
            webbrowser.open(url)
            print(f"[Browser] Auto-launched {url} in your default browser.")
        except Exception:
            pass

    threading.Thread(target=_auto_open, daemon=True).start()

    print(f"\n==================================================================")
    print(f" [PERPETUAL BISTRO] AI Virtual Waiter Tablet Ordering Dashboard")
    print(f" Lesson 7 Final Project — Milestone 2 Full System Integration")
    print(f"==================================================================")
    print(f" • Server URL:         {url} (or http://{host}:{port})")
    print(f" • Engine Mode:        {'FastAPI/Uvicorn' if (HAS_FASTAPI and not force_native) else 'Python Standard Library ThreadingHTTPServer (Zero-Dependency)'}")
    print(f" • FSM State Machine:  Active (fsm_controller.py)")
    print(f" • SQLite Database:    Active (db_tools.py -> restaurant.db)")
    print(f" • Speech Engine:      Active (speech_engine.py)")
    print(f" • UI Dashboard:       Active (ui_dashboard/templates/index.html)")
    print(f" • Browser Launcher:   Active (Auto-opening {url}...)")
    print(f"==================================================================")
    print(f" Press Ctrl+C in this terminal to stop the server.\n")

    if HAS_FASTAPI and not force_native:
        uvicorn.run("app:app", host=host, port=port, reload=False)
    else:
        server = ThreadingHTTPServer((host, port), NativeKioskHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[Perpetual Bistro] Server stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Perpetual Bistro AI Virtual Waiter Application (Milestone 2)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the dashboard on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--native", action="store_true", help="Force Python standard library server without FastAPI")
    args = parser.parse_args()

    start_server(port=args.port, host=args.host, force_native=args.native)
