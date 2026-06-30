"""Local, dependency-free backend for the ReelPeel conference demo."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


HOST = "127.0.0.1"
PORT = 8765
FALLBACK_PORTS = range(PORT, PORT + 11)
ROOT = Path(__file__).resolve().parent
MOCK_ROOT = ROOT.parent / "offline_mock"
REEL_PATH = re.compile(r"/reels?/([^/?#]+)", re.IGNORECASE)
PUBMED_PATH = re.compile(r"/pubmed/(\d+)/?$")
PROCESS_DELAY_SECONDS = 3
SUMMARY_DELAY_SECONDS = 2
DEMO_PATH = "/reels/DT0UIgzDZ79/"


def demo_url(port):
    return f"http://{HOST}:{port}{DEMO_PATH}"


def open_demo_in_browser(url):
    """Open the demo in Chrome when possible, with a default-browser fallback."""
    chrome_commands = [
        "google-chrome",
        "google-chrome-stable",
        "chrome",
        "chromium",
        "chromium-browser",
        "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
        "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    ]

    for command in chrome_commands:
        executable = command if Path(command).is_file() else shutil.which(command)
        if not executable:
            continue
        try:
            subprocess.Popen(
                [executable, url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except OSError:
            continue

    try:
        subprocess.Popen(
            ["cmd.exe", "/c", "start", "", "chrome", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except OSError:
        return webbrowser.open_new(url)


import random

ACTIVE_RUNS = {}

def load_offline_mocks():
    """Load the recorded pipeline responses and evidence summaries by Reel ID."""
    process_responses = {}
    summaries = {}

    runs_found = False
    # Collect all sweep runs
    for sweep_dir in MOCK_ROOT.glob("sweep_*"):
        if not sweep_dir.is_dir(): continue
        for run_dir in sweep_dir.iterdir():
            if not run_dir.is_dir(): continue
            manifest_path = run_dir / "manifest.json"
            if manifest_path.is_file():
                runs_found = True
                run_id = run_dir.name
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                parameters = manifest.get("parameters", {})
                
                for video in manifest.get("videos", []):
                    reel_id = video.get("reel_id")
                    if not reel_id: continue
                    
                    reel_dir_in_run = run_dir / reel_id
                    if not reel_dir_in_run.is_dir(): continue
                    
                    process_path = reel_dir_in_run / Path(video.get("process_response", "process.json")).name
                    if process_path.is_file():
                        process_data = json.loads(process_path.read_text(encoding="utf-8"))
                        process_data["_offline_settings"] = parameters
                        process_responses.setdefault(reel_id, []).append({
                            "run_id": run_id,
                            "process_data": process_data
                        })
                        
                    for entry in video.get("evidence_summaries", []):
                        response_path = reel_dir_in_run / entry["response_file"]
                        pubmed_id = str(entry.get("pubmed_id") or "")
                        if pubmed_id and response_path.is_file():
                            summaries[(reel_id, run_id, pubmed_id)] = json.loads(
                                response_path.read_text(encoding="utf-8")
                            )

    if not runs_found:
        # Fallback to old flat directory logic
        for reel_dir in MOCK_ROOT.iterdir():
            if not reel_dir.is_dir():
                continue

            manifest_path = reel_dir / "manifest.json"
            if not manifest_path.is_file():
                continue

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            process_path = reel_dir / Path(manifest["process_response"]).name
            process_data = json.loads(process_path.read_text(encoding="utf-8"))
            process_responses.setdefault(reel_dir.name, []).append({
                "run_id": "default",
                "process_data": process_data
            })

            for entry in manifest.get("evidence_summaries", []):
                response_path = reel_dir / entry["response_file"]
                pubmed_id = str(entry.get("pubmed_id") or "")
                if pubmed_id and response_path.is_file():
                    summaries[(reel_dir.name, "default", pubmed_id)] = json.loads(
                        response_path.read_text(encoding="utf-8")
                    )

    return process_responses, summaries


def parse_preferred_port():
    raw = os.environ.get("REELPEEL_PORT", "").strip()
    if not raw:
        return PORT
    try:
        port = int(raw)
    except ValueError:
        print(f"Ignoring invalid REELPEEL_PORT={raw!r}; using {PORT}.", flush=True)
        return PORT
    if 1 <= port <= 65535:
        return port
    print(f"Ignoring out-of-range REELPEEL_PORT={raw!r}; using {PORT}.", flush=True)
    return PORT


def server_ports(preferred_port):
    ports = [preferred_port]
    ports.extend(port for port in FALLBACK_PORTS if port != preferred_port)
    return ports


def create_server():
    preferred_port = parse_preferred_port()
    errors = []
    for port in server_ports(preferred_port):
        try:
            return ThreadingHTTPServer((HOST, port), DemoHandler), port, preferred_port
        except OSError as error:
            errors.append(f"{port}: {error}")

    details = "\n".join(errors)
    raise SystemExit(
        "Could not bind the offline demo server to any local port "
        f"in {server_ports(preferred_port)}.\n{details}"
    )


PROCESS_RESPONSES, EVIDENCE_SUMMARIES = load_offline_mocks()


class DemoHandler(SimpleHTTPRequestHandler):
    """Serves the local Reels UI and its two JSON endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok", "offline": True})
            return
        if path == "/" or REEL_PATH.fullmatch(path.rstrip("/")):
            self.path = "/index.html"
        else:
            pubmed_match = PUBMED_PATH.fullmatch(path)
            if pubmed_match:
                self.path = f"/pubmed/{pubmed_match.group(1)}/index.html"
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        payload = self._read_json()
        if payload is None:
            return

        if path in {"/json", "/api/analyze"}:
            reel_id = self._reel_id(payload)
            runs = PROCESS_RESPONSES.get(reel_id)
            if not runs:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"detail": f"No offline process fixture exists for reel '{reel_id or 'unknown'}'."},
                )
                return
            
            requested_run_id = payload.get("run_id")
            selected_run = None
            if requested_run_id:
                for r in runs:
                    if r["run_id"] == requested_run_id:
                        selected_run = r
                        break
            
            if not selected_run:
                selected_run = random.choice(runs)
            
            ACTIVE_RUNS[reel_id] = selected_run["run_id"]
            response = dict(selected_run["process_data"])
            
            available_runs = [
                {"run_id": r["run_id"], "parameters": r["process_data"].get("_offline_settings", {})}
                for r in runs
            ]
            response["_available_runs"] = available_runs
            response["_selected_run_id"] = selected_run["run_id"]
            
            time.sleep(PROCESS_DELAY_SECONDS)
            self._send_json(HTTPStatus.OK, response)
            return

        if path == "/evidence_summary":
            reel_id = self._reel_id({"url": payload.get("reel_url")})
            evidence = payload.get("evidence") or {}
            pubmed_id = str(evidence.get("pubmed_id") or "")
            
            run_id = ACTIVE_RUNS.get(reel_id, "default")
            response = EVIDENCE_SUMMARIES.get((reel_id, run_id, pubmed_id))
            
            if response is None:
                # Fallback to any run if not found
                for (r_id, r_run_id, p_id), summary in EVIDENCE_SUMMARIES.items():
                    if r_id == reel_id and p_id == pubmed_id:
                        response = summary
                        break
            
            if response is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"detail": "No offline evidence summary exists for this Reel and source."},
                )
                return
            time.sleep(SUMMARY_DELAY_SECONDS)
            self._send_json(HTTPStatus.OK, response)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Unknown local endpoint."})

    def _read_json(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            return json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": "Request body must be valid JSON."})
            return None

    @staticmethod
    def _reel_id(payload):
        explicit_id = str(payload.get("reel_id") or "").strip()
        if explicit_id:
            return explicit_id
        match = REEL_PATH.search(str(payload.get("url") or ""))
        return match.group(1) if match else ""

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server, port, preferred_port = create_server()
    url = demo_url(port)
    if port != preferred_port:
        print(f"Port {preferred_port} is unavailable; using {port}.", flush=True)
    print(f"Offline ReelPeel demo: {url}", flush=True)
    if not open_demo_in_browser(url):
        print("Could not open a browser automatically.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.", flush=True)
    finally:
        server.server_close()
