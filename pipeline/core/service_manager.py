import socket
import subprocess
import time
import sys
import os
import urllib.request
import urllib.error

# Must match the path in pubmed_proxy.py
SHARED_DIR = "/data/home/jak38842/disk/fact_checker/pubmed_db_cache"

def wait_for_health(port, timeout=15):
    """Polls the health endpoint until it responds 200 OK."""
    start_time = time.time()
    health_url = f"http://127.0.0.1:{port}/health"

    print(f"     Waiting for service to be healthy at {health_url}...")

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    print("     Service is healthy!")
                    return True
        except (urllib.error.URLError, ConnectionResetError):
            pass
        except Exception as e:
            # Ignore other startup glitches
            pass

        time.sleep(0.5)

    return False


def check_process_alive(proc):
    """Returns False if process has exited."""
    if proc.poll() is not None:
        return False
    return True


def ensure_pubmed_proxy(port=8080):
    """Checks if the proxy is running; if not, starts it and waits for health."""

    # 1. Quick Socket Check (Is anything listening?)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        is_port_open = s.connect_ex(('127.0.0.1', port)) == 0

    if is_port_open:
        # It's open, but is it our proxy? Quick health check to be sure.
        if wait_for_health(port, timeout=2):
            return
        print("     [WARNING] Port is open but service is not responding correctly.")

    print(f"--- Starting Shared PubMed Proxy Service on port {port} ---")

    # 2. Locate script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(pipeline_dir)
    script_path = os.path.join(project_root, "services", "pubmed_proxy.py")

    if not os.path.exists(script_path):
        print(f"[ERROR] Cannot find proxy script at: {script_path}")
        return

    # 3. Prepare Shared Directory
    if not os.path.exists(SHARED_DIR):
        try:
            os.makedirs(SHARED_DIR, mode=0o777, exist_ok=True)
            os.chmod(SHARED_DIR, 0o777)
        except Exception:
            pass

    log_path = os.path.join(SHARED_DIR, "pubmed_proxy.log")
    log_file = open(log_path, "a")

    # 4. Launch Process
    proc = subprocess.Popen(
        [sys.executable, script_path],
        stdout=log_file,
        stderr=log_file,
        start_new_session=True
    )

    # 5. Robust Wait
    if wait_for_health(port, timeout=15):
        return

    # 6. Handle Failure
    print(f"[ERROR] Service failed to start within 15 seconds.")

    # Check if process died immediately
    if not check_process_alive(proc):
        print("        Process exited unexpectedly. Checking last 5 log lines:")
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()[-5:]
                for line in lines:
                    print(f"        LOG: {line.strip()}")
        except Exception:
            print("        (Could not read log file)")
    else:
        print("        Process is running but not responsive. Killing it.")
        proc.kill()

    sys.exit(1)