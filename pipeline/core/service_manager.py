import socket
import subprocess
import time
import sys
import os


def ensure_pubmed_proxy(port=8080):
    """Checks if the proxy is running; if not, starts it as a background process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        is_running = s.connect_ex(('127.0.0.1', port)) == 0

    if not is_running:
        print(f"--- Starting Shared PubMed Proxy Service on port {port} ---")

        # 1. Determine the absolute path to the proxy script
        # We assume this file is in: pipeline/core/service_manager.py
        # We want to find:          services/pubmed_proxy.py

        current_dir = os.path.dirname(os.path.abspath(__file__))  # .../pipeline/core
        pipeline_dir = os.path.dirname(current_dir)  # .../pipeline
        project_root = os.path.dirname(pipeline_dir)  # .../ (Root)

        script_path = os.path.join(project_root, "services", "pubmed_proxy.py")

        # 2. Verify the file exists before trying to run it
        if not os.path.exists(script_path):
            print(f"[ERROR] Service Manager could not find: {script_path}")
            print(f"        Please check that 'pubmed_proxy.py' is in the 'services' folder at the project root.")
            return

        # 3. Setup logging
        log_path = os.path.join(project_root, "pubmed_proxy.log")
        log_file = open(log_path, "a")

        # 4. Launch
        subprocess.Popen(
            [sys.executable, script_path],
            stdout=log_file,
            stderr=log_file,
            start_new_session=True  # Detach the process
        )

        # Wait a moment for it to boot
        time.sleep(2)
    else:
        # Service is already running
        pass