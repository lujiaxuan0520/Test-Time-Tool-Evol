# validate_in_docker.py
import subprocess
import json
import os
import tempfile
import sys
import re

def check_docker():
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("[ERROR] Docker is unavailable:", result.stderr)
            return False
        print("[INFO] Docker is available")
        return True
    except FileNotFoundError:
        print("[ERROR] Docker command not found. Please install Docker Desktop and start it.")
        return False
    except subprocess.TimeoutExpired:
        print("[ERROR] Docker detection timed out")
        return False

def run_validate_in_docker(data, docker_image="python:3.11-slim", timeout=300, max_install_retry=5):
    """
    Validate Python code snippets in an isolated Docker container.
    Automatically installs missing packages (up to max_install_retry times).
    Returns a list of validation results.
    """
    if not check_docker():
        return None

    # Pull image if not present
    try:
        subprocess.run(["docker", "inspect", docker_image], check=True, capture_output=True)
        print(f"[INFO] Image {docker_image} already exists")
    except subprocess.CalledProcessError:
        print(f"[INFO] Image {docker_image} not found, pulling...")
        pull_proc = subprocess.run(["docker", "pull", docker_image], capture_output=True, text=True)
        if pull_proc.returncode != 0:
            print("[ERROR] Failed to pull Docker image:", pull_proc.stderr)
            return None

    results = []

    for idx, code_str in enumerate(data.get("python_tool", [])):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "validate_script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(f"""
import json
import sys, subprocess, ast, tempfile, re

code_str = {json.dumps(code_str)}
timeout = {timeout}
max_install_retry = {max_install_retry}

result = {{"index": {idx}, "success": False, "stdout": "", "stderr": "", "installed": []}}

def run_code():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as tmp_file:
        tmp_file.write(code_str)
        tmp_path = tmp_file.name
    try:
        proc = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=timeout)
        return proc
    finally:
        import os
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Syntax check
try:
    ast.parse(code_str)
except SyntaxError as e:
    result["stderr"] = f"SyntaxError: {{e}}"
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0)

# Auto-install missing modules
for _ in range(max_install_retry):
    proc = run_code()
    result["stdout"] = proc.stdout
    result["stderr"] = proc.stderr
    if proc.returncode == 0:
        result["success"] = True
        break
    match = re.search(r"No module named '([^']+)'", proc.stderr)
    if match:
        missing_lib = match.group(1)
        result["installed"].append(missing_lib)
        print(f"[INFO] Missing library {{missing_lib}}, installing...", file=sys.stderr)
        subprocess.run([sys.executable, "-m", "pip", "install", missing_lib],
                       capture_output=True, text=True)
    else:
        break

print(json.dumps(result, ensure_ascii=False))
""")

            cmd = [
                "docker", "run", "--rm",
                "-v", f"{os.path.abspath(tmpdir)}:/app",
                "-w", "/app",
                docker_image,
                "python", "validate_script.py"
            ]
            try:
                output = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if output.returncode == 0:
                    try:
                        result_json = json.loads(output.stdout.strip())
                        results.append(result_json)
                    except json.JSONDecodeError:
                        results.append({
                            "index": idx,
                            "success": False,
                            "stdout": output.stdout,
                            "stderr": "Failed to parse validation output as JSON",
                            "installed": []
                        })
                else:
                    results.append({
                        "index": idx,
                        "success": False,
                        "stdout": "",
                        "stderr": output.stderr,
                        "installed": []
                    })
            except subprocess.TimeoutExpired:
                results.append({
                    "index": idx,
                    "success": False,
                    "stdout": "",
                    "stderr": "Docker container execution timed out",
                    "installed": []
                })

    return results