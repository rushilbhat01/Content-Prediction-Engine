import os
import sys
import subprocess
from pathlib import Path
import time

# Ensure output is UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

EMB_PID_FILE = DATA_DIR / "embeddings.pid"
HOOK_PID_FILE = DATA_DIR / "hooks.pid"

EMB_LOG = DATA_DIR / "watchdog_embeddings.log"
HOOK_LOG = DATA_DIR / "watchdog_hooks.log"
WD_LOG = DATA_DIR / "watchdog.log"


def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    with open(WD_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_process_running(pid_file):
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        return False

    # Check if PID is running on Windows/Unix
    if os.name == "nt":
        # Windows tasklist check
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}"],
                text=True,
                errors="ignore",
            )
            return str(pid) in out
        except Exception:
            return False
    else:
        # Unix kill check
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def start_process(cmd, pid_file, log_file):
    log(f"Starting command: {' '.join(cmd)}")
    log(f"Redirecting output to: {log_file}")
    
    # Build shell command
    cmd_str = " ".join(f'"{c}"' if " " in c or "\\" in c else c for c in cmd)
    
    # Run with UTF-8 encoding environment
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    
    full_cmd = f'cmd.exe /D /c "{cmd_str} >> "{log_file}" 2>&1"'
    proc = subprocess.Popen(
        full_cmd,
        cwd=str(BASE_DIR),
        env=env,
    )
    pid_file.write_text(str(proc.pid), encoding="utf-8")
    log(f"Process started with PID: {proc.pid} (shell). Waiting for completion...")
    proc.wait()
    log(f"Process exited with code: {proc.returncode}")


def main():
    log("=== Watchdog Run ===")

    # 1. Embeddings phase
    # Check if we still have remaining embeddings
    try:
        import pandas as pd
        df_emb = pd.read_csv(DATA_DIR / "features_embeddings.csv")
        # Check completeness (CLIP + VGGish + FER + Motion)
        required = [
            c for c in df_emb.columns 
            if c.startswith("clip_mean_") or c.startswith("vggish_mean_") or c.startswith("temporal_") or c == "df_emotion_score"
        ]
        complete_count = df_emb[required].notna().all(axis=1).sum()
        total_raw = len(list((BASE_DIR / "data" / "raw_videos").glob("*.mp4")))
        log(f"Embedding Status: {complete_count}/{total_raw} videos complete")
        emb_done = complete_count >= total_raw
    except Exception as e:
        log(f"Warning checking features_embeddings.csv: {e}")
        emb_done = False

    if not emb_done:
        if is_process_running(EMB_PID_FILE):
            log("extract_embeddings.py is already running.")
        else:
            log("extract_embeddings.py is not running. Starting/resuming it...")
            start_process(
                [sys.executable, "-u", "extract_embeddings.py"],
                EMB_PID_FILE,
                EMB_LOG,
            )
        return

    # Cleanup embeddings PID file if done
    if EMB_PID_FILE.exists():
        EMB_PID_FILE.unlink(missing_ok=True)

    # 2. Hook extraction phase
    try:
        import pandas as pd
        df_hook = pd.read_csv(DATA_DIR / "hook_features.csv")
        hook_count = len(df_hook)
        total_raw = len(list((BASE_DIR / "data" / "raw_videos").glob("*.mp4")))
        log(f"Hook Features Status: {hook_count}/{total_raw} videos complete")
        hook_done = hook_count >= total_raw
    except Exception as e:
        log(f"Warning checking hook_features.csv: {e}")
        hook_done = False

    if not hook_done:
        if is_process_running(HOOK_PID_FILE):
            log("run_hook_extraction.py is already running.")
        else:
            log("run_hook_extraction.py is not running. Starting/resuming it...")
            start_process(
                [sys.executable, "-u", "run_hook_extraction.py", "--resume"],
                HOOK_PID_FILE,
                HOOK_LOG,
            )
        return

    # Cleanup hooks PID file if done
    if HOOK_PID_FILE.exists():
        HOOK_PID_FILE.unlink(missing_ok=True)

    # 3. Model training phase
    # If all extractions are done, trigger retraining
    log("All feature extractions completed! Triggering model retraining...")
    
    # Run retraining scripts
    for script in ["model_train_fitness.py", "model_train_food.py"]:
        log(f"Running retraining: {script}")
        ret = subprocess.run([sys.executable, script], cwd=str(BASE_DIR))
        if ret.returncode == 0:
            log(f"Successfully retrained {script}")
        else:
            log(f"ERROR: Retraining failed for {script} (code {ret.returncode})")

    log("Watchdog pipeline fully completed!")


if __name__ == "__main__":
    main()
