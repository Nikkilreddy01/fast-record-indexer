#!/usr/bin/env python3
"""
Odin Business Entity Resolution — Live Progress & ETA Monitor
Real-time tracking of processed entities, inference throughput, and ETA.
Works on macOS, Linux, and Windows.
"""

import os
import sys
import time
import re
import glob
import datetime
import subprocess

# Country totals for Amazon ML Challenge 2026 test set
COUNTRY_TOTALS = {
    "FRANCE": 259452,
    "US": 663106,
    "INDIA": 809986,
}
TOTAL_ENTITIES = sum(COUNTRY_TOTALS.values())  # 1,732,544

def find_active_log_file(custom_path=None):
    """Auto-locate the most relevant log file."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    # 1. Local symlink or project log
    candidates = [
        "pipeline.log",
        "output/pipeline.log",
        os.path.join(os.path.dirname(__file__), "pipeline.log"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    # 2. Check Antigravity / Gemini brain task logs
    home = os.path.expanduser("~")
    task_logs = glob.glob(f"{home}/.gemini/antigravity-ide/brain/*/.system_generated/tasks/task-*.log")
    if task_logs:
        task_logs.sort(key=os.path.getmtime, reverse=True)
        return task_logs[0]

    return None

def get_process_info():
    """Find running pipeline.py process status."""
    info = {"running": False, "pid": None, "cpu": None, "mem": None, "elapsed": None}
    try:
        # Cross-platform check using ps
        if sys.platform != "win32":
            res = subprocess.run(["ps", "-eo", "pid,%cpu,%mem,etime,command"], stdout=subprocess.PIPE, text=True, check=False)
            for line in res.stdout.splitlines():
                if "pipeline.py" in line and "grep" not in line and "monitor.py" not in line:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        info["running"] = True
                        info["pid"] = parts[0]
                        info["cpu"] = parts[1] + "%"
                        info["mem"] = parts[2] + "%"
                        info["elapsed"] = parts[3]
                        break
        else:
            # Windows tasklist check
            res = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"], stdout=subprocess.PIPE, text=True, check=False)
            if "python.exe" in res.stdout:
                info["running"] = True
                info["cpu"] = "Active"
    except Exception:
        pass
    return info

def format_seconds(seconds):
    """Format seconds into human-readable HH:MM:SS string."""
    if seconds is None or seconds < 0:
        return "Unknown"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"

def make_bar(percent, width=30):
    """Generate ASCII progress bar."""
    filled = int(width * (percent / 100.0))
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    return bar

def parse_log(log_path):
    """Parse the log file to extract current progress and ETA."""
    if not log_path or not os.path.exists(log_path):
        return None

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return None

    state = {
        "log_path": log_path,
        "france": {"status": "PENDING", "done": 0, "total": COUNTRY_TOTALS["FRANCE"], "speed": None, "time": None, "matches": None, "singletons": None},
        "us": {"status": "PENDING", "done": 0, "total": COUNTRY_TOTALS["US"], "speed": None, "time": None, "matches": None, "singletons": None},
        "india": {"status": "PENDING", "done": 0, "total": COUNTRY_TOTALS["INDIA"], "speed": None, "time": None, "eta": None, "matches": None, "singletons": None},
        "current_country": None,
        "is_complete": False,
        "speed": 0.0,
        "eta_seconds": None,
    }

    # France checks
    if "Completed France" in content:
        state["france"]["status"] = "COMPLETE"
        state["france"]["done"] = COUNTRY_TOTALS["FRANCE"]
        m = re.search(r"Completed France in ([\d\.]+)s \| Matches: ([\d,]+), Singletons: ([\d,]+)", content)
        if m:
            sec = float(m.group(1))
            state["france"]["time"] = format_seconds(sec)
            state["france"]["matches"] = m.group(2)
            state["france"]["singletons"] = m.group(3)
            state["france"]["speed"] = f"{COUNTRY_TOTALS['FRANCE'] / max(1, sec):.1f} ent/s"
    elif "Processing Country: FRANCE" in content or "Inference France" in content:
        state["france"]["status"] = "RUNNING"
        state["current_country"] = "FRANCE"
        # tqdm regex
        matches = list(re.finditer(r"Inference France:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^<]+)<([^,]+),\s*([\d\.]+) entities/s\]", content))
        if matches:
            last = matches[-1]
            state["france"]["done"] = int(last.group(2))
            state["france"]["speed"] = f"{float(last.group(6)):.1f} ent/s"
            state["speed"] = float(last.group(6))

    # US checks
    if "Completed US" in content:
        state["us"]["status"] = "COMPLETE"
        state["us"]["done"] = COUNTRY_TOTALS["US"]
        m = re.search(r"Completed US in ([\d\.]+)s \| Matches: ([\d,]+), Singletons: ([\d,]+)", content)
        if m:
            sec = float(m.group(1))
            state["us"]["time"] = format_seconds(sec)
            state["us"]["matches"] = m.group(2)
            state["us"]["singletons"] = m.group(3)
            state["us"]["speed"] = f"{COUNTRY_TOTALS['US'] / max(1, sec):.1f} ent/s"
    elif "Processing Country: US" in content or "Inference US" in content:
        state["us"]["status"] = "RUNNING"
        state["current_country"] = "US"
        matches = list(re.finditer(r"Inference US:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^<]+)<([^,]+),\s*([\d\.]+) entities/s\]", content))
        if matches:
            last = matches[-1]
            state["us"]["done"] = int(last.group(2))
            state["us"]["speed"] = f"{float(last.group(6)):.1f} ent/s"
            state["speed"] = float(last.group(6))

    # India checks
    if "Completed India" in content:
        state["india"]["status"] = "COMPLETE"
        state["india"]["done"] = COUNTRY_TOTALS["INDIA"]
        m = re.search(r"Completed India in ([\d\.]+)s \| Matches: ([\d,]+), Singletons: ([\d,]+)", content)
        if m:
            sec = float(m.group(1))
            state["india"]["time"] = format_seconds(sec)
            state["india"]["matches"] = m.group(2)
            state["india"]["singletons"] = m.group(3)
            state["india"]["speed"] = f"{COUNTRY_TOTALS['INDIA'] / max(1, sec):.1f} ent/s"
    elif "Processing Country: INDIA" in content or "Inference India" in content:
        state["india"]["status"] = "RUNNING"
        state["current_country"] = "INDIA"
        matches = list(re.finditer(r"Inference India:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^<]+)<([^,]+),\s*([\d\.]+) entities/s\]", content))
        if matches:
            last = matches[-1]
            state["india"]["done"] = int(last.group(2))
            eta_str = last.group(5).strip()
            state["india"]["eta"] = eta_str
            state["india"]["speed"] = f"{float(last.group(6)):.1f} ent/s"
            state["speed"] = float(last.group(6))

            # Convert eta_str (e.g. 2:17:33 or 45:10) to seconds
            eta_parts = eta_str.split(":")
            if len(eta_parts) == 3:
                state["eta_seconds"] = int(eta_parts[0]) * 3600 + int(eta_parts[1]) * 60 + int(eta_parts[2])
            elif len(eta_parts) == 2:
                state["eta_seconds"] = int(eta_parts[0]) * 60 + int(eta_parts[1])

    # Check overall pipeline completion
    if "Done!" in content or "SUCCESS: Pipeline completed" in content or (
        state["france"]["status"] == "COMPLETE" and 
        state["us"]["status"] == "COMPLETE" and 
        state["india"]["status"] == "COMPLETE"
    ):
        state["is_complete"] = True

    # Compute overall progress
    total_done = state["france"]["done"] + state["us"]["done"] + state["india"]["done"]
    state["total_done"] = total_done
    state["total_percent"] = (total_done / TOTAL_ENTITIES) * 100.0
    state["remaining_entities"] = max(0, TOTAL_ENTITIES - total_done)

    # Calculate overall ETA based on current speed if not extracted from tqdm
    if state["eta_seconds"] is None and state["speed"] > 0:
        state["eta_seconds"] = int(state["remaining_entities"] / state["speed"])

    return state

def render_dashboard(state, proc_info, clear=True):
    """Render the dashboard UI to the console."""
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append(" ⚡ ODIN BUSINESS ENTITY RESOLUTION — LIVE PROGRESS & ETA TRACKER")
    lines.append("=" * 80)

    # Process status
    if proc_info["running"]:
        p_str = f"RUNNING (PID: {proc_info['pid']} | CPU: {proc_info['cpu']} | RAM: {proc_info['mem']} | Uptime: {proc_info['elapsed']})"
    else:
        if state and state.get("is_complete"):
            p_str = "FINISHED 🎉 (Pipeline run completed!)"
        else:
            p_str = "STANDBY / LOG-MONITORING (No active pipeline process detected)"
    
    lines.append(f" Status:        {p_str}")
    lines.append(f" Target Goal:   Macro F0.5 >= 0.982 (Leaderboard #1: 0.980473 | Calibrated Cutoff: 0.660)")
    lines.append(f" Current Time:  {datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
    lines.append("-" * 80)

    if not state:
        lines.append(" [!] Waiting for log file data... Ensure pipeline.py is running.")
        lines.append("=" * 80)
        return "\n".join(lines)

    # Overall Progress
    pct = state["total_percent"]
    bar = make_bar(pct, width=36)
    lines.append(" OVERALL PIPELINE PROGRESS (ALL COUNTRIES):")
    lines.append(f" [{bar}] {pct:5.1f}%  ({state['total_done']:,} / {TOTAL_ENTITIES:,} entities)")
    lines.append("")

    remaining = state["remaining_entities"]
    speed = state["speed"]
    eta_sec = state["eta_seconds"]

    lines.append(f"  • Remaining Entities:  {remaining:,} test Source 1 records")
    lines.append(f"  • Current Speed:       {speed:.1f} entities/sec" if speed > 0 else "  • Current Speed:       Calculating...")
    
    if state["is_complete"]:
        lines.append("  • Estimated Time Left: COMPLETED ✅")
        lines.append("  • Submission File:     output/matching_results.tsv READY!")
    elif eta_sec is not None:
        eta_formatted = format_seconds(eta_sec)
        completion_dt = datetime.datetime.now() + datetime.timedelta(seconds=eta_sec)
        completion_str = completion_dt.strftime("%I:%M:%S %p (%A)")
        lines.append(f"  • Estimated Time Left: ⏳ {eta_formatted}")
        lines.append(f"  • Est. Completion At:  🎯 {completion_str}")
    else:
        lines.append("  • Estimated Time Left: Estimating...")

    lines.append("-" * 80)
    lines.append(" COUNTRY-BY-COUNTRY PROGRESS:")
    lines.append(f" {'Country':<10} {'Status':<14} {'Processed / Total':<24} {'Throughput':<14} {'Elapsed / ETA':<16}")
    lines.append(" " + "-" * 76)

    # Country rows
    for c_key, c_name in [("france", "FRANCE"), ("us", "US"), ("india", "INDIA")]:
        c_data = state[c_key]
        status = c_data["status"]
        if status == "COMPLETE":
            status_icon = "✅ COMPLETE"
            time_str = c_data["time"] or "Done"
        elif status == "RUNNING":
            status_icon = "🔄 RUNNING"
            time_str = f"ETA: {c_data['eta']}" if c_data.get("eta") else "Active"
        else:
            status_icon = "⏳ QUEUED"
            time_str = "Waiting"

        done_str = f"{c_data['done']:,} / {c_data['total']:,}"
        speed_str = c_data["speed"] or "—"
        lines.append(f" {c_name:<10} {status_icon:<14} {done_str:<24} {speed_str:<14} {time_str:<16}")

    lines.append(" " + "-" * 76)

    # Active Country Progress Bar
    curr = state.get("current_country")
    if curr and curr.lower() in state and state[curr.lower()]["status"] == "RUNNING":
        curr_key = curr.lower()
        curr_done = state[curr_key]["done"]
        curr_total = state[curr_key]["total"]
        curr_pct = (curr_done / curr_total) * 100.0
        c_bar = make_bar(curr_pct, width=36)
        lines.append(f" Active Country ({curr}) Progress:")
        lines.append(f" [{c_bar}] {curr_pct:5.1f}%  ({curr_done:,} / {curr_total:,} entities)")
    elif state["is_complete"]:
        lines.append(" All countries successfully processed!")

    lines.append("-" * 80)
    lines.append(" WORKFLOW STAGES:")
    fr_done = state["france"]["status"] == "COMPLETE"
    us_done = state["us"]["status"] == "COMPLETE"
    in_done = state["india"]["status"] == "COMPLETE"
    
    fr_info = f"Matches: {state['france']['matches']} | Singletons: {state['france']['singletons']}" if fr_done else "Pending"
    us_info = f"Matches: {state['us']['matches']} | Singletons: {state['us']['singletons']}" if us_done else "Pending"
    in_info = f"Current batch {state['india']['done']:,} / 809,986" if not in_done else f"Matches: {state['india']['matches']}"

    lines.append(f"  [1] France Stream Inference:  {'[DONE] ' if fr_done else '[RUN]  '} {fr_info}")
    lines.append(f"  [2] US Stream Inference:      {'[DONE] ' if us_done else '[RUN]  '} {us_info}")
    lines.append(f"  [3] India Stream Inference:   {'[DONE] ' if in_done else '[RUN]  '} {in_info}")
    lines.append(f"  [4] Assemble output TSVs:     {'[DONE] ' if state['is_complete'] else '[WAIT] '} output/matching_results.tsv (1,732,544 rows)")
    lines.append(f"  [5] Validate & Package ZIP:   {'[DONE] ' if state['is_complete'] else '[WAIT] '} Odin_submission.zip for Unstop")
    lines.append("=" * 80)
    lines.append(" Press Ctrl+C to exit monitor (pipeline runs uninterrupted in background)")
    lines.append("=" * 80)

    output_text = "\n".join(lines)
    if clear:
        # Clear screen ANSI escape
        sys.stdout.write("\033[2J\033[H")
    return output_text

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live Progress & ETA Monitor for Odin Pipeline")
    parser.add_argument("--log", type=str, default=None, help="Path to pipeline log file")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds (default: 2s)")
    parser.add_argument("--once", action="store_true", help="Print a single snapshot and exit immediately")
    args = parser.parse_args()

    log_file = find_active_log_file(args.log)

    if args.once:
        state = parse_log(log_file)
        proc = get_process_info()
        print(render_dashboard(state, proc, clear=False))
        return

    try:
        while True:
            # Re-check log file if not found initially
            if not log_file or not os.path.exists(log_file):
                log_file = find_active_log_file(args.log)

            state = parse_log(log_file)
            proc = get_process_info()
            dashboard = render_dashboard(state, proc, clear=True)
            sys.stdout.write(dashboard + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        sys.stdout.write("\n\nMonitor stopped. Pipeline is continuing in the background.\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
