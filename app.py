import cv2
import numpy as np
from flask import Flask, render_template_string, Response, jsonify, request, session, redirect, url_for
import datetime
import threading
import requests
import sqlite3
import time
import json
import uuid
from ultralytics import YOLO
from functools import wraps

app = Flask(__name__)
app.secret_key = "stampede_safe_secret_2024"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
VIDEO_SOURCE   = "crowd.mp4"

GRID_R, GRID_C = 6, 6
MAX_CELL       = 4      # per-cell density limit; DANGER only when ALL cells exceed this
WARNING_RATIO  = 0.5    # fraction of cells that must exceed MAX_CELL to trigger WARNING
CONF_THRESHOLD = 0.45
NMS_IOU        = 0.45
HEATMAP_DECAY  = 0.90
FRAME_SKIP     = 2
JPEG_QUALITY   = 72
MAX_ALERTS     = 20
SMS_COOLDOWN   = 60

FAST2SMS_KEY = "bnoAO4R6hGXwT2Eeaz58kQfMFrW3LVldCjyYv0S9cUNDJIBsqPTfbQAKF6LpqdzCUr1x4kcRJBGw2Zui"
ALERT_PHONE  = "9847910674"
DB_PATH      = "stampede_safe_v3.db"

# Demo users  {username: password}
USERS = {"admin": "admin123", "operator": "op456"}

# ─────────────────────────────────────────────
# SSE  — per-client notification queues
# ─────────────────────────────────────────────
sse_clients: dict[str, list] = {}   # session_id → list of pending event strings
sse_lock = threading.Lock()

def sse_push(event_type: str, data: dict):
    """Push an SSE event to ALL connected clients."""
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with sse_lock:
        for q in sse_clients.values():
            q.append(payload)

# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS crowd_log (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                ts       TEXT    NOT NULL,
                count    INTEGER NOT NULL,
                status   TEXT    NOT NULL,
                max_zone INTEGER NOT NULL,
                saturated_cells INTEGER NOT NULL
            )""")
        con.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                ts     TEXT NOT NULL,
                status TEXT NOT NULL,
                count  INTEGER NOT NULL,
                saturated_cells INTEGER NOT NULL
            )""")
        con.commit()

def log_crowd(ts, count, status, max_zone, sat):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO crowd_log VALUES(NULL,?,?,?,?,?)",
            (ts, count, status, max_zone, sat))
        con.commit()

def log_alert_db(ts, status, count, sat):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO alerts VALUES(NULL,?,?,?,?)",
            (ts, status, count, sat))
        con.commit()

def get_analytics():
    with sqlite3.connect(DB_PATH) as con:
        since = (datetime.datetime.now() - datetime.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        rows = con.execute(
            "SELECT ts,count,status,saturated_cells FROM crowd_log WHERE ts>? ORDER BY ts",
            (since,)).fetchall()
        alert_count = con.execute(
            "SELECT COUNT(*) FROM alerts WHERE ts>?", (since,)).fetchone()[0]
        peak = con.execute(
            "SELECT MAX(count) FROM crowd_log WHERE ts>?", (since,)).fetchone()[0] or 0
    return {
        "timeline":       [{"t": r[0][11:16], "c": r[1], "s": r[2], "sat": r[3]} for r in rows[-60:]],
        "alert_count_1h": alert_count,
        "peak_1h":        peak,
        "samples":        len(rows)
    }

init_db()

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
model  = YOLO("yolov8n.pt")
_dummy = np.zeros((480, 640, 3), dtype=np.uint8)
model(_dummy, classes=[0], conf=CONF_THRESHOLD, iou=NMS_IOU, verbose=False)

# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────
last_alert_status = "SAFE"
last_sms_time     = {"DANGER": 0, "WARNING": 0}
state = {
    "count": 0, "status": "SAFE",
    "alerts": [],
    "heatmap_data":    [[0]*GRID_C for _ in range(GRID_R)],
    "saturated_cells": 0,
    "total_cells":     GRID_R * GRID_C,
    "all_saturated":   False,
}
heatmap_acc = np.zeros((GRID_R, GRID_C), dtype=np.float32)
log_ticker  = 0

# ─────────────────────────────────────────────
# SMS
# ─────────────────────────────────────────────
def send_sms(message):
    try:
        r = requests.post(
            "https://www.fast2sms.com/dev/bulkV2",
            headers={"authorization": FAST2SMS_KEY},
            data={"message": message, "language": "english",
                  "route": "v3", "numbers": ALERT_PHONE},
            timeout=5)
        print(f"SMS: {r.text}")
    except Exception as e:
        print(f"SMS failed: {e}")

# ─────────────────────────────────────────────
# SPATIAL HEATMAP  — OpenCV colormap overlay
# ─────────────────────────────────────────────
def build_spatial_heatmap(frame, heatmap_acc):
    """Overlay a smooth spatial heatmap on the frame using cv2 COLORMAP_JET."""
    h, w = frame.shape[:2]
    acc  = heatmap_acc.copy()
    mx   = acc.max()
    if mx > 0:
        acc = acc / mx
    # Upscale accumulator to frame size with smooth interpolation
    hmap_up = cv2.resize(acc.astype(np.float32), (w, h),
                         interpolation=cv2.INTER_CUBIC)
    hmap_up = np.clip(hmap_up, 0, 1)
    hmap_u8 = (hmap_up * 255).astype(np.uint8)
    # Apply JET colormap then blend
    colored = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(frame, 0.60, colored, 0.40, 0)
    return blended

# ─────────────────────────────────────────────
# FRAME GENERATOR
# ─────────────────────────────────────────────
def gen_frames():
    global last_alert_status, heatmap_acc, log_ticker

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    frame_idx   = 0
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 480))
        h, w  = frame.shape[:2]
        frame_idx += 1

        # ── Detection ────────────────────────────────────────────
        if frame_idx % FRAME_SKIP == 0 or last_result is None:
            results     = model(frame, classes=[0],
                                conf=CONF_THRESHOLD, iou=NMS_IOU,
                                verbose=False)
            last_result = results
        else:
            results = last_result

        people = results[0].boxes
        total  = len(people)

        # ── Grid count ────────────────────────────────────────────
        grid = np.zeros((GRID_R, GRID_C), dtype=int)
        for box in people:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            gc = min(cx * GRID_C // w, GRID_C - 1)
            gr = min(cy * GRID_R // h, GRID_R - 1)
            grid[gr][gc] += 1

        # ── Saturation counts ─────────────────────────────────────
        total_cells     = GRID_R * GRID_C
        saturated_cells = int(np.sum(grid >= MAX_CELL))
        all_saturated   = (saturated_cells == total_cells)
        warning_thresh  = int(total_cells * WARNING_RATIO)

        # ── Status: DANGER only when ALL cells exceed MAX_CELL ────
        if all_saturated:
            status = "DANGER"
        elif saturated_cells >= warning_thresh:
            status = "WARNING"
        else:
            status = "SAFE"

        # ── Heatmap accumulator ───────────────────────────────────
        heatmap_acc = heatmap_acc * HEATMAP_DECAY + grid.astype(np.float32)

        # ── Update shared state ───────────────────────────────────
        state["count"]           = total
        state["status"]          = status
        state["heatmap_data"]    = grid.tolist()
        state["saturated_cells"] = saturated_cells
        state["total_cells"]     = total_cells
        state["all_saturated"]   = all_saturated

        # ── Build spatial heatmap frame ───────────────────────────
        annotated  = results[0].plot()
        frame_heat = build_spatial_heatmap(annotated, heatmap_acc)

        # ── Draw grid cell outlines + counts ─────────────────────
        cw, ch = w // GRID_C, h // GRID_R
        for r in range(GRID_R):
            for c in range(GRID_C):
                x1o, y1o = c * cw, r * ch
                x2o, y2o = x1o + cw, y1o + ch
                saturated = grid[r][c] >= MAX_CELL
                # Border: red if saturated, white dim otherwise
                border_col = (0, 0, 255) if saturated else (200, 200, 200)
                border_w   = 2           if saturated else 1
                cv2.rectangle(frame_heat, (x1o, y1o), (x2o, y2o),
                              border_col, border_w)
                if grid[r][c] > 0:
                    label_col = (0, 0, 255) if saturated else (255, 255, 255)
                    cv2.putText(frame_heat, str(grid[r][c]),
                                (x1o + 6, y1o + 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                label_col, 2)

        # ── ALL-SATURATED banner ──────────────────────────────────
        if all_saturated:
            cv2.rectangle(frame_heat, (0, 0), (w, 50), (0, 0, 200), -1)
            cv2.putText(frame_heat, "!! ALL ZONES SATURATED — STAMPEDE RISK !!",
                        (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)
        else:
            # Normal HUD
            hud_col = (0, 0, 180) if status == "DANGER" else (0, 130, 200) if status == "WARNING" else (0, 160, 60)
            label   = f"{status}  |  People: {total}  |  Saturated: {saturated_cells}/{total_cells}"
            cv2.putText(frame_heat, label, (8, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_col, 2)

        # ── Alert + SSE push ──────────────────────────────────────
        now_dt = datetime.datetime.now()
        if status != last_alert_status and status in ("DANGER", "WARNING"):
            ts_str = now_dt.strftime("%H:%M:%S")
            entry  = {"time": ts_str, "status": status,
                      "count": total, "saturated": saturated_cells}
            state["alerts"].insert(0, entry)
            state["alerts"] = state["alerts"][:MAX_ALERTS]
            log_alert_db(now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                         status, total, saturated_cells)
            last_alert_status = status

            # Push browser notification via SSE
            sse_push("alert", {
                "status":    status,
                "count":     total,
                "saturated": saturated_cells,
                "total":     total_cells,
                "time":      ts_str,
                "all_sat":   all_saturated,
                "message":   (
                    f"⚠️ STAMPEDE DANGER! ALL {total_cells} zones saturated! "
                    f"{total} people detected."
                    if all_saturated else
                    f"{'🔴 DANGER' if status=='DANGER' else '🟠 WARNING'}: "
                    f"{saturated_cells}/{total_cells} zones over limit. "
                    f"{total} people detected."
                )
            })

            if time.time() - last_sms_time.get(status, 0) > SMS_COOLDOWN:
                last_sms_time[status] = time.time()
                sms_text = (
                    f"StampedeSafe CRITICAL: ALL {total_cells} zones saturated! "
                    f"{total} people at {ts_str}. Immediate action required!"
                    if all_saturated else
                    f"StampedeSafe {status}: {saturated_cells}/{total_cells} zones over limit. "
                    f"{total} people at {ts_str}."
                )
                threading.Thread(target=send_sms, args=(sms_text,), daemon=True).start()

        elif status == "SAFE":
            if last_alert_status != "SAFE":
                sse_push("clear", {"message": "✅ All zones back to safe density."})
            last_alert_status = "SAFE"

        # ── DB logging ────────────────────────────────────────────
        log_ticker += 1
        if log_ticker % 60 == 0:
            ts_str   = now_dt.strftime("%Y-%m-%d %H:%M:%S")
            max_zone = int(grid.max())
            threading.Thread(
                target=log_crowd,
                args=(ts_str, total, status, max_zone, saturated_cells),
                daemon=True
            ).start()

        # ── Encode ────────────────────────────────────────────────
        _, buf = cv2.imencode(".jpg", frame_heat,
                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")


# ─────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────
LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><title>StampedeSafe — Login</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
<style>
:root{--gold:#C9A84C;--bg:#070B14;--surface:#0C1220;--border:#C9A84C30;--text:#dce4f0;--muted:#5a6b82;--danger:#FF3B3B}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);font-family:'Rajdhani',sans-serif;color:var(--text);
     min-height:100vh;display:flex;align-items:center;justify-content:center}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
      padding:42px 40px;width:360px;box-shadow:0 0 60px #C9A84C0a}
.logo{text-align:center;margin-bottom:32px}
.logo-icon{font-size:36px}
.logo-name{font-size:22px;font-weight:700;color:var(--gold);letter-spacing:2px;text-transform:uppercase;margin-top:6px}
.logo-tag{font-size:10px;color:var(--muted);letter-spacing:3px;text-transform:uppercase}
label{display:block;font-size:10px;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-bottom:6px}
input{width:100%;background:#0a0f1e;border:1px solid var(--border);border-radius:6px;
      padding:10px 14px;color:var(--text);font-family:'JetBrains Mono',monospace;font-size:13px;outline:none;margin-bottom:18px}
input:focus{border-color:var(--gold)}
button{width:100%;background:var(--gold);color:#070B14;border:none;border-radius:6px;
       padding:12px;font-family:'Rajdhani',sans-serif;font-size:14px;font-weight:700;
       letter-spacing:2px;text-transform:uppercase;cursor:pointer;margin-top:4px}
button:hover{background:#e8c96a}
.err{background:#FF3B3B18;border:1px solid #FF3B3B50;color:var(--danger);
     border-radius:6px;padding:8px 12px;font-size:12px;margin-bottom:16px;text-align:center}
</style>
</head>
<body>
<div class="card">
  <div class="logo">
    <div class="logo-icon">🚨</div>
    <div class="logo-name">StampedeSafe</div>
    <div class="logo-tag">Crowd Intelligence System</div>
  </div>
  {% if error %}<div class="err">{{ error }}</div>{% endif %}
  <form method="POST">
    <label>Username</label>
    <input type="text" name="username" autocomplete="off" required>
    <label>Password</label>
    <input type="password" name="password" required>
    <button type="submit">Sign In</button>
  </form>
</div>
</body>
</html>"""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><title>StampedeSafe v3</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --gold:#C9A84C;--danger:#FF3B3B;--warn:#FF9500;
  --safe:#00E676;--bg:#070B14;--surface:#0C1220;--border:#C9A84C28;
  --text:#dce4f0;--muted:#5a6b82;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;background:var(--bg);font-family:'Rajdhani',sans-serif;color:var(--text);overflow-x:hidden}

/* ── Header ── */
.hdr{display:flex;align-items:center;justify-content:space-between;padding:12px 24px;
     background:var(--surface);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:200}
.logo{display:flex;align-items:center;gap:10px}
.logo-icon{font-size:24px}
.logo-name{font-size:18px;font-weight:700;color:var(--gold);letter-spacing:2px;text-transform:uppercase}
.logo-tag{font-size:9px;color:var(--muted);letter-spacing:3px;text-transform:uppercase}
.hdr-right{display:flex;align-items:center;gap:14px}
.user-chip{font-size:11px;color:var(--muted);font-family:'JetBrains Mono',monospace;
           background:#0a0f1e;padding:4px 10px;border-radius:4px;border:1px solid var(--border)}
.logout-btn{font-size:10px;color:var(--muted);text-decoration:none;letter-spacing:1px;
            padding:4px 10px;border:1px solid var(--border);border-radius:4px}
.logout-btn:hover{color:var(--gold);border-color:var(--gold)}
.badge{padding:5px 16px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:2px;
       text-transform:uppercase;font-family:'JetBrains Mono',monospace}
.SAFE   {background:#00E67614;color:var(--safe);border:1px solid #00E67650}
.WARNING{background:#FF950014;color:var(--warn);border:1px solid #FF950050}
.DANGER {background:#FF3B3B18;color:var(--danger);border:1px solid #FF3B3B60;animation:blink .7s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.4}}

/* ── Push notification toast ── */
#toast-container{position:fixed;top:70px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:10px;pointer-events:none}
.toast{background:#0C1220;border-radius:10px;padding:14px 18px;min-width:320px;max-width:400px;
       pointer-events:all;animation:slideIn .3s ease;box-shadow:0 4px 30px #00000060}
.toast.danger-toast{border-left:4px solid var(--danger);border:1px solid #FF3B3B60;border-left:4px solid var(--danger)}
.toast.warning-toast{border-left:4px solid var(--warn);border:1px solid #FF950060;border-left:4px solid var(--warn)}
.toast.safe-toast{border-left:4px solid var(--safe);border:1px solid #00E67660;border-left:4px solid var(--safe)}
.toast-title{font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px}
.toast.danger-toast .toast-title{color:var(--danger)}
.toast.warning-toast .toast-title{color:var(--warn)}
.toast.safe-toast .toast-title{color:var(--safe)}
.toast-msg{font-size:12px;color:var(--text);line-height:1.5}
.toast-time{font-size:10px;color:var(--muted);font-family:'JetBrains Mono',monospace;margin-top:6px}
.toast-close{position:absolute;top:8px;right:12px;font-size:16px;color:var(--muted);cursor:pointer;line-height:1}
.toast-close:hover{color:var(--text)}
@keyframes slideIn{from{transform:translateX(120%);opacity:0}to{transform:translateX(0);opacity:1}}
@keyframes slideOut{from{transform:translateX(0);opacity:1}to{transform:translateX(120%);opacity:0}}

/* ── Saturation bar ── */
.sat-bar-wrap{padding:10px 16px 4px;background:var(--surface);border-bottom:1px solid var(--border)}
.sat-label{font-size:9px;letter-spacing:3px;color:var(--muted);text-transform:uppercase;margin-bottom:5px;display:flex;justify-content:space-between}
.sat-track{background:#0a0f1e;border-radius:3px;height:8px;overflow:hidden}
.sat-fill{height:100%;border-radius:3px;transition:width .5s,background .5s}

/* ── Layout ── */
.body-grid{display:grid;grid-template-columns:1fr 360px;gap:14px;padding:14px 20px;min-height:calc(100vh - 100px)}
.feed-wrap{grid-row:1/3}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden}
.card-hdr{padding:9px 14px;border-bottom:1px solid var(--border);font-size:9px;font-weight:600;
          color:var(--gold);letter-spacing:3px;text-transform:uppercase;display:flex;align-items:center;gap:8px}
.live-dot{width:7px;height:7px;background:var(--safe);border-radius:50%;box-shadow:0 0 6px var(--safe);animation:blink 1.6s infinite}
.feed-wrap img{width:100%;display:block}

.stats-row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.stat-box{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 18px}
.stat-lbl{font-size:9px;letter-spacing:3px;color:var(--muted);text-transform:uppercase;margin-bottom:4px}
.stat-val{font-size:38px;font-weight:700;color:var(--gold);line-height:1;font-family:'JetBrains Mono',monospace}
.stat-sub{font-size:10px;color:var(--muted);margin-top:3px}

/* ── Tabs ── */
.tabs{display:flex;border-bottom:1px solid var(--border)}
.tab{padding:7px 14px;font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;
     color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;transition:all .2s}
.tab.active{color:var(--gold);border-bottom-color:var(--gold)}
.tab-panel{display:none}.tab-panel.active{display:block}

/* ── Heatmap grid ── */
.heatmap-grid{display:grid;gap:2px;padding:10px}
.hm-cell{border-radius:3px;display:flex;align-items:center;justify-content:center;
         font-size:10px;font-weight:600;font-family:'JetBrains Mono',monospace;
         color:rgba(255,255,255,.9);transition:background .3s;aspect-ratio:1;position:relative}
.hm-cell.saturated::after{content:'';position:absolute;inset:0;border:2px solid #FF3B3B;border-radius:3px;animation:blink .7s infinite}
.legend-row{display:flex;gap:14px;padding:8px 14px;flex-wrap:wrap}
.leg-item{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--muted)}
.leg-dot{width:9px;height:9px;border-radius:2px}

/* ── Alerts ── */
.alerts-body{max-height:220px;overflow-y:auto;padding:0 10px 6px}
.alerts-body::-webkit-scrollbar{width:3px}
.alerts-body::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.alert-row{display:flex;align-items:flex-start;gap:8px;padding:6px 0;border-bottom:1px solid #ffffff07;font-size:11px}
.adot{width:6px;height:6px;border-radius:50%;flex-shrink:0;margin-top:3px}
.adot.DANGER{background:var(--danger);box-shadow:0 0 4px var(--danger)}
.adot.WARNING{background:var(--warn);box-shadow:0 0 4px var(--warn)}
.atime{color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:9px;min-width:48px}
.atxt{color:var(--text);line-height:1.4}
.no-alert{font-size:11px;color:var(--muted);text-align:center;padding:16px 0}

/* ── Analytics ── */
.ana-stats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;padding:10px}
.ana-box{background:#0a0f1e;border-radius:5px;padding:8px;text-align:center}
.ana-n{font-size:22px;font-weight:700;color:var(--gold);font-family:'JetBrains Mono',monospace}
.ana-l{font-size:8px;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-top:2px}
#ana-chart-wrap{padding:0 10px 10px;height:110px}
canvas#ana-chart{width:100%!important;height:100px!important}

/* ── Sparkline ── */
#chart-wrap{padding:10px;height:118px}
canvas#chart{width:100%!important;height:98px!important}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
</head>
<body>

<!-- ── Toast container ── -->
<div id="toast-container"></div>

<!-- ── Header ── -->
<div class="hdr">
  <div class="logo">
    <div class="logo-icon">🚨</div>
    <div>
      <div class="logo-name">StampedeSafe</div>
      <div class="logo-tag">AI Crowd Intelligence · v3</div>
    </div>
  </div>
  <div class="hdr-right">
    <span class="user-chip">👤 {{ username }}</span>
    <a href="/logout" class="logout-btn">Logout</a>
    <div id="badge" class="badge SAFE">SAFE</div>
  </div>
</div>

<!-- ── Saturation bar ── -->
<div class="sat-bar-wrap">
  <div class="sat-label">
    <span>Zone Saturation</span>
    <span id="sat-label-right">0 / {{ total_cells }} zones above limit</span>
  </div>
  <div class="sat-track">
    <div id="sat-fill" class="sat-fill" style="width:0%;background:var(--safe)"></div>
  </div>
</div>

<!-- ── Body ── -->
<div class="body-grid">

  <!-- Feed -->
  <div class="card feed-wrap">
    <div class="card-hdr"><div class="live-dot"></div>Spatial Heatmap · YOLOv8 Detection · All-Zone Density Monitoring</div>
    <img src="/video_feed" alt="live feed">
  </div>

  <!-- Right column -->
  <div style="display:flex;flex-direction:column;gap:12px">

    <div class="stats-row">
      <div class="stat-box">
        <div class="stat-lbl">Detected</div>
        <div class="stat-val" id="count">0</div>
        <div class="stat-sub">people in frame</div>
      </div>
      <div class="stat-box">
        <div class="stat-lbl">Saturated</div>
        <div class="stat-val" id="sat-count">0</div>
        <div class="stat-sub">of {{ total_cells }} zones</div>
      </div>
    </div>

    <!-- Tabs -->
    <div class="card" style="flex:1">
      <div class="tabs">
        <div class="tab active" onclick="switchTab('heatmap',this)">Heatmap</div>
        <div class="tab" onclick="switchTab('analytics',this)">Analytics</div>
        <div class="tab" onclick="switchTab('alerts',this)">Alerts</div>
      </div>

      <div id="tab-heatmap" class="tab-panel active">
        <div id="heatmap-grid" class="heatmap-grid"
             style="grid-template-columns:repeat(6,1fr)"></div>
        <div class="legend-row">
          <div class="leg-item"><div class="leg-dot" style="background:#1a3a1a"></div>Empty</div>
          <div class="leg-item"><div class="leg-dot" style="background:#2a7a1a"></div>Low</div>
          <div class="leg-item"><div class="leg-dot" style="background:#c07800"></div>High</div>
          <div class="leg-item"><div class="leg-dot" style="background:#cc1100"></div>Saturated</div>
        </div>
      </div>

      <div id="tab-analytics" class="tab-panel">
        <div class="ana-stats">
          <div class="ana-box"><div class="ana-n" id="ana-peak">–</div><div class="ana-l">Peak 1h</div></div>
          <div class="ana-box"><div class="ana-n" id="ana-alerts">–</div><div class="ana-l">Alerts 1h</div></div>
          <div class="ana-box"><div class="ana-n" id="ana-samples">–</div><div class="ana-l">Samples</div></div>
        </div>
        <div id="ana-chart-wrap"><canvas id="ana-chart"></canvas></div>
      </div>

      <div id="tab-alerts" class="tab-panel">
        <div class="alerts-body" id="alerts"><div class="no-alert">No alerts yet</div></div>
      </div>
    </div>

    <div class="card">
      <div class="card-hdr">Live Count</div>
      <div id="chart-wrap"><canvas id="chart"></canvas></div>
    </div>
  </div>
</div>

<script>
const TOTAL_CELLS = {{ total_cells }};
const MAX_CELL    = {{ max_cell }};

// ── Sparkline ────────────────────────────────────────────────
const sparkCtx = document.getElementById('chart').getContext('2d');
const sparkData={labels:[],datasets:[{data:[],fill:true,
  borderColor:'#C9A84C',backgroundColor:'#C9A84C18',
  borderWidth:1.5,pointRadius:0,tension:0.4}]};
const sparkChart=new Chart(sparkCtx,{
  type:'line',data:sparkData,
  options:{animation:false,responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false},tooltip:{enabled:false}},
    scales:{x:{display:false},y:{display:true,min:0,
      grid:{color:'#ffffff08'},ticks:{color:'#5a6b82',font:{size:9},maxTicksLimit:4}}}}
});
function pushSpark(n){
  const t=new Date().toLocaleTimeString('en',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
  if(sparkData.labels.length>60){sparkData.labels.shift();sparkData.datasets[0].data.shift();}
  sparkData.labels.push(t);sparkData.datasets[0].data.push(n);sparkChart.update('none');
}

// ── Analytics chart ──────────────────────────────────────────
const anaCtx=document.getElementById('ana-chart').getContext('2d');
const anaChart=new Chart(anaCtx,{type:'bar',
  data:{labels:[],datasets:[{data:[],backgroundColor:'#C9A84C44',borderColor:'#C9A84C',borderWidth:1}]},
  options:{animation:false,responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#5a6b82',font:{size:8},maxTicksLimit:10},grid:{display:false}},
            y:{min:0,ticks:{color:'#5a6b82',font:{size:9},maxTicksLimit:4},grid:{color:'#ffffff08'}}}}
});

// ── Heatmap ──────────────────────────────────────────────────
function hmColor(v,mx){
  if(mx===0||v===0) return '#1a3a1a';
  const r=v/mx;
  if(r<0.33) return `rgba(20,${Math.round(60+r*300)},20,0.9)`;
  if(r<0.66) return `rgba(${Math.round(r*350)},${Math.round(120-r*80)},10,0.9)`;
  return `rgba(${Math.round(180+r*75)},${Math.round(30-r*20)},10,0.9)`;
}
function renderHeatmap(grid){
  const el=document.getElementById('heatmap-grid');
  let mx=0;
  grid.forEach(row=>row.forEach(v=>{if(v>mx)mx=v;}));
  el.innerHTML=grid.map(row=>
    row.map(v=>{
      const sat=v>=MAX_CELL;
      return `<div class="hm-cell${sat?' saturated':''}" style="background:${hmColor(v,mx)}">${v||''}</div>`;
    }).join('')
  ).join('');
}

// ── Saturation bar ────────────────────────────────────────────
function updateSatBar(sat,total){
  const pct=Math.round(sat/total*100);
  const fill=document.getElementById('sat-fill');
  fill.style.width=pct+'%';
  fill.style.background=pct>=100?'#FF3B3B':pct>=50?'#FF9500':'#00E676';
  document.getElementById('sat-label-right').textContent=`${sat} / ${total} zones above limit`;
  document.getElementById('sat-count').textContent=sat;
}

// ── Toast notifications ───────────────────────────────────────
function showToast(type,title,msg,time){
  const container=document.getElementById('toast-container');
  const id='toast-'+Date.now();
  const cls=type==='DANGER'?'danger-toast':type==='WARNING'?'warning-toast':'safe-toast';
  const div=document.createElement('div');
  div.id=id; div.className=`toast ${cls}`;
  div.innerHTML=`
    <div class="toast-close" onclick="dismissToast('${id}')">×</div>
    <div class="toast-title">${title}</div>
    <div class="toast-msg">${msg}</div>
    <div class="toast-time">${time}</div>`;
  container.prepend(div);
  // Auto-dismiss after 8 s
  setTimeout(()=>dismissToast(id), 8000);
  // Also try browser Notification API
  if(Notification.permission==='granted'){
    new Notification('StampedeSafe Alert',{body:msg,icon:'/static/icon.png'});
  }
}
function dismissToast(id){
  const el=document.getElementById(id);
  if(!el)return;
  el.style.animation='slideOut .3s ease forwards';
  setTimeout(()=>el.remove(),300);
}

// Request browser notification permission on load
if(Notification.permission==='default') Notification.requestPermission();

// ── SSE  — server-sent events for real-time push ──────────────
const evtSource=new EventSource('/stream');
evtSource.addEventListener('alert',e=>{
  const d=JSON.parse(e.data);
  const title=d.all_sat
    ? '🚨 STAMPEDE DANGER — ALL ZONES SATURATED'
    : (d.status==='DANGER'?'🔴 DANGER ALERT':'🟠 WARNING ALERT');
  showToast(d.status,title,d.message,d.time);
});
evtSource.addEventListener('clear',e=>{
  const d=JSON.parse(e.data);
  showToast('SAFE','✅ CLEARED',d.message,new Date().toLocaleTimeString());
});

// ── Tab switching ────────────────────────────────────────────
function switchTab(name,el){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
  if(name==='analytics') fetchAnalytics();
}

// ── State poll ───────────────────────────────────────────────
function refreshState(){
  fetch('/state').then(r=>r.json()).then(d=>{
    const b=document.getElementById('badge');
    b.textContent=d.status; b.className='badge '+d.status;
    document.getElementById('count').textContent=d.count;
    updateSatBar(d.saturated_cells,d.total_cells);
    pushSpark(d.count);
    renderHeatmap(d.heatmap_data);
    const al=document.getElementById('alerts');
    if(!d.alerts.length){
      al.innerHTML='<div class="no-alert">No alerts yet</div>';
    } else {
      al.innerHTML=d.alerts.map(a=>
        `<div class="alert-row">
          <div class="adot ${a.status}"></div>
          <div class="atime">${a.time}</div>
          <div class="atxt">${a.status} — ${a.count} people · ${a.saturated} zones saturated</div>
        </div>`).join('');
    }
  }).catch(()=>{});
}

function fetchAnalytics(){
  fetch('/analytics').then(r=>r.json()).then(d=>{
    document.getElementById('ana-peak').textContent   =d.peak_1h;
    document.getElementById('ana-alerts').textContent =d.alert_count_1h;
    document.getElementById('ana-samples').textContent=d.samples;
    anaChart.data.labels=d.timeline.map(x=>x.t);
    anaChart.data.datasets[0].data=d.timeline.map(x=>x.c);
    anaChart.data.datasets[0].backgroundColor=d.timeline.map(x=>
      x.s==='DANGER'?'#FF3B3B60':x.s==='WARNING'?'#FF950060':'#C9A84C44');
    anaChart.update('none');
  }).catch(()=>{});
}

setInterval(refreshState,1800);
refreshState();
</script>
</body>
</html>"""

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/login', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        u = request.form.get('username','')
        p = request.form.get('password','')
        if USERS.get(u) == p:
            session['user'] = u
            session['sid']  = str(uuid.uuid4())
            with sse_lock:
                sse_clients[session['sid']] = []
            return redirect(url_for('index'))
        error = "Invalid username or password."
    return render_template_string(LOGIN_HTML, error=error)

@app.route('/logout')
def logout():
    sid = session.get('sid')
    if sid:
        with sse_lock:
            sse_clients.pop(sid, None)
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template_string(
        DASHBOARD_HTML,
        username=session['user'],
        total_cells=GRID_R * GRID_C,
        max_cell=MAX_CELL
    )

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
@login_required
def get_state():
    return jsonify(state)

@app.route('/analytics')
@login_required
def analytics():
    return jsonify(get_analytics())

@app.route('/stream')
@login_required
def stream():
    """SSE endpoint — pushes alert events to the logged-in browser tab."""
    sid = session.get('sid', str(uuid.uuid4()))
    with sse_lock:
        if sid not in sse_clients:
            sse_clients[sid] = []

    def event_generator():
        yield "data: connected\n\n"
        while True:
            with sse_lock:
                q = sse_clients.get(sid, [])
                if q:
                    msgs = q[:]
                    sse_clients[sid] = []
                else:
                    msgs = []
            for msg in msgs:
                yield msg
            time.sleep(0.5)

    return Response(event_generator(),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)