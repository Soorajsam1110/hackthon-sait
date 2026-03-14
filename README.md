# hackthon-sait
🚨 StampedeSafe — AI Crowd Intelligence & Stampede Prevention System
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)
![SQLite](https://img.shields.io/badge/Database-SQLite-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
StampedeSafe is a real-time AI-powered crowd density monitoring and stampede
prevention system. It uses YOLOv8 to detect people in a video feed, maps them
across a spatial grid, and raises DANGER alerts only when every zone exceeds the
maximum density limit — avoiding false alarms from isolated crowding. Logged-in
users receive instant browser push notifications via Server-Sent Events (SSE),
and SMS alerts are dispatched automatically via Fast2SMS.
---
✨ Features
🎯 Real-time person detection using YOLOv8n with configurable confidence and NMS thresholds
🌡️ Spatial heatmap overlay — smooth OpenCV COLORMAP_JET heatmap blended directly onto the video feed
🔲 6×6 zone grid — each cell tracks individual density; saturated cells highlighted with flashing red border
🔴 Smart DANGER logic — DANGER fires ONLY when ALL 36 zones simultaneously exceed `MAX_CELL`
🟠 WARNING logic — triggers when 50%+ of zones are over the density limit
📊 Zone saturation progress bar — live colour-coded bar showing fraction of zones currently saturated
🔔 Browser push notifications via SSE — instant toast alerts in the dashboard, even in background tabs
🖥️ Browser Notification API — native OS-level notifications when the tab is in the background
📱 SMS alerts via Fast2SMS with per-level cooldown to prevent alert flooding
📈 Analytics dashboard — Chart.js bar chart, peak count, alert count, and sample count over the last hour
🗃️ SQLite logging — every crowd snapshot and alert event persisted to `stampede_safe_v3.db`
🔐 Login-protected dashboard — session-based authentication; all routes require login
⚡ Performance optimized — frame skipping, model warmup, threaded DB writes, small camera buffer
---
🛠️ Tech Stack
Layer	Technology
Person Detection	YOLOv8n (Ultralytics)
Backend	Python 3.8+, Flask
Video Processing	OpenCV 4.x
Heatmap	OpenCV COLORMAP_JET + INTER_CUBIC
Database	SQLite3
Real-time Push	Server-Sent Events (SSE)
SMS Alerts	Fast2SMS Bulk API
Frontend Charts	Chart.js 4.x
Auth	Flask session (cookie-based)
---
📁 Project Structure
```
stampede-safe/
├── app.py                  # Main application (single file)
├── crowd.mp4               # Input video file (not committed to git)
├── stampede_safe_v3.db     # SQLite database (auto-created on first run)
├── yolov8n.pt              # YOLOv8 nano model (auto-downloaded by Ultralytics)
└── README.md
```
---
🚀 Installation & Setup
1. Clone the repository
```bash
git clone https://github.com/yourname/stampede-safe
cd stampede-safe
```
2. Install Python dependencies
```bash
pip install flask opencv-python ultralytics requests
```
3. Add your video file
```bash
cp /path/to/your/video.mp4 crowd.mp4
```
> **Using a webcam?** Change `VIDEO_SOURCE = "crowd.mp4"` to `VIDEO_SOURCE = 0`
>
> **Using a Raspberry Pi camera stream?** Change to `VIDEO_SOURCE = "http://<PI_IP>:5001"`
4. Configure your SMS API key
Edit `app.py` and update the CONFIG section:
```python
FAST2SMS_KEY = "your_fast2sms_api_key_here"
ALERT_PHONE  = "your_10_digit_phone_number"
```
5. Run the application
```bash
python app.py
```
6. Open the dashboard
```
http://localhost:5000
```
---
⚙️ Configuration Reference
All settings are at the top of `app.py` under the `# CONFIG` section:
Parameter	Default	Description
`VIDEO_SOURCE`	`"crowd.mp4"`	Video file path, `0` for webcam, or Pi stream URL
`GRID_R`	`6`	Number of grid rows
`GRID_C`	`6`	Number of grid columns (total = 36 cells)
`MAX_CELL`	`4`	People per cell that marks it as saturated
`WARNING_RATIO`	`0.5`	Fraction of cells saturated to trigger WARNING (0.0–1.0)
`CONF_THRESHOLD`	`0.45`	YOLOv8 detection confidence threshold
`NMS_IOU`	`0.45`	Non-max suppression IoU threshold
`HEATMAP_DECAY`	`0.90`	Heatmap persistence per frame (0 = instant, 1 = permanent)
`FRAME_SKIP`	`2`	Run YOLO every Nth frame (2 = every other frame)
`JPEG_QUALITY`	`72`	Video stream JPEG quality (0–100)
`MAX_ALERTS`	`20`	Maximum alerts kept in memory for the dashboard
`SMS_COOLDOWN`	`60`	Seconds before sending another SMS of the same status
`FAST2SMS_KEY`	—	Your Fast2SMS API authorization key
`ALERT_PHONE`	—	Phone number to receive SMS (10 digits, no country code)
`DB_PATH`	`"stampede_safe_v3.db"`	SQLite database file path
---
🔐 Login Credentials
Username	Password
`admin`	`admin123`
`operator`	`op456`
To add or change users, edit the `USERS` dictionary in `app.py`:
```python
USERS = {
    "admin":    "admin123",
    "operator": "op456",
    "yourname": "yourpassword"
}
```
> ⚠️ Change default passwords before deploying to any shared or public network.
---
🧠 Detection Pipeline
```
Video File / Webcam / Pi Stream
        ↓
OpenCV — reads frame, resizes to 640×480
        ↓
YOLOv8n — detects persons (class 0)
[runs every 2nd frame; cached result used for skipped frames]
        ↓
Grid Mapping — each person's centroid mapped to a 6×6 cell
        ↓
Saturation Count — cells where count ≥ MAX_CELL are "saturated"
        ↓
  ALL 36 cells saturated? → DANGER
  ≥ 18 cells saturated?   → WARNING
  < 18 cells saturated?   → SAFE
        ↓
Spatial Heatmap built:
  heatmap_acc = heatmap_acc × DECAY + current_grid
  → resized to 640×480 with INTER_CUBIC interpolation
  → cv2.COLORMAP_JET applied
  → blended 60/40 with annotated frame
        ↓
Cell borders drawn — red + flashing if saturated
HUD text overlaid — status, people count, saturated zones
        ↓
MJPEG stream → /video_feed → dashboard <img>
        ↓
On status change to DANGER or WARNING:
  → SSE push → browser toast notification
  → Browser Notification API (if permission granted)
  → SMS via Fast2SMS (if cooldown elapsed)
  → SQLite alert log entry
Every ~2 seconds:
  → SQLite crowd_log snapshot (threaded, non-blocking)
```
---
🚦 Alert Logic
Status	Trigger Condition	Actions
✅ SAFE	Fewer than 50% of zones saturated	"CLEARED" SSE push if recovering from a previous alert
🟠 WARNING	50%+ zones exceed `MAX_CELL`	Browser toast notification + SMS (with cooldown)
🔴 DANGER	ALL zones exceed `MAX_CELL` simultaneously	Browser toast + red video banner + SMS (with cooldown)
---
🌐 API Endpoints
Route	Method	Auth	Description
`/`	GET	✅	Main dashboard
`/login`	GET/POST	❌	Login page
`/logout`	GET	✅	Clears session, redirects to login
`/video_feed`	GET	✅	MJPEG video stream with heatmap overlay
`/state`	GET	✅	JSON: count, status, heatmap_data, alerts
`/analytics`	GET	✅	JSON: last-hour timeline, peak, alert count
`/stream`	GET	✅	SSE stream for real-time browser notifications
---
🗃️ Database Schema
`crowd_log` — snapshot saved every ~2 seconds
```sql
id              INTEGER PRIMARY KEY AUTOINCREMENT
ts              TEXT     -- "YYYY-MM-DD HH:MM:SS"
count           INTEGER  -- total people detected
status          TEXT     -- "SAFE" | "WARNING" | "DANGER"
max_zone        INTEGER  -- highest count in any single cell
saturated_cells INTEGER  -- number of cells over MAX_CELL
```
`alerts` — one row per status-change event
```sql
id              INTEGER PRIMARY KEY AUTOINCREMENT
ts              TEXT
status          TEXT
count           INTEGER
saturated_cells INTEGER
```
---
⚠️ Known Limitations
YOLOv8n accuracy — the nano model may miss heavily overlapping people in very dense crowds; upgrade to `yolov8m.pt` for better accuracy
SMS requires credits — Fast2SMS account must have sufficient balance
No HTTPS — use an Nginx reverse proxy with SSL for public-facing deployment
SSE reconnect — if the connection drops, the browser tab must be refreshed to reconnect
Single video stream — multiple browser tabs share the same frame generator
---
🔮 Future Improvements
[ ] Multi-camera support with switchable feeds
[ ] Upgrade to YOLOv8m/l for higher accuracy in dense scenes
[ ] Email alert support (SMTP)
[ ] User management UI — add/remove users without editing code
[ ] HTTPS support
[ ] Export crowd_log as CSV from the dashboard
[ ] Mobile-responsive layout
[ ] Configurable grid size from the UI
---
📸 Screenshots
> Add the following to a `/screenshots` folder and link here:
> - `dashboard.png` — main dashboard with spatial heatmap
> - `danger_alert.png` — all-zones-saturated DANGER state with red banner
> - `toast_notification.png` — browser toast push notification
> - `login.png` — login page
> - `analytics_tab.png` — analytics tab with Chart.js bar chart
---
👥 Credits
Detection — Ultralytics YOLOv8
SMS — Fast2SMS
Charts — Chart.js
Video processing — OpenCV
---
📄 License
This project is licensed under the MIT License.
