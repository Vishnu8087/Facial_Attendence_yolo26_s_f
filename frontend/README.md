# FaceTrack — Frontend Dashboard

Production-grade, multi-view dashboard for the Facial Attendance & Motion
Tracking System. Pure **vanilla JS + modern CSS** — no React/Vue, no build
step. Served as static files by FastAPI (or any static host).

## Files

```
frontend/
├── index.html              SPA shell with Live, Attendance Log, People, Config views
├── css/styles.css          Dark theme · segmented controls · toasts · responsive
├── js/
│   ├── app.js              WebSocket manager, frame capture, toggles, tabs, feed
│   ├── api.js              REST API client (fetch wrapper + auth headers)
│   └── charts.js           Chart.js init for the Config analytics panel
├── assets/
│   ├── favicon.svg
│   └── logo.svg
└── README.md               This file
```

## Quick start

1. **Serve the backend** — FastAPI mounts this directory at `/` (or `/static`).
   The dashboard auto-derives the API and WebSocket URLs from `window.location`.
2. Open `http://localhost:8000/` in a modern browser (Chrome/Edge/Firefox).
3. Use the **Input Source** toggle to pick *Live Camera* or *Upload Video*.
4. Pick a **Strategy Mode** (Both / Strategy 1 / Strategy 2).
5. In **Config**, select a detector and set the stream options. Open **Live**
   and click **Start Stream** — the browser targets up to 30 FPS and allows
   at most two frames in flight. Actual
   throughput depends on detector processing speed and the server's 30 FPS cap.
6. Use **Live** for the video and recent events, **Attendance Log** for
   searchable/exportable records, and **People** to enroll or remove employees.
   In People, select several clear photos of the same person; the backend
   averages their face embeddings for enrollment (combined image size limit: 32 MB).

> **Tip:** Camera access requires **HTTPS or localhost**. The browser will
> prompt for permission the first time.

## Two mandatory toggles

### 1. Input Source Toggle
- **Live Camera** — uses `navigator.mediaDevices.getUserMedia`. A device picker
  appears after the first permission grant. Click *Start Camera* to begin.
- **Upload Video** — pick an MP4/WebM/MOV file. Frames are extracted from a
  hidden `<video>` element at a target of up to 30 FPS and sent as JPEG blobs. The next
  frame waits for backend processing to finish, so CPU-heavy processing
  naturally reduces the actual rate without building a backlog. Playback
  controls (play/pause/seek) appear below the preview. When the video ends,
  capture pauses automatically and a "Video Ended" badge shows.

Both modes stream binary JPEG frames to the backend via WebSocket using
`canvas.toBlob(blob => ws.send(blob), 'image/jpeg', 0.7)`.

### 2. Strategy Mode Toggle
| Mode | Strategy | Backend pipeline |
|---|---|---|
| **Both** (default) | full | detect → align → track → liveness → embed → match → log |
| **Strategy 1** | detection + embedding only | re-detect every frame, match DB immediately, skip tracker/liveness |
| **Strategy 2** | tracking + liveness only | track + anti-spoof, skip identity extraction, do NOT log attendance |

Live-switchable — no stream restart required. The dashboard sends
`{type: "config", strategy: "both" | "detection_only" | "tracking_only"}`
immediately on click. If the socket is down, the message is queued and flushed
on reconnect.

## WebSocket protocol

**Client → server**
- `{type: "config", strategy: "..."}` — switch strategy live
- `{type: "config", threshold: 0.68}` — similarity threshold (0.5–0.9)
- `{type: "config", liveness: true|false}` — toggle anti-spoof
- `{type: "source", mode: "upload" | "camera"}` — informational source change
- Binary JPEG `Blob` at a target of up to 30 FPS — `canvas.toBlob(..., 'image/jpeg', 0.7)`;
  the client waits for the corresponding server frame/error response before
  sending another

**Server → client**
- `{type: "frame", image: "...", detections: [...], latency_ms: number}` — annotated
  image is displayed in the `<img>` overlay and per-frame detection stats are updated
- `{type: "attendance", employee_id, name, timestamp, track_id, similarity, liveness_passed, strategy}` — appended to feed + toast
- `{type: "stats", fps, faces_detected, identities_recognized, spoof_rejected, current_strategy}` — updates stat cards
- `{type: "error", message: "..."}` — error toast

## REST endpoints used

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Health probe |
| GET | `/api/employees` | List enrolled employees |
| POST | `/api/employees` | Enroll (multipart form with one or more face images) |
| DELETE | `/api/employees/{id}` | Remove employee |
| GET | `/api/attendance?from=&to=&employee=&page=&page_size=` | Paginated log |
| GET | `/api/attendance/export?...` | CSV export |
| GET | `/api/stats?range=7d|30d|90d` | Aggregates for charts |

Use **Admin Access** to provide credentials configured by the server
administrator. API keys are stored in `sessionStorage` for the current tab and
sent as `X-API-Key`; JWTs are sent as bearer tokens and must contain
`is_admin: true` for write operations. The dashboard cannot create or elevate
admin accounts.

## Features

- **Auto-reconnect** WebSocket with exponential backoff (1s → 2s → 4s → 8s → 16s → 30s cap)
- **Frame pacing** targets up to 30 FPS, with at most two frames in flight per WebSocket
  connection; server rate cap and model latency determine actual CPU/GPU throughput
- **Connection status pill** with pulsing animation (green/red/orange)
- **Toast notifications** — stackable, color-coded, auto-dismiss after 4s
- **Lazy page loading** — Attendance Log, People, and Config fetch data on first view
- **CSV export** of filtered attendance records
- **Charts** (Chart.js via CDN): daily attendance (line), hourly distribution (bar), department breakdown (doughnut)
- **Responsive** — desktop 1280px+, tablet 768px+, mobile degrades gracefully
- **Accessible** — ARIA labels, keyboard-navigable tabs (←/→), focus rings, semantic markup

## Browser support

Latest Chrome, Edge, Firefox, or Safari. Requires `getUserMedia` (HTTPS or
localhost) for camera mode.
