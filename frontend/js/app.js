/**
 * @file app.js
 * @description Main controller for the FaceTrack Attendance System dashboard.
 *   Owns:
 *     · WebSocket connection (auto-reconnect w/ exponential backoff, cap 30s)
 *     · Frame capture loop (requestAnimationFrame, capped at 30 FPS,
 *       JPEG blob → ws.send)
 *     · Input source toggle (Live Camera via getUserMedia / Upload Video)
 *     · Strategy mode toggle (both | detection_only | tracking_only)
 *     · Threshold slider + Liveness toggle (config messages)
 *     · Attendance feed renderer + stat cards updater
 *     · Toast notifications + tab switching
 *     · Attendance log table (filters, pagination, CSV export)
 *     · Employee enroll/list/delete
 *     · Stats tab chart rendering (lazy)
 *
 *   No build step. Pure vanilla JS. Exposes nothing on window except for
 *   debugging helpers under `window.facetrack`.
 */

(function (global) {
  'use strict';

  // =========================================================================
  // Constants & config
  // =========================================================================

  /** WebSocket endpoint. Derives from current origin. */
  const WS_URL = (function () {
    if (global.location.protocol === 'file:') {
      return 'ws://127.0.0.1:8000/ws/stream';
    }
    const proto = global.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${global.location.host}/ws/stream`;
  })();

  /** Request up to 30 FPS; actual throughput depends on camera and inference speed. */
  const TARGET_FPS = 30;
  const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;

  /** Capture canvas dimensions (sent to backend). */
  const CAPTURE_W = 640;
  const CAPTURE_H = 480;

  /** JPEG quality for captured frames. */
  const JPEG_QUALITY = 0.7;

  /** Reconnect backoff caps. */
  const RECONNECT_INITIAL_MS = 1000;
  const RECONNECT_MAX_MS = 30000;

  /** Toast auto-dismiss. */
  const TOAST_TIMEOUT_MS = 4000;

  /** Attendance feed item cap (FIFO). */
  const FEED_MAX_ITEMS = 200;

  /** Strategy metadata for UI hints. */
  const STRATEGY_META = {
    both: {
      label: 'Both',
      hint: 'Both — full pipeline (detect → embed → track → liveness → match → log).',
    },
    detection_only: {
      label: 'Strategy 1',
      hint: 'Strategy 1 — Detection + Embedding only. Re-detect every frame, match against DB immediately. Skips tracker + liveness.',
    },
    tracking_only: {
      label: 'Strategy 2',
      hint: 'Strategy 2 — Tracking + Liveness only. Track faces across frames and run anti-spoof. Skips identity extraction; does NOT log attendance.',
    },
  };

  // =========================================================================
  // Tiny DOM helpers
  // =========================================================================

  /** querySelector shorthand. */
  const qs = (sel, root = document) => root.querySelector(sel);
  /** querySelectorAll → Array shorthand. */
  const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  /** Create element with attributes + children. */
  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    Object.keys(attrs).forEach((k) => {
      if (k === 'class') node.className = attrs[k];
      else if (k === 'dataset') Object.assign(node.dataset, attrs[k]);
      else if (k.startsWith('on') && typeof attrs[k] === 'function') {
        node.addEventListener(k.slice(2).toLowerCase(), attrs[k]);
      } else if (k === 'html') node.innerHTML = attrs[k];
      else if (attrs[k] != null) node.setAttribute(k, attrs[k]);
    });
    (Array.isArray(children) ? children : [children]).forEach((c) => {
      if (c == null) return;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return node;
  }

  /** Format seconds as M:SS. */
  function fmtTime(sec) {
    if (!isFinite(sec) || sec < 0) sec = 0;
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  /** Format ISO timestamp for display. */
  function fmtTimestamp(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    if (isNaN(d.getTime())) return String(ts);
    return d.toLocaleString(undefined, {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  }

  /** Format time-only (HH:MM:SS). */
  function fmtTimeOnly(ts) {
    const d = new Date(ts);
    if (isNaN(d.getTime())) return String(ts);
    return d.toLocaleTimeString(undefined, { hour12: false });
  }

  /** Debounce a function. */
  function debounce(fn, ms) {
    let t = null;
    return function (...args) {
      clearTimeout(t);
      t = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  /** Get initials for avatar. */
  function initials(name) {
    if (!name) return '?';
    return name.trim().split(/\s+/).slice(0, 2).map((w) => w[0] || '').join('').toUpperCase();
  }

  /** Escape text for safe innerHTML insertion. */
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  /** Trigger a CSV download in the browser. */
  function downloadCsv(filename, csvText) {
    const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = el('a', { href: url, download: filename });
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // =========================================================================
  // Toast notification system
  // =========================================================================

  const Toast = {
    _container: null,
    _init() {
      if (this._container) return;
      this._container = qs('#toastContainer');
    },

    /**
     * Show a toast.
     * @param {string} title
     * @param {string} [msg]
     * @param {'success'|'error'|'info'|'warning'} [kind='info']
     * @param {number} [timeout=4000]
     */
    show(title, msg = '', kind = 'info', timeout = TOAST_TIMEOUT_MS) {
      this._init();
      const icons = {
        success: '<path d="M20 6L9 17l-5-5"/>',
        error:   '<path d="M18 6L6 18M6 6l12 12"/>',
        info:    '<circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>',
        warning: '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4M12 17h.01"/>',
      };
      const node = el('div', { class: `toast toast--${kind}`, role: 'alert' }, [
        el('span', {
          class: 'toast__icon',
          html: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">${icons[kind] || icons.info}</svg>`,
        }),
        el('div', { class: 'toast__body' }, [
          el('p', { class: 'toast__title' }, title),
          msg ? el('p', { class: 'toast__msg' }, msg) : null,
        ]),
        el('button', {
          class: 'toast__close',
          'aria-label': 'Dismiss notification',
          onclick: () => this._dismiss(node),
        }, [
          el('span', { html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14" aria-hidden="true"><path d="M18 6L6 18M6 6l12 12"/></svg>' }),
        ]),
      ]);
      this._container.appendChild(node);
      if (timeout > 0) {
        setTimeout(() => this._dismiss(node), timeout);
      }
      return node;
    },

    _dismiss(node) {
      if (!node || node.classList.contains('is-leaving')) return;
      node.classList.add('is-leaving');
      setTimeout(() => node.remove(), 240);
    },

    success(t, m) { return this.show(t, m, 'success'); },
    error(t, m)   { return this.show(t, m, 'error', 6000); },
    info(t, m)    { return this.show(t, m, 'info'); },
    warning(t, m) { return this.show(t, m, 'warning', 5000); },
  };

  // =========================================================================
  // Connection status UI
  // =========================================================================

  const ConnStatus = {
    _pill: null,
    _badge: null,
    init() {
      this._pill = qs('#connStatus');
    },
    set(state, text) {
      if (!this._pill) this.init();
      this._pill.classList.remove('connected', 'disconnected', 'connecting');
      this._pill.classList.add(state);
      qs('.text', this._pill).textContent = text;
    },
  };

  // =========================================================================
  // WebSocket manager (auto-reconnect, exponential backoff capped at 30s)
  // =========================================================================

  class WebSocketManager {
    constructor(url) {
      this.url = url;
      this.ws = null;
      this._shouldReconnect = false;
      this._backoffMs = RECONNECT_INITIAL_MS;
      this._reconnectTimer = null;
      this._pendingConfig = null; // queued config while disconnected
      this._lastCloseInfo = '';
      this._inFlightFrames = 0;
      this._maxInFlightFrames = 2;
      this.onOpen = null;
      this.onClose = null;
      this.onMessage = null;
      this.onError = null;
    }

    /** Open the connection. Auto-reconnects on unexpected close. */
    connect() {
      if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
        return;
      }
      this._shouldReconnect = true;
      ConnStatus.set('connecting', 'Connecting…');
      try {
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = 'arraybuffer';
      } catch (err) {
        Toast.error('WebSocket error', `Could not open connection: ${err.message}`);
        this._scheduleReconnect();
        return;
      }

      this.ws.onopen = () => {
        this._backoffMs = RECONNECT_INITIAL_MS;
        ConnStatus.set('connected', 'Connected');
        if (typeof this.onOpen === 'function') this.onOpen();
        // Flush queued config
        if (this._pendingConfig) {
          this.sendJson(this._pendingConfig);
          this._pendingConfig = null;
        }
      };

      this.ws.onmessage = (evt) => {
        if (typeof this.onMessage === 'function') {
          try {
            const data = typeof evt.data === 'string' ? JSON.parse(evt.data) : evt.data;
            if (
              data &&
              (data.type === 'frame' ||
                data.type === 'frame_dropped' ||
                data.type === 'error')
            ) {
              this._inFlightFrames = Math.max(0, this._inFlightFrames - 1);
            }
            this.onMessage(data);
          } catch (err) {
            console.warn('[ws] could not parse message:', err);
          }
        }
      };

      this.ws.onerror = (evt) => {
        if (typeof this.onError === 'function') this.onError(evt);
      };

      this.ws.onclose = (evt) => {
        this._inFlightFrames = 0;
        ConnStatus.set('disconnected', 'Disconnected');
        const closeReason = evt.reason ? `: ${evt.reason}` : '';
        this._lastCloseInfo = `Connection closed (${evt.code}${closeReason})`;
        if (typeof this.onClose === 'function') this.onClose(evt);
        if (this._shouldReconnect) this._scheduleReconnect();
      };
    }

    /** Schedule a reconnect attempt with exponential backoff (capped). */
    _scheduleReconnect() {
      if (this._reconnectTimer) return;
      const delay = this._backoffMs;
      this._backoffMs = Math.min(this._backoffMs * 2, RECONNECT_MAX_MS);
      const closeInfo = this._lastCloseInfo ? ` — ${this._lastCloseInfo}` : '';
      ConnStatus.set(
        'connecting',
        `Reconnecting in ${(delay / 1000).toFixed(0)}s…${closeInfo}`
      );
      this._reconnectTimer = setTimeout(() => {
        this._reconnectTimer = null;
        this.connect();
      }, delay);
    }

    /** Send a JSON message. Queues config if socket not open. */
    sendJson(obj) {
      if (obj && obj.type === 'config') {
        // Always remember the latest config so it can be re-sent after reconnect.
        this._pendingConfig = { ...this._pendingConfig, ...obj };
      }
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        try {
          this.ws.send(JSON.stringify(obj));
          return true;
        } catch (err) {
          console.warn('[ws] send failed:', err);
          return false;
        }
      }
      return false;
    }

    /** Send a binary Blob (JPEG frame). */
    sendBlob(blob) {
      if (this._inFlightFrames >= this._maxInFlightFrames) return false;
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        try {
          this.ws.send(blob);
          this._inFlightFrames += 1;
          return true;
        } catch (err) {
          console.warn('[ws] blob send failed:', err);
        }
      }
      return false;
    }

    /** Close and stop reconnecting. */
    disconnect() {
      this._shouldReconnect = false;
      if (this._reconnectTimer) {
        clearTimeout(this._reconnectTimer);
        this._reconnectTimer = null;
      }
      if (this.ws) {
        try { this.ws.close(); } catch (_) {}
      }
    }

    get isOpen() {
      return this.ws && this.ws.readyState === WebSocket.OPEN;
    }
  }

  // =========================================================================
  // Frame capture loop (requestAnimationFrame, throttled to the 30 FPS target)
  // =========================================================================

  class FrameCaptureLoop {
    /**
     * @param {HTMLVideoElement} videoEl
     * @param {HTMLCanvasElement} canvasEl
     * @param {(blob: Blob) => void} onFrame  Called with JPEG blob.
     */
    constructor(videoEl, canvasEl, onFrame) {
      this.video = videoEl;
      this.canvas = canvasEl;
      this.ctx = canvasEl.getContext('2d', { alpha: false });
      this.onFrame = onFrame;
      this._rafId = null;
      this._lastCapture = 0;
      this._lastFrameSent = 0;
      this._capturePending = false;
      this._running = false;
    }

    start() {
      if (this._running) return;
      this._running = true;
      this._lastCapture = 0;
      this._lastFrameSent = 0;
      this._loop();
    }

    stop() {
      this._running = false;
      if (this._rafId) cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }

    get isRunning() { return this._running; }

    _loop() {
      if (!this._running) return;
      this._rafId = requestAnimationFrame(() => this._loop());

      const now = performance.now();
      if (this._capturePending || now - this._lastCapture < FRAME_INTERVAL_MS) return;
      this._lastCapture = now;

      // Skip if video not ready
      if (!this.video || this.video.readyState < 2) return;
      if (this.video.videoWidth === 0 || this.video.videoHeight === 0) return;
      // Pause capture if video is paused/ended
      if (this.video.paused || this.video.ended) return;

      // Ensure canvas matches capture size
      if (this.canvas.width !== CAPTURE_W || this.canvas.height !== CAPTURE_H) {
        this.canvas.width = CAPTURE_W;
        this.canvas.height = CAPTURE_H;
      }

      try {
        this.ctx.drawImage(this.video, 0, 0, CAPTURE_W, CAPTURE_H);
      } catch (err) {
        // drawImage can throw if the video frame is not yet decoded; skip silently.
        return;
      }

      try {
        this._capturePending = true;
        this.canvas.toBlob((blob) => {
          try {
            const sentAt = performance.now();
            if (
              blob &&
              this._running &&
              sentAt - this._lastFrameSent >= FRAME_INTERVAL_MS
            ) {
              if (this.onFrame(blob) !== false) this._lastFrameSent = sentAt;
            }
          } finally {
            this._capturePending = false;
          }
        }, 'image/jpeg', JPEG_QUALITY);
      } catch (err) {
        this._capturePending = false;
        console.warn('[capture] toBlob failed:', err);
      }
    }
  }

  // =========================================================================
  // Camera manager (getUserMedia)
  // =========================================================================

  class CameraManager {
    constructor(videoEl) {
      this.video = videoEl;
      this.stream = null;
      this.currentDeviceId = '';
    }

    /** Enumerate video input devices (requires permission for labels). */
    async listDevices() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        return devices.filter((d) => d.kind === 'videoinput');
      } catch (err) {
        console.warn('[camera] enumerateDevices failed:', err);
        return [];
      }
    }

    /**
     * Start the camera.
     * @param {string} [deviceId]  Specific device to use (optional).
     * @returns {Promise<MediaStream>}
     * @throws {Error} With a friendly message on permission denial.
     */
    async start(deviceId = '') {
      if (this.stream) await this.stop();

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera API is not available in this browser. Please use a modern browser over HTTPS or localhost.');
      }

      const constraints = {
        audio: false,
        video: deviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: CAPTURE_W }, height: { ideal: CAPTURE_H } }
          : { width: { ideal: CAPTURE_W }, height: { ideal: CAPTURE_H }, facingMode: 'user' },
      };

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (err) {
        if (err.name === 'NotAllowededError' || err.name === 'PermissionDeniedError') {
          throw new Error('Camera permission denied. Please allow camera access in your browser settings and try again.');
        }
        if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          throw new Error('No camera device was found on this machine.');
        }
        if (err.name === 'NotReadableError') {
          throw new Error('Camera is already in use by another application. Close it and try again.');
        }
        throw new Error(`Could not start camera: ${err.message || err.name}`);
      }

      this.stream = stream;
      this.video.srcObject = stream;
      this.currentDeviceId = deviceId;
      await this.video.play().catch(() => {});
      return stream;
    }

    /** Stop the camera and release the device. */
    async stop() {
      if (this.stream) {
        this.stream.getTracks().forEach((t) => t.stop());
        this.stream = null;
      }
      this.video.srcObject = null;
    }

    get isRunning() { return !!this.stream; }
  }

  // =========================================================================
  // Video upload manager (file → object URL → video element)
  // =========================================================================

  class VideoUploadManager {
    constructor(videoEl) {
      this.video = videoEl;
      this.objectUrl = null;
      this.file = null;
    }

    /** Load a File into the video element via an object URL. */
    async load(file) {
      if (!file) return;
      if (!file.type.startsWith('video/')) {
        throw new Error('Selected file is not a video. Please choose an MP4, WebM, or MOV file.');
      }
      this._cleanup();
      this.file = file;
      this.objectUrl = URL.createObjectURL(file);
      this.video.srcObject = null;
      this.video.src = this.objectUrl;
      this.video.loop = false; // we want "video ended" status, not auto-loop
      await this.video.load();
      // Attempt to play (may require user gesture — handled by play/pause control)
      try { await this.video.play(); } catch (_) {}
    }

    /** Stop playback and release object URL. */
    _cleanup() {
      if (this.objectUrl) {
        URL.revokeObjectURL(this.objectUrl);
        this.objectUrl = null;
      }
      this.video.removeAttribute('src');
      try { this.video.load(); } catch (_) {}
    }

    async stop() {
      this._cleanup();
      this.file = null;
    }

    get isReady() { return !!this.objectUrl; }
  }

  // =========================================================================
  // Attendance feed renderer
  // =========================================================================

  const Feed = {
    _list: null,
    _items: [],
    init() {
      this._list = qs('#feedList');
      qs('#btnClearFeed').addEventListener('click', () => this.clear());
    },
    /** Append an attendance event from a backend message. */
    push(ev) {
      if (!this._list) this.init();
      const li = el('li', { class: 'feed-item' }, [
        el('div', { class: 'feed-item__avatar', 'aria-hidden': 'true' }, [initials(ev.name)]),
        el('div', { class: 'feed-item__main' }, [
          el('div', { class: 'feed-item__name' }, [esc(ev.name || 'Unknown')]),
          el('div', { class: 'feed-item__meta' }, [
            el('span', { class: 'id' }, [fmtTimeOnly(ev.timestamp)]),
            ev.employee_id ? el('span', { class: 'text-muted' }, ['· ' + esc(ev.employee_id)]) : null,
          ]),
        ]),
        el('div', { class: 'feed-item__badges' }, [
          el('span', { class: 'badge badge--sim' }, [`sim ${Number(ev.similarity ?? ev.score ?? 0).toFixed(2)}`]),
          el('div', { class: 'row', style: 'gap:4px;' }, [
            ev.track_id != null ? el('span', { class: 'badge badge--track' }, [`#${ev.track_id}`]) : null,
            ev.liveness_passed === true
              ? el('span', { class: 'badge badge--live-pass', title: 'Liveness passed' }, ['live ✓'])
              : ev.liveness_passed === false
                ? el('span', { class: 'badge badge--live-fail', title: 'Liveness failed' }, ['spoof ✗'])
                : null,
          ]),
        ]),
      ]);
      this._list.prepend(li);
      this._items.unshift(li);
      if (this._items.length > FEED_MAX_ITEMS) {
        const removed = this._items.pop();
        removed.remove();
      }
    },
    clear() {
      if (!this._list) return;
      this._list.innerHTML = '';
      this._items = [];
    },
  };

  // =========================================================================
  // Stats panel (cards)
  // =========================================================================

  const Stats = {
    _els: {},
    init() {
      this._els = {
        fps: qs('#statFps'),
        faces: qs('#statFaces'),
        recog: qs('#statRecog'),
        spoof: qs('#statSpoof'),
      };
    },
    /** Update from a backend {type:"stats"} message. */
    update(s) {
      if (!this._els.fps) this.init();
      if (s.fps != null) this._els.fps.textContent = Number(s.fps).toFixed(1);
      if (s.faces_detected != null) this._els.faces.textContent = s.faces_detected;
      if (s.identities_recognized != null) this._els.recog.textContent = s.identities_recognized;
      if (s.spoof_rejected != null) this._els.spoof.textContent = s.spoof_rejected;
    },
  };

  // =========================================================================
  // Tabs manager
  // =========================================================================

  const Tabs = {
    _loaded: new Set(),
    init() {
      qsa('.tab').forEach((tab) => {
        tab.addEventListener('click', () => this.activate(tab));
        tab.addEventListener('keydown', (e) => {
          if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
          e.preventDefault();
          const tabs = qsa('.tab');
          const i = tabs.indexOf(tab);
          const next = e.key === 'ArrowRight' ? (i + 1) % tabs.length : (i - 1 + tabs.length) % tabs.length;
          tabs[next].focus();
          this.activate(tabs[next]);
        });
      });
    },
    activate(tab) {
      const id = tab.id;
      qsa('.tab').forEach((t) => {
        const selected = t === tab;
        t.setAttribute('aria-selected', selected ? 'true' : 'false');
      });
      qsa('.tab-panel').forEach((p) => p.classList.toggle('active', p.id === tab.getAttribute('aria-controls')));
      document.body.dataset.page = id.replace('tab-', '');
      // Lazy-load handlers
      const key = id;
      if (!this._loaded.has(key)) {
        this._loaded.add(key);
        if (id === 'tab-log') AttendanceLog.refresh();
        if (id === 'tab-emp') Employees.refresh();
        if (id === 'tab-config') StatsTab.refresh();
      }
    },
    show(name) {
      const map = { live: 'tab-live', log: 'tab-log', emp: 'tab-emp', config: 'tab-config' };
      const tab = qs('#' + map[name]);
      if (tab) this.activate(tab);
    },
  };

  // =========================================================================
  // Attendance log table (filters, pagination, CSV export)
  // =========================================================================

  const AttendanceLog = {
    _page: 1,
    _pageSize: 25,
    _total: 0,
    _lastFilters: {},
    init() {
      qs('#btnApplyFilters').addEventListener('click', () => this.refresh(1));
      qs('#btnResetFilters').addEventListener('click', () => this.reset());
      qs('#btnExportCsv').addEventListener('click', () => this.exportCsv());
      qs('#btnPrevPage').addEventListener('click', () => this.refresh(this._page - 1));
      qs('#btnNextPage').addEventListener('click', () => this.refresh(this._page + 1));
    },
    _readFilters() {
      return {
        from: qs('#filterFrom').value || '',
        to: qs('#filterTo').value || '',
        employee: qs('#filterEmployee').value.trim() || '',
        page: this._page,
        page_size: this._pageSize,
      };
    },
    async refresh(page) {
      if (page) this._page = page;
      const filters = this._readFilters();
      this._lastFilters = filters;
      const tbody = qs('#logBody');
      const empty = qs('#logEmpty');
      tbody.innerHTML = '';
      empty.textContent = 'Loading…';
      empty.style.display = 'block';
      try {
        const { items, total, page_size } = await global.apiClient.listAttendance(filters);
        this._total = total || 0;
        this._pageSize = page_size || this._pageSize;
        if (!items.length) {
          empty.textContent = 'No attendance records match the current filters.';
          this._renderPager();
          return;
        }
        empty.style.display = 'none';
        items.forEach((r) => {
          tbody.appendChild(el('tr', {}, [
            el('td', {}, [fmtTimestamp(r.timestamp || r.created_at)]),
            el('td', {}, [esc(r.name || r.employee_name || '—')]),
            el('td', { class: 'text-mono' }, [esc(r.employee_id || '—')]),
            el('td', {}, [esc(r.department || '—')]),
            el('td', {}, [r.similarity != null ? Number(r.similarity).toFixed(3) : '—']),
            el('td', { class: 'text-mono' }, [r.track_id != null ? `#${r.track_id}` : '—']),
            el('td', {}, [
              r.liveness_passed === true ? '✓ pass'
              : r.liveness_passed === false ? '✗ spoof'
              : '—',
            ]),
            el('td', {}, [esc(r.strategy || '—')]),
          ]));
        });
        this._renderPager();
      } catch (err) {
        empty.textContent = `Failed to load: ${err.message}`;
        Toast.error('Could not load attendance log', err.message);
      }
    },
    _renderPager() {
      const totalPages = Math.max(1, Math.ceil(this._total / this._pageSize));
      qs('#logPageInfo').textContent = `Page ${this._page} of ${totalPages} · ${this._total} records`;
      qs('#btnPrevPage').disabled = this._page <= 1;
      qs('#btnNextPage').disabled = this._page >= totalPages;
    },
    reset() {
      qs('#filterFrom').value = '';
      qs('#filterTo').value = '';
      qs('#filterEmployee').value = '';
      this.refresh(1);
    },
    async exportCsv() {
      try {
        Toast.info('Exporting…', 'Generating CSV for current filters.');
        const filters = { ...this._lastFilters };
        delete filters.page; delete filters.page_size;
        const csv = await global.apiClient.exportAttendance(filters);
        const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
        downloadCsv(`attendance-${stamp}.csv`, csv);
        Toast.success('Export ready', 'CSV downloaded.');
      } catch (err) {
        Toast.error('CSV export failed', err.message);
      }
    },
  };

  // =========================================================================
  // Employees panel (enroll + list + delete)
  // =========================================================================

  const Employees = {
    init() {
      const form = qs('#enrollForm');
      const fileInput = qs('#enrollImage');
      const preview = qs('#enrollPreview');

      fileInput.addEventListener('change', () => {
        preview.replaceChildren();
        const files = Array.from(fileInput.files || []);
        if (!files.length) {
          preview.textContent = 'No images selected';
          preview.classList.remove('preview--images');
          return;
        }
        preview.classList.add('preview--images');
        files.forEach((file, index) => {
          const url = URL.createObjectURL(file);
          const image = el('img', { src: url, alt: `Face preview ${index + 1}` });
          image.addEventListener('load', () => URL.revokeObjectURL(url), { once: true });
          preview.appendChild(image);
        });
      });

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = qs('#btnEnroll');
        const txt = qs('.btn-text', btn);
        const orig = txt.textContent;
        btn.disabled = true;
        txt.innerHTML = '<span class="spinner"></span> Enrolling…';
        try {
          const fd = new FormData(form);
          const result = await global.apiClient.enrollEmployee(fd);
          Toast.success('Employee enrolled', result.name || 'New employee added successfully.');
          form.reset();
          preview.textContent = 'No images selected';
          preview.classList.remove('preview--images');
          this.refresh();
        } catch (err) {
          const message = err.status === 403 && err.message === 'Admin privileges required'
            ? 'This credential is valid but is not an admin. Use Admin Access with an admin API key or an admin JWT.'
            : err.status === 403
              ? `The server returned HTTP 403: ${err.message}. Check the server or proxy access policy.`
              : err.message;
          Toast.error('Enrollment failed', message);
        } finally {
          btn.disabled = false;
          txt.textContent = orig;
        }
      });

      qs('#btnRefreshEmployees').addEventListener('click', () => this.refresh());
      qs('#empBody').addEventListener('click', async (e) => {
        const btn = e.target.closest('[data-action="delete"]');
        if (!btn) return;
        const id = btn.dataset.id;
        if (!confirm('Delete this employee? This cannot be undone.')) return;
        try {
          await global.apiClient.deleteEmployee(id);
          Toast.success('Deleted', `Employee ${id} removed.`);
          this.refresh();
        } catch (err) {
          Toast.error('Delete failed', err.message);
        }
      });
    },

    async refresh() {
      const tbody = qs('#empBody');
      const empty = qs('#empEmpty');
      tbody.innerHTML = '';
      empty.textContent = 'Loading…';
      empty.style.display = 'block';
      try {
        const list = await global.apiClient.listEmployees();
        if (!list.length) {
          empty.textContent = 'No employees enrolled yet.';
          return;
        }
        empty.style.display = 'none';
        list.forEach((emp) => {
          const photo = emp.photo_url || emp.image_url || emp.avatar_url || '';
          tbody.appendChild(el('tr', {}, [
            el('td', {}, [
              photo
                ? el('img', { src: photo, alt: '', style: 'width:32px;height:32px;border-radius:50%;object-fit:cover;' })
                : el('div', { style: 'width:32px;height:32px;border-radius:50%;background:#1c2748;display:flex;align-items:center;justify-content:center;font-size:11px;' }, [initials(emp.name)]),
            ]),
            el('td', {}, [esc(emp.name || '—')]),
            el('td', { class: 'text-mono' }, [esc(emp.employee_id || emp.id || '—')]),
            el('td', {}, [esc(emp.department || '—')]),
            el('td', {}, [esc(emp.email || '—')]),
            el('td', {}, [fmtTimestamp(emp.created_at || emp.enrolled_at)]),
            el('td', {}, [
              el('button', {
                class: 'btn btn--ghost btn--sm',
                'data-action': 'delete',
                'data-id': emp.id || emp.employee_id,
                'aria-label': `Delete ${emp.name}`,
              }, ['Delete']),
            ]),
          ]));
        });
      } catch (err) {
        empty.textContent = `Failed to load: ${err.message}`;
      }
    },
  };

  // =========================================================================
  // Stats tab (lazy chart rendering)
  // =========================================================================

  const StatsTab = {
    _inited: false,
    init() {
      qs('#btnRefreshStats').addEventListener('click', () => this.refresh());
      qs('#statsRange').addEventListener('change', () => this.refresh());
      this._inited = true;
    },
    async refresh() {
      if (!this._inited) this.init();
      const range = qs('#statsRange').value;
      try {
        if (global.chartsController) {
          global.chartsController.init();
          await global.chartsController.refresh(range);
        }
      } catch (err) {
        Toast.error('Stats load failed', err.message);
      }
    },
  };

  // =========================================================================
  // Main App controller — wires everything together
  // =========================================================================

  class App {
    constructor() {
      // DOM refs
      this.video = qs('#videoEl');
      this.canvas = qs('#captureCanvas');
      this.overlay = qs('#annotatedOverlay');
      this.placeholder = qs('#videoPlaceholder');

      // Managers
      this.ws = new WebSocketManager(WS_URL);
      this.camera = new CameraManager(this.video);
      this.upload = new VideoUploadManager(this.video);
      this.capture = new FrameCaptureLoop(this.video, this.canvas, (blob) => this.ws.sendBlob(blob));

      // State
      this.state = {
        source: 'camera',           // 'camera' | 'upload' | 'cctv'
        strategy: 'both',           // 'both' | 'detection_only' | 'tracking_only'
        threshold: 0.30,            // lowered from 0.55 — LFW benchmark showed 0% FAR + 99.33% accuracy
        liveness: true,
        streaming: false,
        detectorBackend: 'yolo26',         // DEFAULT flipped to YOLO26 after benchmark
        detectorModel: 'yolo26s-face.pt',  // best mAP/F1 on WIDER FACE val
        cctvConnected: false,
        cctvCameraId: null,
        cctvCameraName: null,
      };
    }

    /** Boot the app. Called on DOMContentLoaded. */
    init() {
      ConnStatus.init();
      Feed.init();
      Stats.init();
      Tabs.init();
      AttendanceLog.init();
      Employees.init();
      this._bindAdminAccess();

      this._bindInputSource();
      this._bindStrategy();
      this._bindDetector();        // NEW: detector dropdown handler
      this._bindStreamControls();
      this._bindThreshold();
      this._bindLiveness();
      this._bindUploadControls();
      this._bindWsHandlers();
      this._bindVideoEvents();

      // Initial UI sync
      this._syncStrategyUi();
      this._syncSourceUi();

      // Open the WebSocket (reconnects automatically)
      this.ws.connect();

      Toast.info('Welcome', 'Select an input source and start the stream to begin.');
    }

    _bindAdminAccess() {
      const dialog = qs('#adminAccessDialog');
      const form = qs('#adminAccessForm');
      const type = qs('#adminCredentialType');
      const credential = qs('#adminCredential');

      qs('#btnAdminAccess').addEventListener('click', () => dialog.showModal());
      qs('#btnCloseAdminAccess').addEventListener('click', () => dialog.close());
      type.addEventListener('change', () => {
        const isApiKey = type.value === 'api-key';
        credential.autocomplete = isApiKey ? 'new-password' : 'off';
        qs('#adminCredentialHint').textContent = isApiKey
          ? "The API key must match the server's configured API_KEY."
          : 'The JWT must contain an is_admin claim set to true.';
      });
      qs('#btnClearAdminAccess').addEventListener('click', () => {
        global.apiClient.clearAuth();
        credential.value = '';
        dialog.close();
        Toast.info('Admin access cleared', 'Saved credentials were removed from this browser.');
      });
      form.addEventListener('submit', (event) => {
        event.preventDefault();
        const value = credential.value.trim();
        if (!value) return;
        if (type.value === 'api-key') global.apiClient.setApiKey(value);
        else global.apiClient.setToken(value);
        credential.value = '';
        dialog.close();
        Toast.success('Credential saved', 'Retry enrollment. The server will verify admin access.');
      });
    }

    // -----------------------------------------------------------------------
    // Input source toggle
    // -----------------------------------------------------------------------

    _bindInputSource() {
      qs('#srcCamera').addEventListener('click', () => this._setSource('camera'));
      qs('#srcUpload').addEventListener('click', () => this._setSource('upload'));
      qs('#srcCctv').addEventListener('click', () => this._setSource('cctv'));
      // CCTV connect button
      const btnCctv = qs('#btnCctvConnect');
      if (btnCctv) {
        btnCctv.addEventListener('click', () => this._connectCctv());
      }
    }

    /**
     * Switch input source. Stops the active source's hardware (camera or upload URL).
     * Does NOT auto-start the new source — user clicks Start Camera / picks file / connects CCTV.
     * @param {'camera'|'upload'|'cctv'} source
     */
    async _setSource(source) {
      if (this.state.source === source) return;

      // Stop current capture + active source
      this.capture.stop();
      if (this.state.source === 'camera' && this.camera.isRunning) {
        await this.camera.stop();
        this._setVideoStatus('idle', 'Idle');
      }
      if (this.state.source === 'camera') {
        this._setCameraBtn(this.camera.isRunning);
      }
      if (this.state.source === 'upload' && this.upload.isReady) {
        await this.upload.stop();
        this._setVideoStatus('idle', 'Idle');
      }
      if (this.state.source === 'cctv' && this.state.cctvConnected) {
        // Disconnect CCTV on the backend
        this.ws.sendJson({ type: 'source', mode: 'cctv', action: 'disconnect' });
        this.state.cctvConnected = false;
        this._setVideoStatus('idle', 'Idle');
      }

      this.state.source = source;
      this._syncSourceUi();

      // Inform backend (informational)
      this.ws.sendJson({ type: 'source', mode: source });

      // Hide annotated overlay until new frames arrive
      this.overlay.classList.add('hidden');
      this.placeholder.classList.remove('hidden');

      const msgs = {
        camera: 'Live camera mode. Click "Start Camera" to grant access.',
        upload: 'Upload video mode. Choose a video file to stream.',
        cctv: 'CCTV / RTSP mode. Enter camera ID, name, URL and click Connect.',
      };
      Toast.info('Input source changed', msgs[source] || `${source} mode active.`);
    }

    /**
     * Connect to a CCTV / RTSP camera. Sends connection details to the backend
     * via WebSocket. The backend opens the RTSP stream server-side using OpenCV
     * and pushes frames through the same pipeline.
     */
    _connectCctv() {
      const id = (qs('#cctvId')?.value || '').trim();
      const name = (qs('#cctvName')?.value || '').trim();
      const url = (qs('#cctvUrl')?.value || '').trim();
      const username = (qs('#cctvUser')?.value || '').trim();
      const password = (qs('#cctvPass')?.value || '').trim();
      const hint = qs('#cctvHint');

      if (!id) {
        Toast.error('Missing Camera ID', 'Please enter a unique Camera ID.');
        return;
      }
      if (!name) {
        Toast.error('Missing Camera Name', 'Please enter a human-readable Camera Name.');
        return;
      }
      if (!url) {
        Toast.error('Missing Stream URL', 'Please enter the RTSP/HTTP stream URL.');
        return;
      }
      // Basic URL validation
      if (!/^(rtsp|http|https|udp|tcp):\/\//.test(url)) {
        Toast.error('Invalid URL', 'URL must start with rtsp://, http://, https://, udp://, or tcp://');
        return;
      }

      // Build credentials-embedded URL if provided (rtsp://user:pass@host/path)
      let effectiveUrl = url;
      if (username && password) {
        try {
          const u = new URL(url);
          u.username = encodeURIComponent(username);
          u.password = encodeURIComponent(password);
          effectiveUrl = u.toString();
        } catch (e) {
          // Fallback: simple string injection
          const proto = url.split('://')[0];
          const rest = url.split('://')[1];
          effectiveUrl = `${proto}://${encodeURIComponent(username)}:${encodeURIComponent(password)}@${rest}`;
        }
      }

      this.state.cctvConnected = true;
      this.state.cctvCameraId = id;
      this.state.cctvCameraName = name;

      const sent = this.ws.sendJson({
        type: 'source',
        mode: 'cctv',
        action: 'connect',
        camera: {
          id,
          name,
          url: effectiveUrl,
          // Don't send raw credentials — they're embedded in the URL
        },
      });

      if (sent) {
        Toast.success('CCTV connecting', `${name} (${id}) — backend opening stream...`);
        if (hint) hint.textContent = `Connecting to ${name} at ${url.replace(/\/\/.*@/, '//')}...`;
      } else {
        Toast.info('CCTV queued', 'Will connect after WebSocket reconnects.');
      }

      // Update button text
      const btn = qs('#btnCctvConnect');
      if (btn) {
        btn.querySelector('.btn-text').textContent = 'Connecting...';
        setTimeout(() => {
          if (btn.querySelector('.btn-text')) {
            btn.querySelector('.btn-text').textContent = 'Reconnect CCTV Camera';
          }
        }, 2000);
      }
    }

    _syncSourceUi() {
      const isCam = this.state.source === 'camera';
      const isUpload = this.state.source === 'upload';
      const isCctv = this.state.source === 'cctv';
      qs('#srcCamera').setAttribute('aria-pressed', isCam);
      qs('#srcUpload').setAttribute('aria-pressed', isUpload);
      qs('#srcCctv').setAttribute('aria-pressed', isCctv);
      qs('#cameraControls').classList.toggle('hidden', !isCam);
      qs('#uploadFileControls').classList.toggle('hidden', !isUpload);
      qs('#cctvControls').classList.toggle('hidden', !isCctv);
      // Hide playback controls unless an upload video is loaded
      qs('#uploadControls').classList.add('hidden');
    }

    // -----------------------------------------------------------------------
    // Strategy toggle (live-switchable)
    // -----------------------------------------------------------------------

    _bindStrategy() {
      qsa('#strategySegmented .segmented__btn').forEach((btn) => {
        btn.addEventListener('click', () => this._setStrategy(btn.dataset.strategy));
      });
    }

    /**
     * Switch strategy mode. Sends a config message immediately; no stream restart needed.
     * @param {'both'|'detection_only'|'tracking_only'} strategy
     */
    _setStrategy(strategy) {
      if (!STRATEGY_META[strategy]) return;
      this.state.strategy = strategy;
      this._syncStrategyUi();
      const sent = this.ws.sendJson({ type: 'config', strategy });
      if (sent) Toast.success('Strategy switched', STRATEGY_META[strategy].label + ' mode active.');
      else Toast.info('Strategy queued', 'Will apply after connection is re-established.');
    }

    _syncStrategyUi() {
      qsa('#strategySegmented .segmented__btn').forEach((btn) => {
        btn.setAttribute('aria-pressed', btn.dataset.strategy === this.state.strategy ? 'true' : 'false');
      });
      qs('#strategyHint').textContent = STRATEGY_META[this.state.strategy].hint;
      qs('#strategyBadge').textContent = `Strategy: ${STRATEGY_META[this.state.strategy].label}`;
    }

    // -----------------------------------------------------------------------
    // Detector model selector (live-switchable, just like strategy)
    // -----------------------------------------------------------------------

    _bindDetector() {
      const sel = qs('#detectorSelect');
      if (!sel) return;
      sel.addEventListener('change', () => {
        const opt = sel.options[sel.selectedIndex];
        const backend = opt.dataset.backend;
        const model = opt.dataset.model;
        if (!backend || !model) return;
        this._setDetector(backend, model);
      });
    }

    /**
     * Switch detector at runtime. Sends {type:"config", detector:{backend,model}}.
     * Backend rebuilds the detector instance for the active WebSocket connection.
     * @param {'insightface'|'yolo26'} backend
     * @param {string} model - e.g. 'yolo26n-face.pt', 'buffalo_l', 'scrfd_500m'
     */
    _setDetector(backend, model) {
      this.state.detectorBackend = backend;
      this.state.detectorModel = model;
      // Update UI badges
      const badge = qs('#detectorBadge');
      if (badge) badge.textContent = `${backend}/${model}`;
      const stats = qs('#detectorStats');
      if (stats) stats.textContent = `backend=${backend} · model=${model}`;
      const switchTime = qs('#detectorSwitchTime');
      if (switchTime) {
        const ts = new Date().toLocaleTimeString();
        switchTime.textContent = `switched ${ts}`;
      }
      // Send to backend
      const sent = this.ws.sendJson({
        type: 'config',
        detector: { backend, model },
      });
      if (sent) {
        Toast.success('Detector switched', `${backend} / ${model} will be used for the next frame.`);
      } else {
        Toast.info('Detector queued', 'Will apply after connection is re-established.');
      }
    }

    // -----------------------------------------------------------------------
    // Stream controls (Start/Stop)
    // -----------------------------------------------------------------------

    _bindStreamControls() {
      qs('#btnStartStream').addEventListener('click', () => this._startStream());
      qs('#btnStopStream').addEventListener('click', () => this._stopStream());

      // Camera button
      qs('#btnCamera').addEventListener('click', () => this._toggleCamera());

      // Device picker
      qs('#deviceSelect').addEventListener('change', async () => {
        if (this.camera.isRunning) {
          const id = qs('#deviceSelect').value;
          await this.camera.start(id);
        }
      });
    }

    /** Start streaming frames to the backend. */
    async _startStream() {
      if (this.state.streaming) return;

      // Ensure a source is active
      if (this.state.source === 'camera') {
        if (!this.camera.isRunning) {
          try {
            await this._startCamera(qs('#deviceSelect').value);
          } catch (err) {
            Toast.error('Camera unavailable', err.message);
            return;
          }
        }
      } else {
        // upload mode
        if (!this.upload.isReady) {
          Toast.error('No video loaded', 'Please choose a video file first.');
          return;
        }
        // Ensure video is playing
        if (this.video.paused) {
          try { await this.video.play(); } catch (_) {}
        }
      }

      // Make sure WS is open
      if (!this.ws.isOpen) {
        Toast.info('Connecting…', 'Establishing WebSocket connection.');
        this.ws.connect();
      }

      // Push current config so backend has the right strategy/threshold/liveness/detector
      this.ws.sendJson({ type: 'config', strategy: this.state.strategy });
      this.ws.sendJson({ type: 'config', threshold: this.state.threshold });
      this.ws.sendJson({ type: 'config', liveness: this.state.liveness });
      this.ws.sendJson({
        type: 'config',
        detector: { backend: this.state.detectorBackend, model: this.state.detectorModel },
      });

      this.state.streaming = true;
      this.capture.start();
      this._toggleStreamButtons(true);
      Toast.success(
        'Stream started',
        `Sending up to ${TARGET_FPS} FPS with at most two frames in flight in "${STRATEGY_META[this.state.strategy].label}" mode.`
      );
    }

    /** Stop streaming and release the webcam so its hardware indicator turns off. */
    async _stopStream() {
      this.capture.stop();
      this.state.streaming = false;
      this._toggleStreamButtons(false);
      const releasedCamera = this.state.source === 'camera' && this.camera.isRunning;
      if (releasedCamera) {
        await this.camera.stop();
        this._setCameraBtn(false);
        this._setVideoStatus('idle', 'Idle');
        this.overlay.classList.add('hidden');
        this.placeholder.classList.remove('hidden');
      }
      Toast.info(
        'Stream stopped',
        releasedCamera ? 'Webcam released.' : 'Frame capture paused.'
      );
    }

    _toggleStreamButtons(streaming) {
      qs('#btnStartStream').classList.toggle('hidden', streaming);
      qs('#btnStopStream').classList.toggle('hidden', !streaming);
    }

    // -----------------------------------------------------------------------
    // Camera toggle (Start/Stop Camera button)
    // -----------------------------------------------------------------------

    async _toggleCamera() {
      if (this.camera.isRunning) {
        if (this.state.streaming) {
          await this._stopStream();
          return;
        }
        await this.camera.stop();
        this._setVideoStatus('idle', 'Idle');
        this.overlay.classList.add('hidden');
        this.placeholder.classList.remove('hidden');
        this._setCameraBtn(false);
        return;
      }
      try {
        await this._startCamera(qs('#deviceSelect').value);
      } catch (err) {
        Toast.error('Camera error', err.message);
      }
    }

    async _startCamera(deviceId) {
      await this.camera.start(deviceId);
      this._setCameraBtn(true);
      this.placeholder.classList.add('hidden');
      this._setVideoStatus('live', 'Camera Live');
      // Populate device list now that we have permission
      const devices = await this.camera.listDevices();
      if (devices.length) {
        const sel = qs('#deviceSelect');
        const cur = sel.value;
        sel.innerHTML = '';
        devices.forEach((d) => {
          const label = d.label || `Camera ${d.deviceId.slice(0, 6)}`;
          sel.appendChild(el('option', { value: d.deviceId }, [label]));
        });
        if (cur) sel.value = cur;
      }
    }

    _setCameraBtn(running) {
      const btn = qs('#btnCamera');
      const txt = qs('.btn-text', btn);
      txt.textContent = running ? 'Stop Camera' : 'Start Camera';
      btn.classList.toggle('btn--primary', !running);
      btn.classList.toggle('btn--danger', running);
    }

    // -----------------------------------------------------------------------
    // Upload controls (file input + playback)
    // -----------------------------------------------------------------------

    _bindUploadControls() {
      qs('#fileInput').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        try {
          await this.upload.load(file);
          this.placeholder.classList.add('hidden');
          qs('#uploadControls').classList.remove('hidden');
          this._setVideoStatus('live', 'Video Loaded');
          qs('#uploadHint').textContent = `Loaded: ${file.name}`;
          Toast.success('Video loaded', file.name);
        } catch (err) {
          Toast.error('Could not load video', err.message);
        }
      });

      // Play/pause + seek bar
      qs('#btnPlayPause').addEventListener('click', () => this._togglePlayPause());
      const seek = qs('#seekBar');
      seek.addEventListener('input', () => {
        if (isFinite(this.video.duration)) {
          this.video.currentTime = (parseFloat(seek.value) / 100) * this.video.duration;
        }
      });
    }

    _bindVideoEvents() {
      this.video.addEventListener('loadedmetadata', () => this._updateSeekBar());
      this.video.addEventListener('timeupdate', () => this._updateSeekBar());
      this.video.addEventListener('play', () => this._updatePlayPauseIcon(true));
      this.video.addEventListener('pause', () => this._updatePlayPauseIcon(false));
      this.video.addEventListener('ended', () => {
        this._setVideoStatus('ended', 'Video Ended');
        if (this.state.streaming) {
          // Stop frame capture but keep stream button state
          this.capture.stop();
          Toast.info('Video ended', 'Frame capture paused. Restart video or load another file.');
        }
      });
    }

    _togglePlayPause() {
      if (this.video.paused) {
        this.video.play().catch((err) => Toast.error('Play failed', err.message));
        if (this.state.streaming && this.state.source === 'upload') {
          this.capture.start();
          this._setVideoStatus('live', 'Streaming');
        }
      } else {
        this.video.pause();
      }
    }

    _updatePlayPauseIcon(playing) {
      const btn = qs('#btnPlayPause');
      btn.innerHTML = playing
        ? '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
      if (playing) this._setVideoStatus('live', 'Streaming');
    }

    _updateSeekBar() {
      const seek = qs('#seekBar');
      const display = qs('#timeDisplay');
      if (isFinite(this.video.duration) && this.video.duration > 0) {
        seek.value = (this.video.currentTime / this.video.duration) * 100;
        display.textContent = `${fmtTime(this.video.currentTime)} / ${fmtTime(this.video.duration)}`;
      } else {
        display.textContent = `0:00 / 0:00`;
      }
    }

    _setVideoStatus(kind, text) {
      const badge = qs('#videoStatus');
      badge.classList.remove('live', 'ended', 'idle');
      badge.classList.add(kind);
      qs('.text', badge).textContent = text;
    }

    // -----------------------------------------------------------------------
    // Threshold + Liveness controls
    // -----------------------------------------------------------------------

    _bindThreshold() {
      const slider = qs('#thresholdSlider');
      const out = qs('#thresholdValue');
      const send = debounce((v) => {
        this.ws.sendJson({ type: 'config', threshold: v });
      }, 300);
      slider.addEventListener('input', () => {
        const v = parseFloat(slider.value);
        this.state.threshold = v;
        out.textContent = v.toFixed(2);
        send(v);
      });
    }

    _bindLiveness() {
      qs('#livenessToggle').addEventListener('change', (e) => {
        this.state.liveness = e.target.checked;
        this.ws.sendJson({ type: 'config', liveness: this.state.liveness });
        Toast.info('Liveness ' + (this.state.liveness ? 'enabled' : 'disabled'),
                   this.state.liveness ? 'Anti-spoof checks will run.' : 'Spoof detection is OFF — use with caution.');
      });
    }

    // -----------------------------------------------------------------------
    // WebSocket message handler
    // -----------------------------------------------------------------------

    _bindWsHandlers() {
      this.ws.onMessage = (msg) => this._handleMessage(msg);
      this.ws.onError = () => {
        // Errors are usually followed by close; keep this quiet to avoid spam.
      };
      this.ws.onClose = () => {
        // If streaming, pause capture until reconnected
        if (this.state.streaming) {
          // Keep capture loop running; frames will just drop silently.
        }
      };
    }

    /**
     * Dispatch a backend JSON message.
     * @param {Object} msg
     */
    _handleMessage(msg) {
      if (!msg || typeof msg !== 'object') return;
      switch (msg.type) {
        case 'frame': {
          const imageB64 = msg.image || msg.annotated_jpeg_b64;
          if (imageB64) {
            this.overlay.src = `data:image/jpeg;base64,${imageB64}`;
            this.overlay.classList.remove('hidden');
            this.placeholder.classList.add('hidden');
          }
          const detections = Array.isArray(msg.detections) ? msg.detections : [];
          const streamFps = Number(msg.fps);
          Stats.update({
            fps: Number.isFinite(streamFps) ? streamFps : null,
            faces_detected: detections.length,
            identities_recognized: detections.filter((d) => d && d.identity).length,
            spoof_rejected: detections.filter((d) => d && d.is_real === false).length,
          });
          break;
        }
        case 'attendance': {
          Feed.push(msg);
          if (msg.liveness_passed === false) {
            Toast.warning('Spoof rejected', `${msg.name || msg.employee_id || 'Unknown'} failed liveness check.`);
          } else {
            Toast.success('Attendance logged', `${msg.name || msg.employee_id} checked in`);
          }
          break;
        }
        case 'stats': {
          Stats.update(msg);
          if (msg.current_strategy && msg.current_strategy !== this.state.strategy) {
            // Backend reports a different strategy (maybe overridden); sync UI.
            this.state.strategy = msg.current_strategy;
            this._syncStrategyUi();
          }
          break;
        }
        case 'error': {
          Toast.error('Backend error', msg.message || 'Unknown error');
          break;
        }
        default: {
          // Unknown message type — log for debugging.
          console.debug('[ws] unhandled message type:', msg.type, msg);
        }
      }
    }
  }

  // =========================================================================
  // Boot
  // =========================================================================

  function boot() {
    const app = new App();
    app.init();
    // Expose for debugging in the console.
    global.facetrack = { app, ws: app.ws, api: global.apiClient };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})(window);
