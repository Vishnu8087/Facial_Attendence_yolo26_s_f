/**
 * @file api.js
 * @description REST API client for the FaceTrack Attendance System dashboard.
 *   Wraps fetch() with consistent error handling, JSON parsing, auth header
 *   injection, and query-string construction. Exposes a single global
 *   `apiClient` instance consumed by app.js.
 *
 *   Endpoints (relative to API_BASE):
 *     GET    /api/health
 *     GET    /api/employees
 *     POST   /api/employees          (multipart/form-data)
 *     DELETE /api/employees/{id}
 *     GET    /api/attendance?from=&to=&employee=&page=&page_size=
 *     GET    /api/attendance/export?...   (returns text/csv)
 *     GET    /api/stats?range=7d|30d|90d
 */

(function (global) {
  'use strict';

  /** Base URL of the backend REST API. Served by the same origin by FastAPI. */
  const API_BASE =
    (global.API_BASE_URL) ||
    (window.location.protocol === 'file:'
      ? 'http://127.0.0.1:8000/api'
      : `${window.location.origin}/api`);

  /** Default request timeout in ms. */
  const DEFAULT_TIMEOUT = 15000;
  /** Face detection and embedding can take longer on the first CPU-backed request. */
  const ENROLLMENT_TIMEOUT = 120000;

  /**
   * Build a URL-encoded query string from a filter object, skipping
   * null/undefined/empty values.
   * @param {Object} filters
   * @returns {string}
   */
  function buildQuery(filters) {
    if (!filters) return '';
    const params = new URLSearchParams();
    Object.keys(filters).forEach((key) => {
      const v = filters[key];
      if (v === null || v === undefined || v === '') return;
      params.append(key, v);
    });
    const s = params.toString();
    return s ? `?${s}` : '';
  }

  /**
   * Core fetch wrapper with timeout, auth headers, and consistent error
   * normalisation.
   * @param {string} path       Path under API_BASE (e.g. "/employees").
   * @param {Object} [opts]     fetch() options.
   * @param {number} [opts.timeout] Override default timeout (ms).
   * @returns {Promise<Response>}
   * @throws {Error} On network failure, timeout, or non-2xx response.
   */
  async function request(path, opts = {}) {
    const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
    const timeout = opts.timeout || DEFAULT_TIMEOUT;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);

    // Prefer the explicitly supplied API key; otherwise use a saved JWT.
    const apiKey = sessionStorage.getItem('facetrack_api_key');
    const token = localStorage.getItem('facetrack_token');
    const headers = new Headers(opts.headers || {});
    if (apiKey && !headers.has('X-API-Key')) {
      headers.set('X-API-Key', apiKey);
    } else if (token && !headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${token}`);
    }

    try {
      const res = await fetch(url, { ...opts, headers, signal: controller.signal });
      return res;
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error(`Request to ${path} timed out after ${timeout}ms`);
      }
      throw new Error(`Network error: ${err.message}`);
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Expect JSON response; throw an Error with backend message on non-2xx.
   * @param {string} path
   * @param {Object} opts
   * @returns {Promise<any>}
   */
  async function requestJson(path, opts) {
    const res = await request(path, opts);
    const isJson = (res.headers.get('content-type') || '').includes('application/json');
    const body = isJson ? await res.json() : await res.text();
    if (!res.ok) {
      const msg = (isJson && body && (body.detail || body.message)) || `HTTP ${res.status}`;
      const e = new Error(msg);
      e.status = res.status;
      e.body = body;
      throw e;
    }
    return body;
  }

  /**
   * ApiClient — public surface used by app.js.
   */
  class ApiClient {
    constructor() {
      this.base = API_BASE;
    }

    /** Set/override the JWT bearer token persisted in localStorage. */
    setToken(token) {
      if (token) {
        localStorage.setItem('facetrack_token', token);
        sessionStorage.removeItem('facetrack_api_key');
      } else {
        localStorage.removeItem('facetrack_token');
      }
    }

    /** Store an API key for this browser tab only. */
    setApiKey(apiKey) {
      if (apiKey) {
        sessionStorage.setItem('facetrack_api_key', apiKey);
        localStorage.removeItem('facetrack_token');
      } else {
        sessionStorage.removeItem('facetrack_api_key');
      }
    }

    /** Clear credentials saved by this dashboard. */
    clearAuth() {
      sessionStorage.removeItem('facetrack_api_key');
      localStorage.removeItem('facetrack_token');
    }

    /** Health probe. Returns {status:"ok", ...}. */
    async getHealth() {
      return requestJson('/health', { method: 'GET' });
    }

    // ----- Employees -----------------------------------------------------

    /** List enrolled employees. Returns array (or {items:[]} depending on backend). */
    async listEmployees() {
      const data = await requestJson('/employees', { method: 'GET' });
      // Normalise common shapes
      if (Array.isArray(data)) return data;
      if (data && Array.isArray(data.items)) return data.items;
      if (data && Array.isArray(data.employees)) return data.employees;
      return [];
    }

    /**
     * Enroll a new employee.
     * @param {FormData} formData  Must include: name, employee_id, and one or more image files. Optional: department, email.
     * @returns {Promise<Object>} Created employee record.
     */
    async enrollEmployee(formData) {
      return requestJson('/employees', {
        method: 'POST',
        body: formData,
        timeout: ENROLLMENT_TIMEOUT,
        // Do NOT set Content-Type — browser sets multipart boundary automatically.
      });
    }

    /**
     * Delete an employee by ID.
     * @param {string|number} id
     */
    async deleteEmployee(id) {
      return requestJson(`/employees/${encodeURIComponent(id)}`, { method: 'DELETE' });
    }

    // ----- Attendance ----------------------------------------------------

    /**
     * List attendance records with filters and pagination.
     * @param {Object} [filters]  {from, to, employee, page, page_size}
     * @returns {Promise<{items: Array, total: number, page: number, page_size: number}>}
     */
    async listAttendance(filters = {}) {
      const data = await requestJson(`/attendance${buildQuery(filters)}`, { method: 'GET' });
      // Normalise response shapes
      if (Array.isArray(data)) {
        return { items: data, total: data.length, page: filters.page || 1, page_size: filters.page_size || data.length };
      }
      return {
        items: data.items || data.records || [],
        total: data.total ?? (data.items ? data.items.length : 0),
        page: data.page ?? filters.page ?? 1,
        page_size: data.page_size ?? filters.page_size ?? 20,
      };
    }

    /**
     * Export attendance records as CSV text.
     * @param {Object} [filters]
     * @returns {Promise<string>} CSV text
     */
    async exportAttendance(filters = {}) {
      const res = await request(`/attendance/export${buildQuery(filters)}`, { method: 'GET' });
      if (!res.ok) {
        const t = await res.text().catch(() => '');
        throw new Error(t || `Export failed: HTTP ${res.status}`);
      }
      return res.text();
    }

    // ----- Stats ---------------------------------------------------------

    /**
     * Aggregate statistics for charts.
     * @param {string} [range='7d']  '7d' | '30d' | '90d'
     * @returns {Promise<{daily: Array<{date, count}>, hourly: Array<{hour, count}>, departments: Array<{department, count}>}>}
     */
    async getStats(range = '7d') {
      const data = await requestJson(`/stats${buildQuery({ range })}`, { method: 'GET' });
      return {
        daily: data.daily || [],
        hourly: data.hourly || [],
        departments: data.departments || [],
        raw: data,
      };
    }
  }

  /** Singleton instance consumed by other scripts. */
  global.apiClient = new ApiClient();
  global.ApiClient = ApiClient; // expose class for testing
})(window);
