/**
 * @file charts.js
 * @description Chart.js initialisation for the Stats tab. Three charts:
 *   1. Daily Attendance Count    — line chart
 *   2. Hourly Distribution       — bar chart
 *   3. Department Breakdown      — doughnut chart
 *
 *   Exposes a global `chartsController` with `render(stats)` and `destroy()`.
 *   Loads data via `apiClient.getStats(range)`.
 */

(function (global) {
  'use strict';

  // Guard: Chart.js must be loaded via CDN before this script.
  if (typeof Chart === 'undefined') {
    console.error('[charts] Chart.js is not loaded.');
    return;
  }

  /** Common dark-theme Chart.js defaults. */
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = 'rgba(148, 163, 184, 0.12)';
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.size = 11;

  /** Colour palette reused across charts. */
  const PALETTE = ['#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6'];

  /**
   * ChartsController — owns the three Chart.js instances and provides
   * a single `render()` entry point that takes API stats data.
   */
  class ChartsController {
    constructor() {
      this._daily = null;
      this._hourly = null;
      this._dept = null;
    }

    /**
     * Initialise all three charts with empty data. Call once on DOM ready.
     * Safe to call multiple times (idempotent).
     */
    init() {
      if (this._daily) return;

      const dailyCanvas = document.getElementById('chartDaily');
      const hourlyCanvas = document.getElementById('chartHourly');
      const deptCanvas = document.getElementById('chartDept');
      if (!dailyCanvas || !hourlyCanvas || !deptCanvas) return;

      // 1. Daily attendance — line chart
      this._daily = new Chart(dailyCanvas, {
        type: 'line',
        data: {
          labels: [],
          datasets: [{
            label: 'Check-ins',
            data: [],
            borderColor: PALETTE[0],
            backgroundColor: 'rgba(59, 130, 246, 0.18)',
            fill: true,
            tension: 0.35,
            pointRadius: 3,
            pointHoverRadius: 5,
            pointBackgroundColor: PALETTE[0],
            borderWidth: 2,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { color: 'rgba(148,163,184,0.06)' }, ticks: { maxRotation: 0 } },
            y: { grid: { color: 'rgba(148,163,184,0.06)' }, beginAtZero: true, ticks: { precision: 0 } },
          },
        },
      });

      // 2. Hourly distribution — bar chart
      this._hourly = new Chart(hourlyCanvas, {
        type: 'bar',
        data: {
          labels: Array.from({ length: 24 }, (_, h) => `${h.toString().padStart(2, '0')}`),
          datasets: [{
            label: 'Check-ins',
            data: Array(24).fill(0),
            backgroundColor: PALETTE[1] + 'cc',
            borderColor: PALETTE[1],
            borderWidth: 1,
            borderRadius: 4,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false } },
            y: { grid: { color: 'rgba(148,163,184,0.06)' }, beginAtZero: true, ticks: { precision: 0 } },
          },
        },
      });

      // 3. Department breakdown — doughnut chart
      this._dept = new Chart(deptCanvas, {
        type: 'doughnut',
        data: {
          labels: [],
          datasets: [{
            data: [],
            backgroundColor: PALETTE,
            borderColor: '#0f1729',
            borderWidth: 2,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '62%',
          plugins: {
            legend: {
              position: 'right',
              labels: { boxWidth: 12, padding: 12, color: '#e2e8f0' },
            },
          },
        },
      });
    }

    /**
     * Render new data into all three charts.
     * @param {{daily: Array<{date:string,count:number}>,
     *          hourly: Array<{hour:number,count:number}>,
     *          departments: Array<{department:string,count:number}>}} stats
     */
    render(stats = {}) {
      if (!this._daily) this.init();
      if (!this._daily) return;

      // Daily
      const daily = stats.daily || [];
      this._daily.data.labels = daily.map((d) => d.date || d.day || '');
      this._daily.data.datasets[0].data = daily.map((d) => Number(d.count ?? d.value ?? 0));
      this._daily.update();

      // Hourly — fill 24 slots
      const hourlyMap = new Array(24).fill(0);
      (stats.hourly || []).forEach((h) => {
        const idx = Number(h.hour ?? h.hour_of_day ?? 0);
        if (idx >= 0 && idx < 24) hourlyMap[idx] = Number(h.count ?? h.value ?? 0);
      });
      this._hourly.data.datasets[0].data = hourlyMap;
      this._hourly.update();

      // Departments
      const depts = stats.departments || [];
      this._dept.data.labels = depts.map((d) => d.department || d.name || 'Unknown');
      this._dept.data.datasets[0].data = depts.map((d) => Number(d.count ?? d.value ?? 0));
      this._dept.update();
    }

    /**
     * Fetch fresh stats from the API and render.
     * @param {string} range  '7d' | '30d' | '90d'
     */
    async refresh(range = '7d') {
      if (!global.apiClient) throw new Error('apiClient is not available');
      const stats = await global.apiClient.getStats(range);
      this.render(stats);
      return stats;
    }

    /** Tear down all charts (for HMR/testing). */
    destroy() {
      [this._daily, this._hourly, this._dept].forEach((c) => c && c.destroy());
      this._daily = this._hourly = this._dept = null;
    }
  }

  global.chartsController = new ChartsController();
})(window);
