(function() {
  const Verity = {
    session: {
      startTime: null,
      keyEvents: [],
      mouseEvents: [],
      lastKeyTime: null,
      lastMouseX: null,
      lastMouseY: null,
      lastMouseTime: null,
      lastMouseVX: null,
      lastMouseVY: null,
      editCount: 0,
      totalKeys: 0,
      dwellTimes: {},
      flightTimes: [],
      keyDownTimes: {},
      enrolled: false,
      baseline: null,
      enrollBuffer: {
        dwellTimes: [],
        flightTimes: [],
        mouseSpeeds: [],
        mouseAccels: [],
        mouseJerks: [],
        mouseCurvatures: [],
        mousePauses: [],
        backspaceRatio: 0,
        backspaceCount: 0,
        totalKeysInEnroll: 0
      },
      enrolling: false,
      enrollStartTime: null,
      enrollDuration: 50000
    },

    init: function(options) {
      this.session.startTime = Date.now();
      this.options = options || {};
      this._attachKeyboard();
      this._attachMouse();
      console.log('Verity SDK initialized');
    },

    startEnrollment: function() {
      const s = this.session;
      s.enrolling = true;
      s.enrolled = false;
      s.baseline = null;
      s.enrollStartTime = Date.now();
      s.enrollBuffer = {
        dwellTimes: [],
        flightTimes: [],
        mouseSpeeds: [],
        mouseAccels: [],
        mouseJerks: [],
        mouseCurvatures: [],
        mousePauses: [],
        backspaceCount: 0,
        totalKeysInEnroll: 0
      };
      console.log('Enrollment started');
      return true;
    },

    _attachKeyboard: function() {
      const s = this.session;

      document.addEventListener('keydown', (e) => {
        const now = Date.now();
        s.keyDownTimes[e.code] = now;
        s.totalKeys++;

        if (s.enrolling) {
          s.enrollBuffer.totalKeysInEnroll++;
          if (e.key === 'Backspace') {
            s.enrollBuffer.backspaceCount++;
          }
          if (s.lastKeyTime !== null) {
            const flight = now - s.lastKeyTime;
            if (flight > 0 && flight < 2000) {
              s.enrollBuffer.flightTimes.push(flight);
            }
          }
        }

        if (e.key === 'Backspace' || e.key === 'Delete') {
          s.editCount++;
        }

        s.lastKeyTime = now;
      });

      document.addEventListener('keyup', (e) => {
        const now = Date.now();
        const downTime = s.keyDownTimes[e.code];
        if (downTime) {
          const dwell = now - downTime;
          if (dwell > 0 && dwell < 500) {
            if (!s.dwellTimes[e.code]) s.dwellTimes[e.code] = [];
            s.dwellTimes[e.code].push(dwell);
            if (s.enrolling) {
              s.enrollBuffer.dwellTimes.push(dwell);
            }
          }
          delete s.keyDownTimes[e.code];
        }

        if (s.enrolling) {
          const elapsed = Date.now() - s.enrollStartTime;
          if (elapsed >= s.enrollDuration && 
              s.enrollBuffer.dwellTimes.length >= 20) {
            this._finalizeEnrollment();
          }
        }
      });
    },

    _attachMouse: function() {
      const s = this.session;
      let lastVX = 0, lastVY = 0;
      let lastAX = 0, lastAY = 0;

      document.addEventListener('mousemove', (e) => {
        const now = Date.now();
        const x = e.clientX, y = e.clientY;

        if (s.lastMouseX !== null && s.lastMouseTime !== null) {
          const dt = (now - s.lastMouseTime) / 1000;
          if (dt > 0 && dt < 0.5) {
            const dx = x - s.lastMouseX;
            const dy = y - s.lastMouseY;
            const dist = Math.sqrt(dx*dx + dy*dy);
            const speed = dist / dt;

            const vx = dx / dt, vy = dy / dt;
            const ax = (vx - lastVX) / dt;
            const ay = (vy - lastVY) / dt;
            const accel = Math.sqrt(ax*ax + ay*ay);

            const jx = (ax - lastAX) / dt;
            const jy = (ay - lastAY) / dt;
            const jerk = Math.sqrt(jx*jx + jy*jy);

            const curvature = dist > 0 ?
              Math.abs(dx * ay - dy * ax) / Math.pow(dist, 3) : 0;

            s.mouseEvents.push({t: now, x, y, speed, accel, jerk, curvature});

            if (s.enrolling) {
              if (speed < 2000) s.enrollBuffer.mouseSpeeds.push(speed);
              if (accel < 50000) s.enrollBuffer.mouseAccels.push(accel);
              if (jerk < 500000) s.enrollBuffer.mouseJerks.push(jerk);
              s.enrollBuffer.mouseCurvatures.push(Math.min(curvature, 1));
            }

            lastVX = vx; lastVY = vy;
            lastAX = ax; lastAY = ay;
          }

          if (now - s.lastMouseTime > 800) {
            if (s.enrolling) {
              s.enrollBuffer.mousePauses.push(now - s.lastMouseTime);
            }
          }
        }

        s.lastMouseX = x;
        s.lastMouseY = y;
        s.lastMouseTime = now;
      });
    },

    _mean: function(arr) {
      if (!arr || arr.length === 0) return 0;
      return arr.reduce((a,b) => a+b, 0) / arr.length;
    },

    _std: function(arr) {
      if (!arr || arr.length < 2) return 1;
      const m = this._mean(arr);
      const variance = arr.reduce((a,b) => a + (b-m)**2, 0) / arr.length;
      return Math.sqrt(variance) || 1;
    },

    _finalizeEnrollment: function() {
      const s = this.session;
      const b = s.enrollBuffer;

      s.baseline = {
        dwell: {
          mean: this._mean(b.dwellTimes),
          std: this._std(b.dwellTimes)
        },
        flight: {
          mean: this._mean(b.flightTimes),
          std: this._std(b.flightTimes)
        },
        mouseSpeed: {
          mean: this._mean(b.mouseSpeeds),
          std: this._std(b.mouseSpeeds)
        },
        mouseAccel: {
          mean: this._mean(b.mouseAccels),
          std: this._std(b.mouseAccels)
        },
        mouseJerk: {
          mean: this._mean(b.mouseJerks),
          std: this._std(b.mouseJerks)
        },
        mouseCurvature: {
          mean: this._mean(b.mouseCurvatures),
          std: this._std(b.mouseCurvatures)
        },
        backspaceRatio: b.totalKeysInEnroll > 0 ?
          b.backspaceCount / b.totalKeysInEnroll : 0,
        pauseFrequency: b.mousePauses.length,
        rawSamples: {
          dwellCount: b.dwellTimes.length,
          flightCount: b.flightTimes.length,
          mouseCount: b.mouseSpeeds.length
        }
      };

      s.enrolling = false;
      s.enrolled = true;
      console.log('Enrollment complete', s.baseline);

      if (this.options.onEnrollmentComplete) {
        this.options.onEnrollmentComplete(s.baseline);
      }
    },

    getEnrollmentProgress: function() {
      const s = this.session;
      if (!s.enrolling) return s.enrolled ? 100 : 0;
      const timeProgress = Math.min(
        (Date.now() - s.enrollStartTime) / s.enrollDuration * 100, 100
      );
      const dataProgress = Math.min(
        s.enrollBuffer.dwellTimes.length / 20 * 100, 100
      );
      return Math.floor((timeProgress * 0.5) + (dataProgress * 0.5));
    },

    collectSignals: function() {
      const s = this.session;
      const allDwells = Object.values(s.dwellTimes).flat();
      const recentMouse = s.mouseEvents.slice(-100);

      const dwellMean = this._mean(allDwells);
      const dwellStd = this._std(allDwells);
      const flightMean = this._mean(s.flightTimes);
      const flightStd = this._std(s.flightTimes);

      const speeds = recentMouse.map(e => e.speed);
      const accels = recentMouse.map(e => e.accel);
      const jerks = recentMouse.map(e => e.jerk);
      const curvatures = recentMouse.map(e => e.curvature);

      const sessionDuration = (Date.now() - s.startTime) / 1000;
      const backspaceRatio = s.totalKeys > 0 ?
        s.editCount / s.totalKeys : 0;

      const mousePauses = [];
      for (let i = 1; i < s.mouseEvents.length; i++) {
        const gap = s.mouseEvents[i].t - s.mouseEvents[i-1].t;
        if (gap > 800) mousePauses.push(gap);
      }

      return {
        keyboard: {
          dwell_mean: dwellMean,
          dwell_std: dwellStd,
          flight_mean: flightMean,
          flight_std: flightStd,
          backspace_ratio: backspaceRatio,
          total_keys: s.totalKeys,
          edit_count: s.editCount
        },
        mouse: {
          speed_mean: this._mean(speeds),
          speed_std: this._std(speeds),
          accel_mean: this._mean(accels),
          accel_std: this._std(accels),
          jerk_mean: this._mean(jerks),
          curvature_mean: this._mean(curvatures),
          pause_count: mousePauses.length,
          pause_mean: this._mean(mousePauses)
        },
        session: {
          duration_s: sessionDuration,
          start_time: s.startTime
        },
        baseline: s.baseline,
        enrolled: s.enrolled
      };
    },

    verify: async function(content, apiUrl) {
      const signals = this.collectSignals();
      signals.content = content;

      const response = await fetch((apiUrl || '') + '/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(signals)
      });

      return await response.json();
    }
  };

  window.Verity = Verity;
})();
