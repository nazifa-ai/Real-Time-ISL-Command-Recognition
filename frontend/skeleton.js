/* Draws the landmarks the backend actually extracted and fed to the model for the
   most recent prediction -- NOT a separate live client-side detector. After each
   /predict response that includes `landmarks` (shape: timesteps x 75 x 3, in the
   same [0,1] image-normalised space MediaPipe outputs), call
   window.landmarkReplay.play(landmarks, video) to draw them frame-by-frame over
   ~1s so you can see exactly what the model saw. Between captures the canvas
   is idle. */
(function () {
  // Standard 21-point MediaPipe Hand topology (index pairs).
  const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4], [0,5],[5,6],[6,7],[7,8], [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16], [13,17],[17,18],[18,19],[19,20], [0,17],
  ];
  // Simplified upper-body subset of the 33-point MediaPipe Pose topology --
  // matches the named landmarks this project's config.py actually relies on.
  const POSE_CONNECTIONS = [
    [11,12], [11,13],[13,15], [12,14],[14,16], [11,23],[12,24],[23,24], [0,11],[0,12],
  ];
  const POSE_SLICE = [0, 33], LEFT_SLICE = [33, 54], RIGHT_SLICE = [54, 75];

  let canvas, ctx, badge;

  function init() {
    canvas = document.getElementById('skeletonCanvas');
    badge = document.getElementById('trackBadge');
    if (canvas) ctx = canvas.getContext('2d');
  }

  function sizeToVideo(video) {
    if (!video || !video.videoWidth) return;
    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
  }

  function drawGroup(points, slice, connections, dotColor, lineColor) {
    const start = slice[0], end = slice[1];
    const w = canvas.width, h = canvas.height;
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2;
    for (const pair of connections) {
      const pa = points[start + pair[0]], pb = points[start + pair[1]];
      if (!pa || !pb || (pa[0] === 0 && pa[1] === 0) || (pb[0] === 0 && pb[1] === 0)) continue;
      ctx.beginPath();
      ctx.moveTo(pa[0] * w, pa[1] * h);
      ctx.lineTo(pb[0] * w, pb[1] * h);
      ctx.stroke();
    }
    ctx.fillStyle = dotColor;
    for (let i = start; i < end; i++) {
      const p = points[i];
      if (!p || (p[0] === 0 && p[1] === 0)) continue;
      ctx.beginPath();
      ctx.arc(p[0] * w, p[1] * h, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawFrame(points) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawGroup(points, POSE_SLICE, POSE_CONNECTIONS, '#7c3aedaa', '#7c3aed66');
    drawGroup(points, LEFT_SLICE, HAND_CONNECTIONS, '#22d3ee', '#a855f7');
    drawGroup(points, RIGHT_SLICE, HAND_CONNECTIONS, '#22d3ee', '#a855f7');
  }

  function countActive(points) {
    let hands = 0, pose = false, rightSeen = false;
    for (let i = LEFT_SLICE[0]; i < LEFT_SLICE[1]; i++) {
      if (points[i] && (points[i][0] || points[i][1])) { hands++; break; }
    }
    for (let i = RIGHT_SLICE[0]; i < RIGHT_SLICE[1]; i++) {
      if (points[i] && (points[i][0] || points[i][1])) { rightSeen = true; break; }
    }
    if (rightSeen) hands++;
    for (let i = POSE_SLICE[0]; i < POSE_SLICE[1]; i++) {
      if (points[i] && (points[i][0] || points[i][1])) { pose = true; break; }
    }
    return { hands: hands, pose: pose };
  }

  window.landmarkReplay = {
    /** landmarks: (T,75,3) as returned by /predict. video: the <video> element, for sizing. */
    play: function (landmarks, video) {
      if (!ctx) init();
      if (!ctx || !landmarks || !landmarks.length) return;
      sizeToVideo(video);
      let i = 0;
      const stepMs = Math.max(20, Math.floor(900 / landmarks.length));
      const last = countActive(landmarks[landmarks.length - 1]);
      if (badge) {
        badge.textContent = 'Verified: ' + last.hands + ' hand' + (last.hands === 1 ? '' : 's') +
          (last.pose ? ' + pose' : '') + ' (from actual model input)';
      }
      const timer = setInterval(function () {
        if (i >= landmarks.length) { clearInterval(timer); return; }
        drawFrame(landmarks[i]);
        i++;
        }, stepMs);
    },
    /** Draw one current camera frame of the landmarks actually extracted by the backend. */
    draw: function (points, video) {
      if (!ctx) init();
      if (!ctx || !points || !points.length) return;
      sizeToVideo(video);
      drawFrame(points);
      const active = countActive(points);
      if (badge) badge.textContent = 'Live landmarks: ' + active.hands + ' hand' +
        (active.hands === 1 ? '' : 's') + (active.pose ? ' + pose' : '');
    },
    clear: function () {
      if (!ctx) init();
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (badge) badge.textContent = 'Verified: — (capture a sign first)';
    },
  };

  init();
})();
