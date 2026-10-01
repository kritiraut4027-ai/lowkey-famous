/* =============================================================
   HAMSHAKAL FINDER — script.js
   Handles: image upload/preview, drag & drop, the scan/loading
   sequence, result rendering, error states, toasts, confetti,
   and the placeholder hook for the real Flask backend.
   ============================================================= */

(() => {
  "use strict";

  /* ---------------------------------------------------------
     0. ELEMENT REFERENCES
     --------------------------------------------------------- */
  const dropzone       = document.getElementById("dropzone");
  const fileInput      = document.getElementById("fileInput");
  const previewState   = document.getElementById("previewState");
  const previewImg     = document.getElementById("previewImg");
  const loadingState    = document.getElementById("loadingState");
  const loadingText    = document.getElementById("loadingText");
  const loaderPct       = document.getElementById("loaderPct");
  const loaderArc       = document.querySelector(".radial-loader__arc");
  const findBtn        = document.getElementById("findBtn");
  const fileMeta       = document.getElementById("fileMeta");

  const uploadSection  = document.getElementById("uploadSection");
  const resultSection  = document.getElementById("resultSection");
  const errorSection   = document.getElementById("errorSection");

  const resultUserImg  = document.getElementById("resultUserImg");
  const resultCelebImg = document.getElementById("resultCelebImg");
  const celebName      = document.getElementById("celebName");
  const matchValue     = document.getElementById("matchValue");
  const matchArc       = document.getElementById("matchArc");
  const matchPercentText = document.getElementById("matchPercentText");

  const retryBtn       = document.getElementById("retryBtn");
  const downloadBtn    = document.getElementById("downloadBtn");
  const shareBtn       = document.getElementById("shareBtn");
  const errorTitle     = document.getElementById("errorTitle");
  const errorDesc      = document.getElementById("errorDesc");
  const errorRetryBtn  = document.getElementById("errorRetryBtn");

  const toastStack     = document.getElementById("toastStack");
  const confettiCanvas = document.getElementById("confettiCanvas");

  document.getElementById("year").textContent = new Date().getFullYear();

  /* ---------------------------------------------------------
     1. STATE
     --------------------------------------------------------- */
  let selectedFile = null;
  let selectedFileDataUrl = null;

  const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp"];
  const MAX_FILE_MB = 8;

  /* Rotating "AI is thinking" copy shown during the scan.
     Swap or extend this list freely — it is purely cosmetic. */
  const PROCESSING_MESSAGES = [
    "Detecting facial features…",
    "Generating embeddings…",
    "Comparing with celebrity database…",
    "Finding your Hamshakal…",
  ];

  /* ---------------------------------------------------------
     2. TOASTS
     --------------------------------------------------------- */
  function showToast(title, message, type = "error") {
    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;
    toast.innerHTML = `
      <div>
        <p class="toast__title">${title}</p>
        <p class="toast__msg">${message}</p>
      </div>`;
    toastStack.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("is-leaving");
      toast.addEventListener("animationend", () => toast.remove());
    }, 4200);
  }

  /* ---------------------------------------------------------
     3. FILE SELECTION (click-to-browse + drag & drop)
     --------------------------------------------------------- */
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  });
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  });

  function handleFile(file) {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      showToast("Unsupported file type", "Please upload a JPG, PNG, or WEBP image.");
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      showToast("File too large", `Please upload an image under ${MAX_FILE_MB}MB.`);
      return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      selectedFileDataUrl = e.target.result;
      previewImg.src = selectedFileDataUrl;
      dropzone.hidden = true;
      previewState.hidden = false;
      findBtn.disabled = false;
      fileMeta.textContent = `· ${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
    };
    reader.readAsDataURL(file);
  }

  /* ---------------------------------------------------------
     4. SCAN / LOADING SEQUENCE
     --------------------------------------------------------- */
  const LOADER_RADIUS = 52;
  const LOADER_CIRCUMFERENCE = 2 * Math.PI * LOADER_RADIUS;

  function setLoaderProgress(pct) {
    const offset = LOADER_CIRCUMFERENCE * (1 - pct / 100);
    loaderArc.style.strokeDashoffset = offset;
    loaderPct.textContent = `${Math.round(pct)}%`;
  }

  function runScanSequence() {
    return new Promise((resolve) => {
      previewState.hidden = true;
      loadingState.hidden = false;
      document.body.classList.add("is-processing");

      let msgIndex = 0;
      loadingText.textContent = PROCESSING_MESSAGES[0];
      const msgInterval = setInterval(() => {
        msgIndex = (msgIndex + 1) % PROCESSING_MESSAGES.length;
        loadingText.textContent = PROCESSING_MESSAGES[msgIndex];
      }, 700);

      let progress = 0;
      const progressInterval = setInterval(() => {
        // Ease toward 100 without ever quite finishing on its own —
        // the final jump to 100% happens once the "API" responds.
        progress += (100 - progress) * 0.06 + 1;
        if (progress > 96) progress = 96;
        setLoaderProgress(progress);
      }, 90);

      // Total simulated processing time.
      setTimeout(() => {
        clearInterval(msgInterval);
        clearInterval(progressInterval);
        setLoaderProgress(100);
        setTimeout(resolve, 250);
      }, 2800);
    });
  }

  /* ---------------------------------------------------------
     5. BACKEND INTEGRATION PLACEHOLDER
     ---------------------------------------------------------
     Replace mockFaceMatchRequest() with a real call to your
     Flask backend, e.g.:

     async function fetchFaceMatch(file) {
       const formData = new FormData();
       formData.append("image", file);

       const response = await fetch("https://your-api.example.com/api/match", {
         method: "POST",
         body: formData,
       });

       if (!response.ok) {
         throw new Error("API_ERROR");
       }

       return await response.json();
       // Expected shape:
       // {
       //   status: "ok" | "no_face" | "multiple_faces" | "no_match",
       //   celebrity_name: "Deepika Padukone",
       //   celebrity_image_url: "https://.../deepika.jpg",
       //   match_percentage: 98.42
       // }
     }
     --------------------------------------------------------- */
  // Configurable API base URL: empty string uses relative paths (unified deployment)
  const isLocalDevHost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
  const API_BASE = (typeof window.API_BASE_URL !== "undefined" && window.API_BASE_URL)
    ? window.API_BASE_URL.replace(/\/$/, "")
    : (window.location.protocol === "file:" || (isLocalDevHost && window.location.port && window.location.port !== "5000"))
      ? "http://127.0.0.1:5000"
      : "";

  async function fetchFaceMatch(file) {
    const formData = new FormData();
    formData.append("image", file);

    const uploadUrl = API_BASE ? `${API_BASE}/upload` : "/upload";
    const response = await fetch(uploadUrl, {
      method: "POST",
      body: formData,
    });

    const data = await response.json().catch(() => null);
    if (!data) {
      throw new Error("Invalid response from server");
    }
    return data;
  }   
  /*function mockFaceMatchRequest() {
    // Small demo roster so the UI has something believable to render.
    // Swap this out entirely once the real API is wired up.
    const roster = [
      { name: "Deepika Padukone", seed: "DP", hue: 320 },
      { name: "Ranbir Kapoor", seed: "RK", hue: 210 },
      { name: "Priyanka Chopra", seed: "PC", hue: 280 },
      { name: "Shah Rukh Khan", seed: "SRK", hue: 40 },
      { name: "Alia Bhatt", seed: "AB", hue: 260 },
      { name: "Hrithik Roshan", seed: "HR", hue: 190 },
    ];

    // Roll a small chance of each error state so all paths are reachable in a demo.
    const roll = Math.random();
    if (roll < 0.08) return Promise.resolve({ status: "no_face" });
    if (roll < 0.14) return Promise.resolve({ status: "multiple_faces" });
    if (roll < 0.18) return Promise.resolve({ status: "no_match" });

    const pick = roster[Math.floor(Math.random() * roster.length)];
    const matchPct = +(65 + Math.random() * 34).toFixed(2); // 65.00 – 99.00

    return Promise.resolve({
      status: "ok",
      celebrity_name: pick.name,
      celebrity_image_url: generateAvatarDataUrl(pick.seed, pick.hue),
      match_percentage: matchPct,
    });
  }*/

  /* Generates a soft gradient "avatar" placeholder with initials so the
     demo never depends on external celebrity photos. Swap celebrity_image_url
     for a real, licensed photo URL from your backend in production. */
  function generateAvatarDataUrl(initials, hue) {
    const svg = `
      <svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">
        <defs>
          <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="hsl(${hue},70%,55%)"/>
            <stop offset="100%" stop-color="hsl(${(hue + 60) % 360},70%,45%)"/>
          </linearGradient>
        </defs>
        <rect width="300" height="300" fill="url(#g)"/>
        <text x="50%" y="54%" font-family="Space Grotesk, sans-serif" font-size="96"
          font-weight="700" fill="rgba(255,255,255,0.92)" text-anchor="middle" dominant-baseline="middle">
          ${initials}
        </text>
      </svg>`;
    return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
  }

  /* ---------------------------------------------------------
     6. MAIN "FIND MY HAMSHAKAL" FLOW
     --------------------------------------------------------- */
  findBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    findBtn.disabled = true;

    try {
      const [response] = await Promise.all([fetchFaceMatch(selectedFile), runScanSequence()]);
      document.body.classList.remove("is-processing");

      if (response.status === "no_face") {
        showError(
          "No Face Detected",
          "We couldn't find a face in this image. Try a clearer, front-facing photo with good lighting."
        );
        return;
      }
      if (response.status === "multiple_faces") {
        showError(
          "Multiple Faces Detected",
          "This photo has more than one face. Upload a solo photo so we can match the right one."
        );
        return;
      }
      if (response.status === "invalid_image") {
        showError(
          "Invalid Image",
          response.error || "The uploaded image could not be processed. Please upload a clear JPG, PNG, or WEBP photo."
        );
        return;
      }
      if (response.status === "no_match") {
        showError(
          "No Close Match Found",
          "We compared your face against our database but couldn't find a confident celebrity match. Try a different photo."
        );
        return;
      }
      if (response.status !== "ok") {
        showError(
          "Matching Error",
          response.error || "An unexpected error occurred while analyzing the face. Please try again."
        );
        return;
      }

      renderResult(response);
    } catch (err) {
      document.body.classList.remove("is-processing");
      showToast("Something went wrong", "The AI service is unreachable right now. Please try again.");
      resetToUpload();
    } finally {
      findBtn.disabled = false;
    }
  });

  /* ---------------------------------------------------------
     7. RESULT RENDERING
     --------------------------------------------------------- */
  const MATCH_CIRCUMFERENCE = 2 * Math.PI * 70;

  function renderResult(data) {
    uploadSection.hidden = true;
    errorSection.hidden = true;
    resultSection.hidden = false;

    // Display user image from local data URL (instant, no extra network roundtrip)
    resultUserImg.src = selectedFileDataUrl || (data.user_image_url.startsWith("http") ? data.user_image_url : `${API_BASE}${data.user_image_url}`);
    
    // Display celebrity match image
    resultCelebImg.src = data.celebrity_image_url.startsWith("http")
      ? data.celebrity_image_url
      : `${API_BASE}${data.celebrity_image_url}`;
    
    celebName.textContent = data.celebrity_name;

    animateMatchRing(data.match_percentage);

    if (data.match_percentage > 95) {
      launchConfetti();
    }

    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function animateMatchRing(targetPct) {
    // Color rule: green ≥90, yellow 70–90, red <70.
    let color = "var(--danger)";
    if (targetPct >= 90) color = "var(--success)";
    else if (targetPct >= 70) color = "var(--warning)";
    matchArc.style.stroke = getComputedColor(color);

    const duration = 1400;
    const start = performance.now();

    function tick(now) {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      const current = +(eased * targetPct).toFixed(2);

      matchValue.textContent = `${current.toFixed(0)}%`;
      matchPercentText.textContent = `${current.toFixed(2)}%`;
      matchArc.style.strokeDashoffset = MATCH_CIRCUMFERENCE * (1 - eased * (targetPct / 100));

      if (t < 1) requestAnimationFrame(tick);
      else {
        matchValue.textContent = `${targetPct.toFixed(0)}%`;
        matchPercentText.textContent = `${targetPct.toFixed(2)}%`;
      }
    }
    requestAnimationFrame(tick);
  }

  function getComputedColor(cssVarExpr) {
    // Resolves e.g. "var(--success)" to its actual hex so it can be
    // assigned directly to an SVG stroke attribute-style property.
    const varName = cssVarExpr.match(/--[\w-]+/)[0];
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }

  /* ---------------------------------------------------------
     8. ERROR RENDERING
     --------------------------------------------------------- */
  function showError(title, desc) {
    uploadSection.hidden = true;
    resultSection.hidden = true;
    errorSection.hidden = false;
    errorTitle.textContent = title;
    errorDesc.textContent = desc;
    errorSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  errorRetryBtn.addEventListener("click", resetToUpload);
  retryBtn.addEventListener("click", resetToUpload);

  function resetToUpload() {
    selectedFile = null;
    selectedFileDataUrl = null;
    fileInput.value = "";
    fileMeta.textContent = "";
    findBtn.disabled = true;

    previewState.hidden = true;
    loadingState.hidden = true;
    dropzone.hidden = false;

    resultSection.hidden = true;
    errorSection.hidden = true;
    uploadSection.hidden = false;

    setLoaderProgress(0);
    uploadSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ---------------------------------------------------------
     9. DOWNLOAD / SHARE
     --------------------------------------------------------- */
  downloadBtn.addEventListener("click", () => {
    // Renders the result card's key facts onto a canvas and downloads it as an image.
    const canvas = document.createElement("canvas");
    canvas.width = 600;
    canvas.height = 320;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#12131f";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f2f1fa";
    ctx.font = "600 26px 'Space Grotesk', sans-serif";
    ctx.fillText("Hamshakal Finder — Result", 30, 50);
    ctx.font = "400 16px Inter, sans-serif";
    ctx.fillStyle = "#8b8aa3";
    ctx.fillText("Celebrity Match:", 30, 110);
    ctx.fillStyle = "#f2f1fa";
    ctx.font = "600 22px 'Space Grotesk', sans-serif";
    ctx.fillText(celebName.textContent, 30, 140);
    ctx.fillStyle = "#8b8aa3";
    ctx.font = "400 16px Inter, sans-serif";
    ctx.fillText("Match Percentage:", 30, 190);
    ctx.fillStyle = "#4ce0a0";
    ctx.font = "600 30px 'JetBrains Mono', monospace";
    ctx.fillText(matchPercentText.textContent, 30, 226);

    const link = document.createElement("a");
    link.download = "hamshakal-result.png";
    link.href = canvas.toDataURL("image/png");
    link.click();

    showToast("Downloaded", "Your result card has been saved.", "success");
  });

  shareBtn.addEventListener("click", async () => {
    const shareText = `I got matched with ${celebName.textContent} (${matchPercentText.textContent}) on Hamshakal Finder! 🎭`;
    if (navigator.share) {
      try {
        await navigator.share({ title: "Hamshakal Finder", text: shareText });
      } catch {
        /* user cancelled share — no action needed */
      }
    } else {
      await navigator.clipboard.writeText(shareText);
      showToast("Copied to clipboard", "Share text copied — paste it anywhere!", "success");
    }
  });

  /* ---------------------------------------------------------
     10. CONFETTI (match > 95%)
     --------------------------------------------------------- */
  function launchConfetti() {
    const ctx = confettiCanvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    confettiCanvas.width = window.innerWidth * dpr;
    confettiCanvas.height = window.innerHeight * dpr;
    confettiCanvas.style.width = window.innerWidth + "px";
    confettiCanvas.style.height = window.innerHeight + "px";
    ctx.scale(dpr, dpr);

    const colors = ["#7b5cff", "#ff5ca8", "#4ce0d2", "#4ce0a0", "#ffc24c"];
    const pieces = Array.from({ length: 140 }, () => ({
      x: Math.random() * window.innerWidth,
      y: -20 - Math.random() * window.innerHeight * 0.5,
      size: 5 + Math.random() * 6,
      color: colors[Math.floor(Math.random() * colors.length)],
      speedY: 2 + Math.random() * 3,
      speedX: -1.5 + Math.random() * 3,
      rotation: Math.random() * 360,
      spin: -6 + Math.random() * 12,
    }));

    let frame = 0;
    const maxFrames = 200;

    function draw() {
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      pieces.forEach((p) => {
        p.x += p.speedX;
        p.y += p.speedY;
        p.rotation += p.spin;

        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate((p.rotation * Math.PI) / 180);
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
        ctx.restore();
      });

      frame++;
      if (frame < maxFrames) {
        requestAnimationFrame(draw);
      } else {
        ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      }
    }
    draw();
  }

  window.addEventListener("resize", () => {
    confettiCanvas.width = 0;
    confettiCanvas.height = 0;
  });
})();
