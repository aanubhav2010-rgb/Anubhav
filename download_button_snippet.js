/**
 * ReelTranscribe — Download Button Snippet
 * ==========================================
 * Apne templates/index.html mein:
 *   1. HTML section ke andar #download-btn button add karo (below ke HTML structure dekho)
 *   2. Ye poora JS block </body> se pehle paste karo
 *
 * ── Required HTML Structure (URL section mein) ──────────────────────────────
 *
 *  <div class="url-actions">
 *    <!-- Ye button pehle se existing hai -->
 *    <button id="transcribe-btn" ... >Transcribe</button>
 *
 *    <!-- YE NAYA BUTTON ADD KARO — initially hidden -->
 *    <button id="download-btn" style="display:none;">
 *      ⬇ Download Video
 *    </button>
 *  </div>
 *
 *  <!-- Status text for download -->
 *  <p id="download-status" style="display:none; margin-top:8px;"></p>
 *
 * ────────────────────────────────────────────────────────────────────────────
 */

(function () {
  "use strict";

  // ── Element references ──────────────────────────────────────────────────
  // Apne actual element IDs se match karo ↓
  const urlInput      = document.getElementById("instagram-url");   // URL input field
  const transcribeBtn = document.getElementById("transcribe-btn");  // existing Transcribe btn
  const downloadBtn   = document.getElementById("download-btn");    // new Download btn
  const dlStatus      = document.getElementById("download-status"); // status text

  if (!urlInput || !downloadBtn) {
    console.warn("[ReelTranscribe] URL input ya download button nahi mila. IDs check karo.");
    return;
  }

  // ── Helper: Valid Instagram URL check ──────────────────────────────────
  function isInstagramUrl(val) {
    return /instagram\.com\/(reel|p|tv|reels)\/[A-Za-z0-9_-]+/i.test(val.trim());
  }

  // ── URL input pe listen karo ────────────────────────────────────────────
  urlInput.addEventListener("input", function () {
    const show = isInstagramUrl(this.value);
    downloadBtn.style.display = show ? "inline-flex" : "none";
    if (!show) resetDownloadUI();
  });

  // Also check on paste
  urlInput.addEventListener("paste", function () {
    setTimeout(() => {
      const show = isInstagramUrl(urlInput.value);
      downloadBtn.style.display = show ? "inline-flex" : "none";
      if (!show) resetDownloadUI();
    }, 50);
  });

  // ── Download button click ───────────────────────────────────────────────
  downloadBtn.addEventListener("click", async function () {
    const url = urlInput.value.trim();
    if (!isInstagramUrl(url)) return;

    setDownloadState("loading", "⏳ Downloading... please wait");

    try {
      // Step 1: Start download job
      const formData = new FormData();
      formData.append("url", url);

      const startRes = await fetch("/api/download-only", {
        method: "POST",
        body:   formData,
      });

      if (!startRes.ok) {
        const err = await startRes.json().catch(() => ({}));
        throw new Error(err.detail || "Server error starting download.");
      }

      const { job_id } = await startRes.json();

      // Step 2: Poll status
      const finalStatus = await pollJobStatus(job_id);

      if (finalStatus.status === "completed") {
        // Step 3: Trigger browser download
        setDownloadState("success", "✅ Done! Video save ho raha hai...");
        triggerFileDownload(`/api/serve-download/${job_id}`, finalStatus.result?.video_filename);
        setTimeout(resetDownloadUI, 4000);
      } else {
        throw new Error(finalStatus.error || "Download failed.");
      }

    } catch (err) {
      setDownloadState("error", `❌ Error: ${err.message}`);
      setTimeout(resetDownloadUI, 5000);
    }
  });

  // ── Poll /api/status/{job_id} until done/failed ─────────────────────────
  async function pollJobStatus(jobId, interval = 2000, maxWait = 120000) {
    const deadline = Date.now() + maxWait;

    while (Date.now() < deadline) {
      await sleep(interval);

      const res  = await fetch(`/api/status/${jobId}`);
      const data = await res.json();

      // Update progress text
      if (data.step) {
        const stepMap = {
          initializing: "⏳ Starting...",
          downloading:  "⬇ Downloading video...",
          done:         "✅ Complete!",
        };
        setDownloadState("loading", stepMap[data.step] || `⏳ ${data.step}...`);
      }

      if (data.status === "completed" || data.status === "failed") {
        return data;
      }
    }

    throw new Error("Timeout: download took too long.");
  }

  // ── Trigger <a download> trick to save file ─────────────────────────────
  function triggerFileDownload(url, filename) {
    const a    = document.createElement("a");
    a.href     = url;
    a.download = filename || "instagram_video.mp4";
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    setTimeout(() => document.body.removeChild(a), 500);
  }

  // ── UI helpers ──────────────────────────────────────────────────────────
  function setDownloadState(state, message) {
    if (dlStatus) {
      dlStatus.style.display = "block";
      dlStatus.textContent   = message;
    }

    if (state === "loading") {
      downloadBtn.disabled     = true;
      downloadBtn.textContent  = "⏳ Downloading...";
      if (transcribeBtn) transcribeBtn.disabled = true;
    } else if (state === "success") {
      downloadBtn.disabled    = false;
      downloadBtn.textContent = "⬇ Download Video";
      if (transcribeBtn) transcribeBtn.disabled = false;
    } else if (state === "error") {
      downloadBtn.disabled    = false;
      downloadBtn.textContent = "⬇ Download Video";
      if (transcribeBtn) transcribeBtn.disabled = false;
    }
  }

  function resetDownloadUI() {
    if (dlStatus) dlStatus.style.display = "none";
    downloadBtn.disabled    = false;
    downloadBtn.textContent = "⬇ Download Video";
    if (transcribeBtn) transcribeBtn.disabled = false;
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
})();
