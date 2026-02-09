/**
 * GPU Detection Module
 *
 * Detects NVIDIA GPU presence via nvidia-smi command.
 * Used to determine whether to download CPU or CUDA whisper.cpp builds.
 */

const { spawn } = require("child_process");
const debugLogger = require("./debugLogger");

// Cache detection result to avoid repeated checks
let cachedResult = null;
let detecting = false;
let detectionPromise = null;

/**
 * Detects if an NVIDIA GPU is present on the system.
 * Uses nvidia-smi command which is available when NVIDIA drivers are installed.
 *
 * @returns {Promise<{hasNvidiaGpu: boolean, gpuName?: string, gpuCount?: number, gpus?: string[], error?: string}>}
 */
async function detectNvidiaGpu() {
  // macOS uses Metal, not CUDA - skip detection entirely
  if (process.platform === "darwin") {
    return { hasNvidiaGpu: false, error: "macOS uses Metal, not CUDA" };
  }

  // Return cached result if available
  if (cachedResult !== null) {
    debugLogger.log("[gpuDetector] Returning cached result:", cachedResult);
    return cachedResult;
  }

  // Prevent multiple simultaneous detections
  if (detecting && detectionPromise) {
    debugLogger.log("[gpuDetector] Detection already in progress, waiting...");
    return detectionPromise;
  }

  detecting = true;
  detectionPromise = performDetection();

  try {
    cachedResult = await detectionPromise;
    return cachedResult;
  } finally {
    detecting = false;
    detectionPromise = null;
  }
}

/**
 * Performs the actual GPU detection using nvidia-smi.
 * @returns {Promise<{hasNvidiaGpu: boolean, gpuName?: string, gpuCount?: number, error?: string}>}
 */
function performDetection() {
  return new Promise((resolve) => {
    const TIMEOUT_MS = 3000;
    let settled = false;

    debugLogger.log("[gpuDetector] Starting NVIDIA GPU detection...");

    const proc = spawn("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"], {
      stdio: ["ignore", "pipe", "pipe"],
      shell: process.platform === "win32",
    });

    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      debugLogger.warn("[gpuDetector] Detection timed out after", TIMEOUT_MS, "ms");
      if (proc.exitCode === null) {
        proc.kill();
      }
      resolve({
        hasNvidiaGpu: false,
        error: "Detection timed out",
      });
    }, TIMEOUT_MS);

    let stdout = "";
    let stderr = "";

    proc.stdout?.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr?.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);

      if (code === 0 && stdout.trim()) {
        const gpuNames = stdout
          .trim()
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line.length > 0);

        debugLogger.log("[gpuDetector] Found NVIDIA GPU(s):", gpuNames);

        resolve({
          hasNvidiaGpu: true,
          gpuName: gpuNames[0],
          gpuCount: gpuNames.length,
          gpus: gpuNames,
        });
      } else {
        debugLogger.log("[gpuDetector] No NVIDIA GPU detected (exit code:", code, ")");
        if (stderr.trim()) {
          debugLogger.log("[gpuDetector] stderr:", stderr.trim());
        }

        resolve({
          hasNvidiaGpu: false,
          error: code === null ? "Process killed" : `nvidia-smi exited with code ${code}`,
        });
      }
    });

    proc.on("error", (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);

      const errorMsg = err.code === "ENOENT" ? "nvidia-smi not found (NVIDIA drivers not installed)" : err.message;

      debugLogger.log("[gpuDetector] Error:", errorMsg);

      resolve({
        hasNvidiaGpu: false,
        error: errorMsg,
      });
    });
  });
}

/**
 * Clears the cached detection result.
 */
function clearCache() {
  cachedResult = null;
  debugLogger.log("[gpuDetector] Cache cleared");
}

/**
 * Returns the recommended whisper.cpp variant based on GPU detection.
 * @returns {Promise<'cpu' | 'gpu'>}
 */
async function getRecommendedVariant() {
  if (process.platform !== "win32" && process.platform !== "linux") {
    return "cpu";
  }

  const result = await detectNvidiaGpu();
  return result.hasNvidiaGpu ? "gpu" : "cpu";
}

/**
 * Synchronous version using cached results or execSync fallback.
 * @returns {'cpu' | 'gpu'}
 */
function getRecommendedVariantSync() {
  if (process.platform !== "win32" && process.platform !== "linux") {
    return "cpu";
  }

  if (cachedResult !== null) {
    return cachedResult.hasNvidiaGpu ? "gpu" : "cpu";
  }

  try {
    const { execSync } = require("child_process");
    const stdout = execSync("nvidia-smi --query-gpu=name --format=csv,noheader", {
      encoding: "utf8",
      timeout: 3000,
      stdio: ["pipe", "pipe", "ignore"],
      windowsHide: true,
    });

    if (stdout && stdout.trim()) {
      const gpuNames = stdout
        .trim()
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 0);

      debugLogger.log("[gpuDetector] GPU detected (sync):", gpuNames[0]);

      cachedResult = {
        hasNvidiaGpu: true,
        gpuName: gpuNames[0],
        gpuCount: gpuNames.length,
        gpus: gpuNames,
      };

      return "gpu";
    }
  } catch (error) {
    debugLogger.log("[gpuDetector] Sync detection failed, using CPU:", error.message);
  }

  cachedResult = {
    hasNvidiaGpu: false,
    error: "Synchronous detection failed or no GPU found",
  };

  return "cpu";
}

module.exports = {
  detectNvidiaGpu,
  clearCache,
  getRecommendedVariant,
  getRecommendedVariantSync,
};
