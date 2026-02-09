#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const {
  downloadFile,
  extractZip,
  fetchLatestRelease,
  parseArgs,
  setExecutable,
  cleanupFiles,
} = require("./lib/download-utils");

/**
 * Detects if an NVIDIA GPU is present using nvidia-smi command.
 * Returns 'gpu' if GPU found, 'cpu' otherwise.
 */
function detectGpuVariant() {
  return new Promise((resolve) => {
    if (process.platform !== "win32" && process.platform !== "linux") {
      resolve("cpu");
      return;
    }

    let settled = false;

    const proc = spawn("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"], {
      stdio: ["ignore", "pipe", "pipe"],
      shell: process.platform === "win32",
    });

    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      console.log("  GPU detection timed out, defaulting to CPU");
      if (proc.exitCode === null) {
        proc.kill();
      }
      resolve("cpu");
    }, 3000);

    let stdout = "";

    proc.stdout?.on("data", (data) => {
      stdout += data.toString();
    });

    proc.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      if (code === 0 && stdout.trim()) {
        const gpuName = stdout.trim().split("\n")[0];
        console.log(`  Detected NVIDIA GPU: ${gpuName}`);
        resolve("gpu");
      } else {
        console.log("  No NVIDIA GPU detected, using CPU variant");
        resolve("cpu");
      }
    });

    proc.on("error", () => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      console.log("  nvidia-smi not found, using CPU variant");
      resolve("cpu");
    });
  });
}

const WHISPER_CPP_REPO = "OpenWhispr/whisper.cpp";

// Version can be pinned via environment variable for reproducible builds
const VERSION_OVERRIDE = process.env.WHISPER_CPP_VERSION || null;

const BINARIES = {
  "darwin-arm64": {
    zipName: "whisper-server-darwin-arm64.zip",
    binaryName: "whisper-server-darwin-arm64",
    outputName: "whisper-server-darwin-arm64",
  },
  "darwin-x64": {
    zipName: "whisper-server-darwin-x64.zip",
    binaryName: "whisper-server-darwin-x64",
    outputName: "whisper-server-darwin-x64",
  },
  "win32-x64": {
    cpu: {
      zipName: "whisper-server-win32-x64-cpu.zip",
      binaryName: "whisper-server-win32-x64-cpu.exe",
      outputName: "whisper-server-win32-x64.exe",
    },
    gpu: {
      zipName: "whisper-server-win32-x64-cuda.zip",
      binaryName: "whisper-server-win32-x64-cuda.exe",
      outputName: "whisper-server-win32-x64-gpu.exe",
    },
  },
  "linux-x64": {
    cpu: {
      zipName: "whisper-server-linux-x64-cpu.zip",
      binaryName: "whisper-server-linux-x64-cpu",
      outputName: "whisper-server-linux-x64",
    },
    gpu: {
      zipName: "whisper-server-linux-x64-cuda.zip",
      binaryName: "whisper-server-linux-x64-cuda",
      outputName: "whisper-server-linux-x64-gpu",
    },
  },
};

const BIN_DIR = path.join(__dirname, "..", "resources", "bin");

// Cache the release info to avoid multiple API calls
let cachedRelease = null;

async function getRelease() {
  if (cachedRelease) return cachedRelease;

  if (VERSION_OVERRIDE) {
    cachedRelease = await fetchLatestRelease(WHISPER_CPP_REPO, { tagPrefix: VERSION_OVERRIDE });
  } else {
    cachedRelease = await fetchLatestRelease(WHISPER_CPP_REPO);
  }
  return cachedRelease;
}

function getDownloadUrl(release, zipName) {
  const asset = release?.assets?.find((a) => a.name === zipName);
  return asset?.url || null;
}

function resolveVariantConfig(config, variant) {
  if (!config) return null;
  // If config has cpu/gpu sub-configs, select the right one
  if (config.cpu || config.gpu) {
    const selected = variant || "cpu";
    if (!config[selected]) return null;
    return { ...config[selected], variant: selected };
  }
  // macOS-style single config (no variants)
  return config;
}

async function downloadBinary(platformArch, config, release, variant = null, isForce = false) {
  const resolved = resolveVariantConfig(config, variant);
  if (!resolved) {
    const tag = variant ? ` (${variant})` : "";
    console.log(`  [server]${tag} ${platformArch}: Not supported`);
    return false;
  }

  const outputPath = path.join(BIN_DIR, resolved.outputName);
  const tag = resolved.variant ? ` (${resolved.variant})` : "";

  if (fs.existsSync(outputPath) && !isForce) {
    console.log(`  [server]${tag} ${platformArch}: Already exists (use --force to re-download)`);
    return true;
  }

  const url = getDownloadUrl(release, resolved.zipName);
  if (!url) {
    console.error(`  [server]${tag} ${platformArch}: Asset ${resolved.zipName} not found in release`);
    return false;
  }
  console.log(`  [server]${tag} ${platformArch}: Downloading from ${url}`);

  const zipPath = path.join(BIN_DIR, resolved.zipName);

  try {
    await downloadFile(url, zipPath);

    const extractDir = path.join(BIN_DIR, `temp-whisper-${platformArch}${resolved.variant ? `-${resolved.variant}` : ""}`);
    fs.mkdirSync(extractDir, { recursive: true });
    extractZip(zipPath, extractDir);

    const binaryPath = path.join(extractDir, resolved.binaryName);
    if (fs.existsSync(binaryPath)) {
      fs.copyFileSync(binaryPath, outputPath);
      setExecutable(outputPath);
      console.log(`  [server]${tag} ${platformArch}: Extracted to ${resolved.outputName}`);

      // Copy companion DLLs/shared libraries (needed for CUDA builds)
      const extractedFiles = fs.readdirSync(extractDir);
      for (const file of extractedFiles) {
        if (file === resolved.binaryName) continue;
        const ext = path.extname(file).toLowerCase();
        if (ext === ".dll" || ext === ".so" || file.endsWith(".so.11") || file.endsWith(".so.12")) {
          const destPath = path.join(BIN_DIR, file);
          fs.copyFileSync(path.join(extractDir, file), destPath);
          console.log(`  [server]${tag} ${platformArch}: Copied companion ${file}`);
        }
      }
    } else {
      console.error(`  [server]${tag} ${platformArch}: Binary not found in archive`);
      return false;
    }

    fs.rmSync(extractDir, { recursive: true, force: true });
    if (fs.existsSync(zipPath)) fs.unlinkSync(zipPath);
    return true;
  } catch (error) {
    console.error(`  [server]${tag} ${platformArch}: Failed - ${error.message}`);
    if (fs.existsSync(zipPath)) fs.unlinkSync(zipPath);
    return false;
  }
}

async function main() {
  if (VERSION_OVERRIDE) {
    console.log(`\n[whisper-server] Using pinned version: ${VERSION_OVERRIDE}`);
  } else {
    console.log("\n[whisper-server] Fetching latest release...");
  }
  const release = await getRelease();

  if (!release) {
    console.error(`[whisper-server] Could not fetch release from ${WHISPER_CPP_REPO}`);
    console.log(`\nMake sure release exists: https://github.com/${WHISPER_CPP_REPO}/releases`);
    process.exitCode = 1;
    return;
  }

  // Parse explicit variant flags
  let variant = null;
  let forceDownload = false;
  if (process.argv.includes("--cpu")) {
    variant = "cpu";
    forceDownload = true;
  } else if (process.argv.includes("--gpu") || process.argv.includes("--cuda")) {
    variant = "gpu";
    forceDownload = true;
  }

  const args = parseArgs();

  // Auto-detect GPU when downloading for current platform without explicit variant
  if (args.isCurrent && !variant && (process.platform === "win32" || process.platform === "linux")) {
    console.log("\nDetecting GPU...");
    variant = await detectGpuVariant();
    forceDownload = true;
  }

  const variantTag = variant ? ` ${variant}` : "";
  console.log(`\nDownloading whisper-server binaries (${release.tag})${variantTag}...\n`);

  fs.mkdirSync(BIN_DIR, { recursive: true });

  if (args.isCurrent) {
    if (!BINARIES[args.platformArch]) {
      console.error(`Unsupported platform/arch: ${args.platformArch}`);
      process.exitCode = 1;
      return;
    }

    console.log(`Downloading for target platform (${args.platformArch}):`);
    const ok = await downloadBinary(args.platformArch, BINARIES[args.platformArch], release, variant, forceDownload || args.isForce);
    if (!ok) {
      console.error(`Failed to download binaries for ${args.platformArch}`);
      process.exitCode = 1;
      return;
    }

    if (args.shouldCleanup) {
      cleanupFiles(BIN_DIR, "whisper-server", `whisper-server-${args.platformArch}`);
    }
  } else {
    console.log("Downloading binaries for all platforms:");
    for (const platformArch of Object.keys(BINARIES)) {
      // Download both CPU and GPU variants when doing --all
      const config = BINARIES[platformArch];
      if (config.cpu && config.gpu) {
        await downloadBinary(platformArch, config, release, "cpu", args.isForce);
        await downloadBinary(platformArch, config, release, "gpu", args.isForce);
      } else {
        await downloadBinary(platformArch, config, release, null, args.isForce);
      }
    }
  }

  console.log("\n---");

  const files = fs.readdirSync(BIN_DIR).filter((f) => f.startsWith("whisper-server"));
  if (files.length > 0) {
    console.log("Available whisper-server binaries:\n");
    files.forEach((f) => {
      const stats = fs.statSync(path.join(BIN_DIR, f));
      console.log(`  - ${f} (${Math.round(stats.size / 1024 / 1024)}MB)`);
    });
  } else {
    console.log("No binaries downloaded yet.");
    console.log(`\nMake sure release exists: https://github.com/${WHISPER_CPP_REPO}/releases`);
  }
}

main().catch(console.error);
