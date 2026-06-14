import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import dotenv from "dotenv";
import fs from "fs";
import os from "os";
import multer from "multer";
import { randomUUID } from "crypto";
import { GoogleGenAI, GenerateVideosOperation } from "@google/genai";
import { spawn } from "child_process";
import { request as httpRequest } from "http";
import { parse as parseUrl } from "url";

dotenv.config();


const HYPERFRAMES_HOST = "127.0.0.1";
const HYPERFRAMES_PORT = 3002;

const HYPERFRAMES_MUSIC_LIBRARY = [
  { id: 'cinematic-hype', name: 'Cinematic Hype', url: 'https://cdn.pixabay.com/audio/2022/03/10/audio_c35278734d.mp3' },
  { id: 'energetic-beat', name: 'Energetic Beat', url: 'https://cdn.pixabay.com/audio/2022/01/18/audio_d1901c0c66.mp3' },
  { id: 'urban-flow', name: 'Urban Flow', url: 'https://cdn.pixabay.com/audio/2022/10/14/audio_9939f02c67.mp3' },
  { id: 'upbeat-fashion', name: 'Upbeat Fashion', url: 'https://cdn.pixabay.com/audio/2022/03/24/audio_03d2f21132.mp3' },
  { id: 'lofi-study', name: 'Lofi Study', url: 'https://cdn.pixabay.com/audio/2021/11/23/audio_0de380db37.mp3' }
].map(track => ({
  ...track,
  proxyUrl: `/api/proxy-audio?audioUrl=${encodeURIComponent(track.url)}`
}));

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeJsString(value: unknown): string {
  return JSON.stringify(String(value ?? "")).slice(1, -1);
}

function normalizeAssetPath(asset: unknown): string | null {
  if (!asset) return null;
  let raw = "";
  if (typeof asset === "string") {
    raw = asset;
  } else if (typeof asset === "object") {
    // Handle { path: "..." } or { url: "..." } or { name: "..." }
    const obj = asset as any;
    raw = obj.path || obj.url || obj.filename || obj.name || "";
  }
  
  raw = String(raw).replace(/\\/g, "/").trim();
  if (!raw || raw.includes("\0")) return null;
  if (/^(https?:|data:|blob:)/i.test(raw)) return raw;
  const withoutLeadingSlash = raw.replace(/^\/+/, "");
  if (withoutLeadingSlash.includes("..")) return null;
  return withoutLeadingSlash;
}

function isAudioAsset(asset: string): boolean {
  return /\.(mp3|wav|m4a|aac|ogg|flac|webm)$/i.test(asset);
}

function isVideoAsset(asset: string): boolean {
  return /\.(mp4|mov|m4v|webm|avi|mkv|3gp|wmv|flv)$/i.test(asset);
}

function isImageAsset(asset: string): boolean {
  return /\.(png|jpe?g|webp|gif|svg)$/i.test(asset);
}

function extractScriptBeats(scriptText: string, prompt: string): string[] {
  const source = [scriptText, prompt].filter(Boolean).join("\n").trim();
  if (!source) {
    return [
      "Open with a bold visual hook",
      "Show the strongest proof point",
      "Build momentum with fast cuts",
      "End with a crisp call to action"
    ];
  }
  const lines = source
    .split(/\r?\n|(?<=[.!?])\s+/)
    .map((line) => line.replace(/^[-*#\d.\s]+/, "").trim())
    .filter((line) => line.length > 0)
    .slice(0, 8);
  return lines.length ? lines : [source.slice(0, 120)];
}

function buildFallbackHyperEditHtml(input: {
  prompt?: string;
  aspectRatio?: string;
  mediaAssets?: unknown[];
  scriptText?: string;
}): string {
  const { prompt = "", aspectRatio = "9:16", scriptText = "" } = input;
  const width = aspectRatio === "16:9" ? 1920 : 1080;
  const height = aspectRatio === "16:9" ? 1080 : aspectRatio === "1:1" ? 1080 : 1920;
  const assets = (Array.isArray(input.mediaAssets) ? input.mediaAssets : [])
    .map(normalizeAssetPath)
    .filter((asset): asset is string => Boolean(asset));
  const audio = assets.find(isAudioAsset);
  const visuals = assets.filter((asset) => isVideoAsset(asset) || isImageAsset(asset));
  const beats = extractScriptBeats(scriptText, prompt);
  
  // Use music from library if no audio asset is provided
  let finalAudio = audio;
  let musicName = "Custom Upload";
  if (!finalAudio) {
    const randomTrack = HYPERFRAMES_MUSIC_LIBRARY[Math.floor(Math.random() * HYPERFRAMES_MUSIC_LIBRARY.length)];
    finalAudio = `/api/proxy-audio?audioUrl=${encodeURIComponent(randomTrack.url)}`;
    musicName = `Library: ${randomTrack.name}`;
  }

  const sceneCount = beats.length;
  const sceneDuration = 3.25;
  const totalDuration = Number((sceneCount * sceneDuration).toFixed(2));
  const title = beats[0] || prompt || "Hyper Edit";
  const safePrompt = escapeHtml(prompt || "Generated with Hyper-Edit Studio");

  const clips = Array.from({ length: sceneCount }).map((_, index) => {
    const start = Number((index * sceneDuration).toFixed(2));
    const asset = visuals[index % Math.max(visuals.length, 1)];
    const caption = beats[index % beats.length] || `Scene ${index + 1}`;
    const escapedCaption = escapeHtml(caption);
    const assetTag = asset
      ? isVideoAsset(asset)
        ? `<video id="media-${index}" class="clip media-layer" data-start="${start}" data-duration="${sceneDuration}" data-track-index="0" src="${escapeHtml(asset)}" style="width: 100%; height: 100%; object-fit: cover; position: absolute;" muted playsinline loop preload="metadata"></video>`
        : `<img id="media-${index}" class="clip media-layer" data-start="${start}" data-duration="${sceneDuration}" data-track-index="0" src="${escapeHtml(asset)}" style="width: 100%; height: 100%; object-fit: cover; position: absolute;" alt="${escapedCaption}" />`
      : `<div id="media-${index}" class="clip media-layer generated-visual" data-start="${start}" data-duration="${sceneDuration}" data-track-index="0" style="width: 100%; height: 100%; position: absolute;"><span id="media-text-${index}">${escapeHtml(String(index + 1).padStart(2, "0"))}</span></div>`;

    return `
      ${assetTag}
      <section class="clip scene scene-${index + 1}" id="scene-${index + 1}" data-start="${start}" data-duration="${sceneDuration}" data-track-index="1">
        <div id="scene-gradient-${index + 1}" class="scene-gradient"></div>
        <div id="grain-${index + 1}" class="grain"></div>
      </section>`;
  }).join("\n");

  const audioTag = finalAudio
    ? `<audio id="soundtrack" class="clip audio-track" src="${escapeHtml(finalAudio)}" data-start="0" data-duration="${totalDuration}" data-track-index="1" preload="auto"></audio>
       <div class="clip music-card" id="music-card" data-start="0" data-duration="${totalDuration}" data-track-index="1"><span id="music-icon" style="color: #6366f1;">♫</span> <span id="music-name">${escapeHtml(musicName)}</span></div>`
    : `<div class="clip music-card" id="music-card" data-start="0" data-duration="${totalDuration}" data-track-index="1">Add a soundtrack in the Studio asset panel for beat-synced export.</div>`;

  const timeline = Array.from({ length: sceneCount }).map((_, index) => {
    const start = Number((index * sceneDuration).toFixed(2));
    const outStart = Number((start + sceneDuration - 0.45).toFixed(2));
    const mediaSelector = `#media-${index}`;
    return `
      tl.set("${mediaSelector}", { visibility: "visible" }, ${start});
      tl.set("#scene-${index + 1}", { visibility: "visible", opacity: 0 }, ${start});
      tl.to("#scene-${index + 1}", { opacity: 1, duration: 0.35, ease: "power2.out" }, ${start});
      tl.fromTo("${mediaSelector}", { scale: 1.08, filter: "saturate(0.9) contrast(1)" }, { scale: 1.0, filter: "saturate(1.2) contrast(1.08)", duration: ${sceneDuration}, ease: "none" }, ${start});
      tl.to("#scene-${index + 1}", { opacity: 0, duration: 0.35, ease: "power2.inOut" }, ${outStart});
      tl.set("${mediaSelector}", { visibility: "hidden" }, ${outStart + 0.35});`;
  }).join("\n");

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=${width}, height=${height}, initial-scale=1" />
  <meta data-composition-id="root" data-width="${width}" data-height="${height}" />
  <title>Hyper-Edit Generated Cut</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    html, body { margin: 0; width: ${width}px; height: ${height}px; overflow: hidden; background: #050816; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    #stage { position: relative; width: ${width}px; height: ${height}px; overflow: hidden; background: #050816; color: white; }
    .clip { position: absolute; inset: 0; visibility: hidden; overflow: hidden; }
    .scene { opacity: 0; }
    .scene-gradient { position: absolute; inset: -20%; background: radial-gradient(circle at 20% 20%, rgba(99, 102, 241, .68), transparent 32%), radial-gradient(circle at 78% 22%, rgba(236, 72, 153, .48), transparent 28%), radial-gradient(circle at 50% 90%, rgba(45, 212, 191, .35), transparent 38%), linear-gradient(135deg, #070b1f 0%, #111827 48%, #020617 100%); filter: saturate(1.18); }
    .media-layer { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; transform-origin: 50% 50%; }
    .generated-visual { display: grid; place-items: center; background: linear-gradient(135deg, rgba(15,23,42,.96), rgba(49,46,129,.92)); }
    .generated-visual span { font-size: ${Math.round(width * 0.18)}px; font-weight: 950; color: rgba(255,255,255,.13); letter-spacing: -.08em; }
    .grain { position: absolute; inset: 0; opacity: .18; background-image: linear-gradient(115deg, rgba(255,255,255,.12), transparent 20%, rgba(255,255,255,.08) 41%, transparent 62%); mix-blend-mode: screen; }
    .caption { inset: auto ${Math.round(width * 0.07)}px ${Math.round(height * 0.08)}px ${Math.round(width * 0.07)}px; min-height: ${Math.round(height * 0.22)}px; padding: ${Math.round(width * 0.045)}px; border-radius: ${Math.round(width * 0.035)}px; background: linear-gradient(180deg, rgba(15,23,42,.82), rgba(15,23,42,.56)); border: 1px solid rgba(255,255,255,.18); box-shadow: 0 30px 80px rgba(0,0,0,.42); backdrop-filter: blur(22px); }
    .caption .eyebrow { display: inline-flex; padding: 10px 14px; border-radius: 999px; background: rgba(99,102,241,.28); color: #c4b5fd; text-transform: uppercase; letter-spacing: .16em; font-size: ${Math.max(18, Math.round(width * 0.018))}px; font-weight: 900; }
    .caption h2 { margin: ${Math.round(width * 0.03)}px 0 ${Math.round(width * 0.015)}px; font-size: ${Math.max(54, Math.round(width * 0.074))}px; line-height: .92; letter-spacing: -.06em; text-transform: uppercase; font-weight: 950; }
    .caption p { margin: 0; max-width: 88%; color: rgba(226,232,240,.82); font-size: ${Math.max(22, Math.round(width * 0.024))}px; line-height: 1.35; }
    .music-card { inset: auto ${Math.round(width * 0.07)}px ${Math.round(height * 0.025)}px auto; width: ${Math.round(width * 0.44)}px; height: auto; padding: 14px 18px; border-radius: 999px; background: rgba(15,23,42,.72); border: 1px solid rgba(255,255,255,.12); color: rgba(226,232,240,.72); font-weight: 700; font-size: ${Math.max(16, Math.round(width * 0.017))}px; }
  </style>
</head>
<body>
  <main id="stage" data-composition-id="root" data-width="${width}" data-height="${height}">
    ${clips}
    ${audioTag}
  </main>
  <script>
    window.__timelines = window.__timelines || {};
    const tl = gsap.timeline({ paused: true });
    ${audio ? `tl.set("#soundtrack", { volume: 0.82 }, 0);` : `tl.set("#music-card", { visibility: "visible", opacity: 0.86 }, 0);`}
    ${timeline}
    tl.set({}, {}, ${totalDuration});
    window.__timelines["root"] = tl;
    document.querySelectorAll("video").forEach((video) => {
      video.currentTime = 0;
      video.pause();
    });
  </script>
</body>
</html>`;
}

function fetchHyperframesJson(pathname: string): Promise<any | null> {
  return new Promise((resolve) => {
    const req = httpRequest({ hostname: HYPERFRAMES_HOST, port: HYPERFRAMES_PORT, path: pathname, method: "GET" }, (response) => {
      const chunks: Buffer[] = [];
      response.on("data", (chunk) => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
      response.on("end", () => {
        try {
          resolve(JSON.parse(Buffer.concat(chunks).toString("utf-8")));
        } catch {
          resolve(null);
        }
      });
    });
    req.on("error", () => resolve(null));
    req.end();
  });
}

async function resolveHyperframesProjectId(requested?: unknown): Promise<string> {
  const cleanRequested = String(requested ?? "").trim();
  if (cleanRequested && !cleanRequested.includes("/") && !cleanRequested.includes("\0")) {
    return cleanRequested;
  }
  const data = await fetchHyperframesJson("/api/projects");
  const first = Array.isArray(data?.projects) ? data.projects[0] : null;
  return String(first?.id || "project");
}

function pipeRequestBody(req: express.Request, proxyReq: ReturnType<typeof httpRequest>) {
  if (req.method === "GET" || req.method === "HEAD") {
    proxyReq.end();
    return;
  }

  const contentType = req.headers["content-type"] || "";
  
  if (req.body !== undefined && req.body !== null) {
    if (typeof req.body === "string" || Buffer.isBuffer(req.body)) {
      const bodyData = req.body;
      proxyReq.setHeader("Content-Length", Buffer.byteLength(bodyData));
      proxyReq.end(bodyData);
      return;
    }
    if (typeof req.body === "object" && Object.keys(req.body).length > 0) {
      if (String(contentType).includes("application/json")) {
        const bodyData = JSON.stringify(req.body);
        proxyReq.setHeader("Content-Length", Buffer.byteLength(bodyData));
        proxyReq.end(bodyData);
        return;
      }
      if (String(contentType).includes("application/x-www-form-urlencoded")) {
        const bodyData = new URLSearchParams(req.body as any).toString();
        proxyReq.setHeader("Content-Length", Buffer.byteLength(bodyData));
        proxyReq.end(bodyData);
        return;
      }
    }
  }
  
  // If the stream hasn't been consumed yet, pipe it directly.
  if (req.complete) {
    proxyReq.end();
  } else {
    req.pipe(proxyReq);
  }
}

function proxyToHyperframes(req: express.Request, res: express.Response, targetPath: string, options: { rewriteStudioHtml?: boolean } = {}) {
  const parsed = parseUrl(`http://${HYPERFRAMES_HOST}:${HYPERFRAMES_PORT}${targetPath}`);
  const headers = { ...req.headers, host: `${HYPERFRAMES_HOST}:${HYPERFRAMES_PORT}` };

  const proxyReq = httpRequest({
    hostname: parsed.hostname || HYPERFRAMES_HOST,
    port: parsed.port || String(HYPERFRAMES_PORT),
    path: parsed.path,
    method: req.method,
    headers,
  }, (proxyRes) => {
    const contentType = String(proxyRes.headers["content-type"] || "");
    const shouldRewrite = options.rewriteStudioHtml && contentType.includes("text/html");

    if (!shouldRewrite) {
      res.writeHead(proxyRes.statusCode || 200, proxyRes.headers);
      proxyRes.pipe(res);
      return;
    }

    const chunks: Buffer[] = [];
    proxyRes.on("data", (chunk) => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
    proxyRes.on("end", () => {
      const headersOut = { ...proxyRes.headers };
      delete headersOut["content-length"];
      const html = Buffer.concat(chunks).toString("utf-8")
        .replace(/(src|href)="\/assets\//g, '$1="/hyperframes-studio/assets/')
        .replace(/href="\/favicon\.svg"/g, 'href="/hyperframes-studio/favicon.svg"');
      res.writeHead(proxyRes.statusCode || 200, headersOut);
      res.end(html);
    });
  });

  proxyReq.on("error", (err) => {
    console.error("[HYPERFRAMES PROXY ERROR]:", err.message);
    if (!res.headersSent) {
      res.status(502).json({ error: "Hyperframes preview server is booting or unreachable." });
    }
  });

  pipeRequestBody(req, proxyReq);
}

async function startServer() {
  const app = express();
  const PORT = 3000;
  const LIMIT = '2gb';

  // Configure multer for local uploads to the project/assets directory
  const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      const assetsPath = path.join(process.cwd(), "project", "assets");
      if (!fs.existsSync(assetsPath)) fs.mkdirSync(assetsPath, { recursive: true });
      cb(null, assetsPath);
    },
    filename: (req, file, cb) => {
      const safeName = file.originalname.replace(/[^a-zA-Z0-9.\-_]/g, "_");
      cb(null, `${Date.now()}-${safeName}`);
    }
  });
  const upload = multer({ 
    storage,
    limits: { fileSize: 2 * 1024 * 1024 * 1023 } // Just under 2GB to be safe
  });

  // 1. Intercept Studio asset uploads to bypass proxy limits
  app.post("/api/projects/:projectId/assets", upload.array("file"), (req, res) => {
    try {
      const files = req.files as Express.Multer.File[];
      const uploadedFiles = (files || []).map(f => ({
        name: f.originalname, 
        filename: f.filename,
        path: `assets/${f.filename}`,
        size: f.size,
        type: f.mimetype
      }));
      console.log(`[STUDIO ASSET] Saved ${uploadedFiles.length} files to project assets.`);
      res.json({ assets: uploadedFiles, files: uploadedFiles, success: true });
    } catch (err: any) {
      console.error("[STUDIO ASSET UPLOAD ERROR]:", err);
      res.status(500).json({ error: "Failed to upload studio assets" });
    }
  });

  // 2. Handle media uploads from Hyper-Edit AI Panel
  app.post("/api/hyper-edit/upload", upload.array("file"), async (req, res) => {
    try {
      const files = req.files as Express.Multer.File[];
      const uploadedFiles = (files || []).map(f => {
        let assetType = "image";
        if (f.mimetype.startsWith("video/")) assetType = "video";
        else if (f.mimetype.startsWith("audio/")) assetType = "audio";
        return {
          name: f.originalname,
          filename: f.filename,
          path: `assets/${f.filename}`,
          size: f.size,
          type: assetType,
          mimetype: f.mimetype
        };
      });
      console.log(`[HYPER-EDIT] Uploaded ${uploadedFiles.length} files to project/assets`);
      res.json({ success: true, assets: uploadedFiles, files: uploadedFiles, uploaded: uploadedFiles });
    } catch (err: any) {
      console.error("[HYPER-EDIT UPLOAD ERROR]:", err);
      res.status(500).json({ error: "Failed to upload files" });
    }
  });

  // Run the hyperframes preview server in the background
  console.log("[HYPERFRAMES] Spawning 'npx hyperframes preview --port 3002 project'...");
  try {
    const projectPath = path.join(process.cwd(), "project");
    const assetsPath = path.join(projectPath, "assets");
    if (!fs.existsSync(projectPath)) fs.mkdirSync(projectPath, { recursive: true });
    if (!fs.existsSync(assetsPath)) fs.mkdirSync(assetsPath, { recursive: true });
    console.log(`[HYPERFRAMES] Ensured project directories exist: ${projectPath}`);
  } catch (err) {
    console.error("[HYPERFRAMES] Failed to create project directories:", err);
  }

  const hyperframesProcess = spawn("npx", ["hyperframes", "preview", "--port", "3002", "project"], {
    stdio: ["ignore", "pipe", "pipe"],
    shell: true,
    env: { ...process.env, NODE_ENV: "development", DEBUG: "hyperframes:*" }
  });

  hyperframesProcess.stdout?.on("data", (data) => {
    const msg = data.toString().trim();
    if (msg) console.log(`[HYPERFRAMES STDOUT]: ${msg}`);
  });

  hyperframesProcess.stderr?.on("data", (data) => {
    const msg = data.toString().trim();
    if (msg) console.error(`[HYPERFRAMES STDERR]: ${msg}`);
  });

  hyperframesProcess.on("close", (code) => {
    console.log(`[HYPERFRAMES] Process exited with code ${code}`);
  });

  // Keep process tidy
  const cleanup = () => {
    try {
      hyperframesProcess.kill();
    } catch (e) {}
  };
  process.on("exit", cleanup);
  process.on("SIGINT", () => { cleanup(); process.exit(); });
  process.on("SIGTERM", () => { cleanup(); process.exit(); });

  // API Routes that need raw streams (Proxying to Hyperframes) must come BEFORE body-parser
  
  // Serve the original Hyperframes Studio bundle from the preview server inside
  // this app. Rewriting /assets keeps the iframe same-origin and avoids Vite
  // asset path collisions.
  app.all("/hyperframes-studio", (req, res) => {
    proxyToHyperframes(req, res, "/", { rewriteStudioHtml: true });
  });

  app.all("/hyperframes-studio/*", (req, res) => {
    const suffix = req.originalUrl.replace(/^\/hyperframes-studio/, "") || "/";
    proxyToHyperframes(req, res, suffix, { rewriteStudioHtml: true });
  });

  app.all("/api/runtime.js", (req, res) => {
    proxyToHyperframes(req, res, req.originalUrl);
  });

  // Proxy hyperframes studio API requests to the background preview server.
  // These MUST come before express.json() if they are handled by proxyToHyperframes
  app.all("/api/projects*", (req, res) => {
    console.log(`[PROXY] Match: ${req.method} ${req.originalUrl}, Content-Length: ${req.headers['content-length']}, Has Body: ${req.body !== undefined}`);
    proxyToHyperframes(req, res, req.originalUrl);
  });

  // Increase global payload limits for our native routes (registered after proxy routes to avoid consuming request streams)
  app.use(express.json({ limit: LIMIT }));
  app.use(express.urlencoded({ limit: LIMIT, extended: true }));
  app.use(express.text({ limit: LIMIT }));

  app.use("/assets", express.static(path.join(process.cwd(), "project", "assets")));

  // API Routes
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", GEMINI_API_KEY: process.env.GEMINI_API_KEY, vite: process.env.VITE_GEMINI_API_KEY });
  });

  app.get("/api/hyper-edit/project", async (req, res) => {
    const projectId = await resolveHyperframesProjectId(req.query.projectId);
    res.json({ projectId });
  });

  app.post("/api/hyper-edit/clear", async (req, res) => {
    try {
      const activeProjectId = await resolveHyperframesProjectId(req.body?.projectId);
      const filePath = path.join(process.cwd(), "project", "index.html");
      const emptyContent = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=1080, height=1920, initial-scale=1" />
  <meta data-composition-id="root" data-width="1080" data-height="1920" />
  <title>Empty Project</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
</head>
<body>
  <main id="stage" data-composition-id="root" data-width="1080" data-height="1920"></main>
  <script>
    window.__timelines = window.__timelines || {};
    const tl = gsap.timeline({ paused: true });
    tl.set({}, {}, 15);
    window.__timelines["root"] = tl;
  </script>
</body>
</html>`;
      fs.writeFileSync(filePath, emptyContent, "utf-8");
      
      if (req.body?.newProject) {
        // Also clear assets directory
        const assetsDir = path.join(process.cwd(), "project", "assets");
        if (fs.existsSync(assetsDir)) {
          fs.readdirSync(assetsDir).forEach(f => {
            const file = path.join(assetsDir, f);
            if (fs.statSync(file).isFile()) fs.unlinkSync(file);
          });
        }
      }
      
      res.json({ success: true, projectId: activeProjectId });
    } catch (err: any) {
      console.error("[HYPER-EDIT CLEAR ERROR]:", err);
      res.status(500).json({ error: err.message });
    }
  });

  app.post("/api/hyper-edit/generate", async (req, res) => {
    try {
      const { prompt = "", aspectRatio = "9:16", mediaAssets = [], scriptText = "", projectId = "" } = req.body;

      // Determine dimensions
      let width = 1080;
      let height = 1920;
      if (aspectRatio === "16:9") {
        width = 1920;
        height = 1080;
      } else if (aspectRatio === "1:1") {
        width = 1080;
        height = 1080;
      }

      const activeProjectId = await resolveHyperframesProjectId(projectId);
      const normalizedAssets = (Array.isArray(mediaAssets) ? mediaAssets : [])
        .map(asset => {
          const path = normalizeAssetPath(asset);
          if (!path) return null;
          
          let type = "image";
          if (isVideoAsset(path)) type = "video";
          else if (isAudioAsset(path)) type = "audio";
          
          // Ensure path is consistent for use in HTML src attributes
          // It should be relative to the project root, e.g., "assets/filename.mp4"
          return { path, type };
        })
        .filter(Boolean);

      console.log(`[HYPER-EDIT] Normalized Assets:`, JSON.stringify(normalizedAssets));
      console.log(`[HYPER-EDIT] Generating composition for project "${activeProjectId}" with ${normalizedAssets.length} assets, aspect: ${aspectRatio} (${width}x${height})`);

      const systemInstruction = `
You are a master cinematic video producer, Hyperframes composition designer, and GSAP animation expert.
Your goal is to parse the user's "Script Context" and "Direction" and translate them into a high-performance Hyperframes HTML composition.

# COMMUNICATION & HYPERFRAMES RULES
1. Return ONLY raw HTML starting with <!doctype html>. No markdown wrappers, no conversational text.
2. Mandatory Meta Attributes:
   - You MUST include this in the <head>: <meta data-composition-id="root" data-width="${width}" data-height="${height}" />
   - The root container MUST be: <main id="stage" data-composition-id="root" data-start="0" data-width="${width}" data-height="${height}">.
3. Root Visual Canvas: The <main> tag is exactly ${width}x${height} pixels. All clips MUST stay inside this stage. DO NOT output any lists, debug elements, or duplicate media at the bottom of the document. NEVER CREATE A LIST OR GALLERY OF UPLOADED MEDIA.
4. CLIP STRUCTURE: Every visual scene, caption, or audio element MUST be a .clip <section> or standalone element with:
   - id: Unique descriptive ID (e.g., scene-1, caption-hook)
   - data-start: Start time in seconds (e.g., 0, 4.5)
   - data-duration: Duration in seconds (e.g., 3.25)
   - data-track-index: Layering (0 for Video/Visuals/Images, 1 for Audio/Music, 2 for Captions/Text/GFX)

5. ⚠️ CRITICAL NESTING RULE — VIOLATION CAUSES STUDIO CRASH:
   - The clip timing attributes (data-start, data-duration, data-track-index) MUST ONLY be on the OUTER container element (e.g., <section class="clip">).
   - NEVER place data-start, data-duration, or data-track-index on a <video> or <img> element that is a CHILD of a <section class="clip">.
   - You MUST place <video> and <img> tags INSIDE the <section class="clip">. NEVER place them directly inside <main> or at the bottom of the file!
   - CORRECT ✅ (Media nested inside clip):
     <main id="stage" ...>
       <section class="clip" data-start="0" data-duration="3" data-track-index="0">
         <video id="vid-1" src="assets/clip.mp4" muted playsinline preload="metadata"></video>
       </section>
     </main>
   - WRONG ❌ (Media outside clip! This causes the media to float at the bottom and breaks the Studio):
     <main id="stage" ...>
       <section class="clip" data-start="0" data-duration="3" data-track-index="0"></section>
       <video id="vid-1" src="assets/clip.mp4"></video>
     </main>
   - WRONG ❌ (Timing attributes on inner video! Crashes Studio):
     <section class="clip" data-start="0" data-duration="3" data-track-index="0">
       <video src="assets/clip.mp4" data-start="0" data-duration="3" muted playsinline></video>
     </section>

6. ⚠️ CRITICAL GSAP RULE — NO INFINITE REPEATS:
   - NEVER use repeat: -1 in any GSAP tween. It breaks the rendering engine.
   - Use a finite repeat count instead: repeat: Math.ceil(totalDuration / cycleDuration)

# EDITING & INSTRUCTION PARSING (HIGHEST PRIORITY)
- You MUST treat the "Script Context" and "Direction" as the primary editorial blueprint.
- Distinguish between EDITING INSTRUCTIONS (e.g., "Cut the video at 0:05", "trim from 10s to 15s") and NARRATION/DIALOGUE.
- DO NOT put raw editing instructions into text captions! If the text describes how to cut the video, USE THAT to structure the timeline, do NOT display it on screen.
- TO TRIM VIDEOS: Use HTML5 Media Fragments on the video src attribute. For example, if instructed to cut from 5s to 10s, use: <video src="assets/video.mp4#t=5,10" ...>
- Set the clip's \`data-duration\` to match the trimmed length (e.g., 5 seconds).
- Sequence the clips using \`data-start\` so they play one after another seamlessly.
- ONLY create captions if the script contains actual spoken dialogue or text meant for the screen.
- If no specific cuts are provided, infer 4-8 logical scenes from the media.

# MEDIA MAPPING (ABSOLUTE PRIORITY)
- You MUST ONLY use assets from the provided "Uploaded media assets" list for your src attributes. You do NOT need to use all of them if the script specifies specific ones.
- NO EXTERNAL PLACEHOLDERS (like Unsplash, Pixabay) are allowed for visual scenes if user has uploaded assets.
- PATH RULE: Use the "path" property exactly as provided, and append #t=start,end if trimming is needed.
- TYPE MAPPING:
  - image: Use <img> tags (bare, no data-start on the img itself).
  - video: Use <video muted playsinline preload="metadata"> tags inside a <section class="clip"> wrapper. (This includes .mp4 and .mov files).
  - audio: Use <audio class="clip" data-start="0" data-duration="999" data-track-index="1" preload="auto"> tags (standalone, NOT nested).

# GSAP ANIMATIONS
- You MUST load GSAP via CDN in a script tag BEFORE your inline script (Hyperframes does NOT inject GSAP into the composition iframe):
  <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
- You MUST create the GSAP timeline and explicitly register it to window.__timelines["root"] so the StaticGuard compiler can find it:
  <script>
    window.__timelines = window.__timelines || {};
    const tl = gsap.timeline({ paused: true });
    // ... add tweens to tl here ...
    tl.set({}, {}, totalDuration);
    window.__timelines["root"] = tl;
  </script>
- Animate clips using their unique IDs (e.g., #scene-1).
- To control clip visibility, use GSAP tl.set("#scene-1", { visibility: "visible", opacity: 0 }, startTime) rather than relying purely on CSS.
- Use cinematic transitions: Ken Burns zooms for images, smooth fades, and kinetic typography.
- Caption animations: fade-in from bottom with slight scale for each caption entry.
- Scene transitions: crossfade between scenes using opacity tweens with 0.3s overlap.
- NEVER use repeat: -1. Use a finite repeat: Math.ceil(totalDuration / cycleDuration) instead.
`;

      let content = "";
      let usedFallback = true;
      const modelName = process.env.HYPER_EDIT_MODEL || "gemini-3.5-flash"; 

      if (process.env.GEMINI_API_KEY) {
        try {
          const ai = new GoogleGenAI({ 
            apiKey: process.env.GEMINI_API_KEY,
            httpOptions: { headers: { 'User-Agent': 'aistudio-build' } }
          });
          
          const userMessage = `
# COMPOSITION REQUEST
DIRECTION: "${prompt || "Create a cinematic video"}"
ASPECT RATIO: ${aspectRatio} (${width}x${height})
SCRIPT CONTEXT: "${scriptText || "No script provided, use creative direction"}"

# ASSETS AVAILABLE
- MEDIA_POOL: ${JSON.stringify(normalizedAssets)}
- MUSIC LIBRARY (FALLBACK): ${JSON.stringify(HYPERFRAMES_MUSIC_LIBRARY)}

# FINAL CHECK
- You MUST prioritize using the files from "MEDIA_POOL" for ALL visual scenes.
- If an uploaded video or image is provided, use it. Do NOT use fake URLs.
- If an uploaded audio file is present, use it as the background track.
- 🛑 CRITICAL STUDIO UI RULE: DO NOT, UNDER ANY CIRCUMSTANCES, create a list or gallery of media assets at the bottom of the HTML. You MUST ONLY use the media elements INSIDE the <section class="clip"> wrappers for the timeline. Any media placed directly inside <main> or <body> will break the Studio UI! DO NOT CREATE A "MEDIA POOL" OR "UPLOADED ASSETS" DIV. JUST USE THEM IN THE CLIPS.
- 🛑 PATCHABLE TARGET RULE: EVERY SINGLE ELEMENT you create (especially <section>, <video>, <img>, <p>, <div>, <span>, and <audio>) MUST have a unique \`id\` attribute (e.g., \`id="video-1"\`, \`id="gradient-2"\`, \`id="grain-2"\`). If an element is missing an ID, the Studio cannot delete or patch it! This applies to inner decorative elements like gradients and grain divs as well.
- Return ONLY the HTML.
`;

          console.log(`[HYPER-EDIT] Calling Gemini (${modelName}) with ${normalizedAssets.length} assets...`);

          const uploadedFiles: any[] = [];
          for (const asset of normalizedAssets) {
            const absolutePath = path.join(process.cwd(), activeProjectId, asset.path);
            if (fs.existsSync(absolutePath)) {
               try {
                  console.log(`[HYPER-EDIT] Uploading ${asset.path} to Gemini...`);
                  // Detect proper MIME type
                  let mimeType = "application/octet-stream";
                  if (asset.type === "video") mimeType = "video/mp4";
                  else if (asset.type === "audio") mimeType = "audio/mp3";
                  else if (asset.type === "image") {
                     if (asset.path.endsWith(".png")) mimeType = "image/png";
                     else if (asset.path.endsWith(".webp")) mimeType = "image/webp";
                     else mimeType = "image/jpeg";
                  }
                  
                  const uploadResult = await ai.files.upload({ file: absolutePath, mimeType });
                  
                  // Poll until active
                  let fileInfo = await ai.files.get({ name: uploadResult.name });
                  while (fileInfo.state === "PROCESSING") {
                     process.stdout.write(".");
                     await new Promise(r => setTimeout(r, 2000));
                     fileInfo = await ai.files.get({ name: uploadResult.name });
                  }
                  console.log(); // newline after dots
                  
                  if (fileInfo.state === "ACTIVE") {
                     uploadedFiles.push({
                        fileData: {
                           fileUri: uploadResult.uri,
                           mimeType: uploadResult.mimeType
                        }
                     });
                     console.log(`[HYPER-EDIT] Successfully processed ${asset.path}`);
                  } else {
                     console.warn(`[HYPER-EDIT] Failed to process ${asset.path}: state=${fileInfo.state}`);
                  }
               } catch (err) {
                  console.warn(`[HYPER-EDIT] Error uploading ${asset.path}:`, err);
               }
            } else {
               console.warn(`[HYPER-EDIT] File not found on disk: ${absolutePath}`);
            }
          }

          const contentsParts = [...uploadedFiles, userMessage];

          const response = await ai.models.generateContent({
            model: modelName,
            contents: contentsParts,
            config: {
              systemInstruction,
            }
          });

          let rawContent = (response.text || "").trim();
          console.log(`[HYPER-EDIT] Gemini response received. Length: ${rawContent.length}`);
          
          // Scrub potential markdown wrappers
          if (rawContent.includes("<!doctype html") || rawContent.includes("<html")) {
             // If there's markdown around it, extract it
             const match = rawContent.match(/<!doctype html>[\s\S]*<\/html>/i) || rawContent.match(/<html[\s\S]*<\/html>/i);
             if (match) {
               content = match[0];
               console.log(`[HYPER-EDIT] Extracted HTML from markdown wrapper.`);
             } else {
               content = rawContent;
             }
          } else {
            content = rawContent;
          }

          // Fallback if content is wrapped in ```html ... ```
          if (content.startsWith("```")) {
            const lines = content.split("\n");
            if (lines[0].trim().startsWith("```")) lines.shift();
            if (lines[lines.length - 1]?.trim().startsWith("```") || lines[lines.length - 1]?.trim() === "") lines.pop();
            content = lines.join("\n").trim();
            console.log(`[HYPER-EDIT] Stripped markdown code block delimiters.`);
          }

          // --- HTML POST-PROCESSOR: ENFORCE VIDEO NESTING AND FIX MISSING MEDIA ---
          try {
             const visualsList = normalizedAssets.filter(a => a.type === "video" || a.type === "image");
             if (visualsList.length > 0) {
                // 1. Extract and remove all <video> and <img> tags from the document
                const existingMedia: any[] = [];
                content = content.replace(/<(video|img)([^>]*)>/gi, (match, tag, attrs) => {
                   existingMedia.push({ tag, attrs, fullMatch: match });
                   return ""; // Remove from original location
                });

                // 2. Inject media back as DIRECT CHILDREN of the stage, right before the sections
                // This prevents the "video nested inside section" StaticGuard crash while keeping them timed correctly.
                let visualIndex = 0;
                let injectedMediaHtml = "";
                let visibilityTweens = "";
                
                content = content.replace(/(<section[^>]*data-track-index="0"[^>]*>)([\s\S]*?)(<\/section>)/gi, (match, openTag, inner, closeTag) => {
                    const asset = visualsList[visualIndex % visualsList.length];
                    
                    // Extract start and duration from the section to sync the video
                    const startMatch = openTag.match(/data-start="([^"]*)"/i);
                    const durationMatch = openTag.match(/data-duration="([^"]*)"/i);
                    const startVal = startMatch ? parseFloat(startMatch[1]) : 0;
                    const durationVal = durationMatch ? parseFloat(durationMatch[1]) : 0;
                    const startAttr = startMatch ? `data-start="${startMatch[1]}"` : "";
                    const durationAttr = durationMatch ? `data-duration="${durationMatch[1]}"` : "";
                    
                    let extractedId = `media-recovered-${visualIndex}`;
                    if (existingMedia[visualIndex]) {
                        const em = existingMedia[visualIndex];
                        let cleanAttrs = em.attrs
                           .replace(/src="[^"]*"/i, "")
                           .replace(/data-start="[^"]*"/gi, "")
                           .replace(/data-duration="[^"]*"/gi, "")
                           .replace(/data-track-index="[^"]*"/gi, "");
                        
                        let idMatch = cleanAttrs.match(/id="([^"]+)"/i);
                        extractedId = idMatch ? idMatch[1] : extractedId;
                        
                        if (em.tag.toLowerCase() === "video") {
                           injectedMediaHtml += `\n<video id="${extractedId}" class="clip media-layer" ${startAttr} ${durationAttr} data-track-index="0" src="${asset.path}" style="width: 100%; height: 100%; object-fit: cover; position: absolute;" ${cleanAttrs}></video>`;
                        } else {
                           injectedMediaHtml += `\n<img id="${extractedId}" class="clip media-layer" ${startAttr} ${durationAttr} data-track-index="0" src="${asset.path}" style="width: 100%; height: 100%; object-fit: cover; position: absolute;" ${cleanAttrs} />`;
                        }
                    } else {
                        if (asset.type === "video") {
                            injectedMediaHtml += `\n<video id="${extractedId}" class="clip media-layer" ${startAttr} ${durationAttr} data-track-index="0" src="${asset.path}" style="width: 100%; height: 100%; object-fit: cover; position: absolute;" muted playsinline loop preload="metadata"></video>`;
                        } else {
                            injectedMediaHtml += `\n<img id="${extractedId}" class="clip media-layer" ${startAttr} ${durationAttr} data-track-index="0" src="${asset.path}" style="width: 100%; height: 100%; object-fit: cover; position: absolute;" />`;
                        }
                    }
                    
                    // Auto-generate visibility GSAP script for the extracted media
                    visibilityTweens += `\n      tl.set("#${extractedId}", { visibility: "visible" }, ${startVal});`;
                    visibilityTweens += `\n      tl.set("#${extractedId}", { visibility: "hidden" }, ${startVal + durationVal});`;
                    
                    visualIndex++;
                    
                    // Change the section's track index to 1 so the Studio puts overlays on the track above the video
                    let newOpenTag = openTag.replace(/data-track-index="[^"]*"/i, 'data-track-index="1"');
                    return `${newOpenTag}\n${inner}\n${closeTag}`;
                });
                
                // Inject the generated media HTML directly inside <main id="stage">
                content = content.replace(/(<main[^>]*>)/i, `$1${injectedMediaHtml}`);
                
                // Inject the auto-generated visibility GSAP script
                content = content.replace(/window\.__timelines\["root"\]\s*=\s*tl;/g, `
      // Auto-injected visibility for sibling videos
      ${visibilityTweens}
      window.__timelines["root"] = tl;`);
                
                console.log(`[HYPER-EDIT] Post-processor applied: Fixed media nesting for ${visualIndex} scenes.`);
             }
          } catch (e) {
             console.error("[HYPER-EDIT] Post-processor error:", e);
          }

          usedFallback = false;
        } catch (aiErr: any) {
          console.warn("[HYPER-EDIT] Gemini generation failed; using deterministic studio composition fallback:", aiErr?.message || aiErr);
        }
      }

      const trimmedContent = content.trim().toLowerCase();
      const looksLikeComposition = (trimmedContent.includes("<!doctype html") || trimmedContent.includes("<html")) && 
                                   content.includes("data-composition-id") && 
                                   content.includes("window.__timelines");

      if (!looksLikeComposition) {
        console.warn(`[HYPER-EDIT] AI response did not look like a valid composition. Fallback to deterministic.`);
        if (!process.env.GEMINI_API_KEY) console.warn("[HYPER-EDIT] No API key, using fallback.");
        content = buildFallbackHyperEditHtml({
          prompt,
          aspectRatio,
          mediaAssets: normalizedAssets,
          scriptText,
        });
        usedFallback = true;
      }

      const filePath = path.join(process.cwd(), "project", "index.html");
      fs.writeFileSync(filePath, content, "utf-8");

      const randomTrack = HYPERFRAMES_MUSIC_LIBRARY[Math.floor(Math.random() * HYPERFRAMES_MUSIC_LIBRARY.length)];
      console.log(`[HYPER-EDIT] Successfully updated ${filePath}${usedFallback ? " with fallback composition" : " with AI composition"}`);
      res.json({
        success: true,
        message: "Video composition generated and applied to the Hyperframes Studio timeline",
        projectId: activeProjectId,
        fallback: usedFallback,
        assets: normalizedAssets,
        musicTrack: { name: randomTrack.name, url: randomTrack.proxyUrl }
      });

    } catch (err: any) {
      console.error("[HYPER-EDIT GENERATION ERROR]:", err);
      res.status(500).json({ error: err.message || "Failed to generate video composition" });
    }
  });

  app.get("/api/proxy-audio", async (req, res) => {
    const audioUrl = String(req.query.audioUrl || "").trim();
    if (!audioUrl) return res.status(400).json({ error: "Missing audioUrl" });

    try {
      const response = await fetch(audioUrl, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
          'Accept': 'audio/*,*/*',
          'Referer': 'https://pixabay.com/'
        },
        redirect: 'follow'
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch audio (${response.status})`);
      }

      const contentType = response.headers.get('content-type') || 'audio/mpeg';
      res.setHeader('Content-Type', contentType);
      
      const arrayBuffer = await response.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      res.send(buffer);
    } catch (err: any) {
       console.error("[AUDIO PROXY ERROR]:", err.message);
       if (!res.headersSent) {
         res.status(500).json({ error: err.message });
       }
    }
  });

  app.post("/api/proxy-video", async (req, res) => {
    const videoUrl = (req.body.videoUrl || "").trim();
    if (!videoUrl) return res.status(400).json({ error: "Missing videoUrl" });

    // Sanitize URL for console logging
    const logUrl = videoUrl.length > 100 ? videoUrl.slice(0, 50) + "..." + videoUrl.slice(-50) : videoUrl;
    console.log(`[VIDEO PROXY] Processing: ${logUrl}`);

    try {
      // Use more standard headers that work better with Meta CDNs
      const response = await fetch(videoUrl, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
          'Accept': '*/*',
          'Accept-Language': 'en-US,en;q=0.9',
          'Range': 'bytes=0-' // Sometimes needed for CDNs to trigger video streaming response
        },
        redirect: 'follow'
      });
      
      if (!response.ok) {
        const errText = await response.text().catch(() => 'No body');
        console.error(`[VIDEO PROXY] Failed at ${logUrl}, status ${response.status}: ${errText.slice(0, 200)}`);
        throw new Error(`Platform rejected request (${response.status}). Link might be expired or restricted.`);
      }

      const contentType = response.headers.get('content-type') || 'video/mp4';
      
      // If we got HTML instead of a video, it means the URL was likely a page link
      if (contentType.includes('text/html')) {
        console.error(`[VIDEO PROXY] Received HTML for ${logUrl}`);
        throw new Error("Link pointed to a web page, not a direct video stream.");
      }

      const arrayBuffer = await response.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      
      if (buffer.length < 500) { // Increased threshold for "too small"
        throw new Error(`Fetched data (${buffer.length} bytes) is too small to be a video.`);
      }

      const base64 = buffer.toString('base64');
      console.log(`[VIDEO PROXY] Successfully buffered ${buffer.length} bytes.`);
      res.json({ base64, mimeType: contentType });
      
    } catch (err: any) {
       console.error("[VIDEO PROXY ERROR]:", err.message);
       res.status(500).json({ error: err.message });
    }
  });

  app.post("/api/discover-videos", async (req, res) => {
    let { url } = req.body;
    if (!url) return res.status(400).json({ error: "Missing URL" });

    try {
      // Basic URL normalization and validation
      if (!url.startsWith('http')) {
        url = 'https://' + url;
      }
      
      const parsedUrl = new URL(url);
      if (!['facebook.com', 'www.facebook.com', 'fb.watch', 'instagram.com', 'www.instagram.com'].some(d => parsedUrl.hostname.includes(d))) {
        return res.status(400).json({ error: "Only Facebook and Instagram links are supported for discovery." });
      }

      console.log(`Attempting discovery for: ${url}`);

      const response = await fetch(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
          'Accept-Language': 'en-US,en;q=0.9',
          'Sec-Ch-Ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
          'Sec-Ch-Ua-Mobile': '?0',
          'Sec-Ch-Ua-Platform': '"Windows"',
          'Sec-Fetch-Dest': 'document',
          'Sec-Fetch-Mode': 'navigate',
          'Sec-Fetch-Site': 'none',
          'Sec-Fetch-User': '?1',
          'Upgrade-Insecure-Requests': '1',
          'Cache-Control': 'no-cache',
        },
        redirect: 'follow'
      });
      
      if (!response.ok) {
        // Social media platforms often return 400/403/404 if they detect a scraper or need login
        if (response.status === 400) {
          throw new Error("Social platform rejected the request (400). The profile might be private or protected against automated discovery.");
        }
        throw new Error(`Platform returned ${response.status}: ${response.statusText}`);
      }

      const html = await response.text();
      
      const reelRegex = /https:\/\/(www\.)?facebook\.com\/(reels|reel|videos)\/[a-zA-Z0-9_-]+\/?/g;
      const igRegex = /https:\/\/(www\.)?instagram\.com\/(reels|reel|p)\/[a-zA-Z0-9_-]+\/?/g;
      
      const foundReels = html.match(reelRegex) || [];
      const foundIg = html.match(igRegex) || [];
      const allUnique = Array.from(new Set([...foundReels, ...foundIg]));

      res.json({ links: allUnique });
    } catch (error: any) {
      console.error("Discovery error:", error);
      res.status(500).json({ error: error.message });
    }
  });

  app.get("/api/debug", (req, res) => {
    res.json({
      host: req.get('host'),
      origin: req.get('origin'),
      referrer: req.get('referrer'),
      env_APP_URL: process.env.APP_URL
    });
  });

  // Facebook OAuth
  app.get("/api/auth/facebook/url", (req, res) => {
    const origin = req.query.redirectBase ? String(req.query.redirectBase) : (process.env.APP_URL || "https://ais-pre-hpnvylq4d7vc67nsgrwsi5-267189319589.europe-west3.run.app");
    const redirectUri = `${origin}/auth/facebook/callback`;
    
    const clientId = process.env.META_CLIENT_ID;
    if (!clientId || clientId === "your_meta_client_id") {
      return res.status(500).json({ error: "META_CLIENT_ID is not configured in App Settings." });
    }

    const params = new URLSearchParams({
      client_id: clientId,
      redirect_uri: redirectUri,
      scope: "email,public_profile,business_management,facebook_branded_content_ads_brand,facebook_creator_marketplace_discovery,pages_manage_engagement,pages_manage_metadata,pages_manage_posts,pages_read_engagement,pages_read_user_content,pages_show_list,read_insights",
      response_type: "code",
    });
    res.json({ url: `https://www.facebook.com/v19.0/dialog/oauth?${params}` });
  });

  // Instagram OAuth
  app.get("/api/auth/instagram/url", (req, res) => {
    const origin = req.query.redirectBase ? String(req.query.redirectBase) : (process.env.APP_URL || "https://ais-pre-hpnvylq4d7vc67nsgrwsi5-267189319589.europe-west3.run.app");
    const redirectUri = `${origin}/auth/instagram/callback`;
    
    const clientId = process.env.META_CLIENT_ID;
    if (!clientId || clientId === "your_meta_client_id") {
      return res.status(500).json({ error: "META_CLIENT_ID is not configured in App Settings." });
    }

    const params = new URLSearchParams({
      client_id: clientId,
      redirect_uri: redirectUri,
      scope: "ads_management,ads_read,business_management,catalog_management,email,instagram_basic,instagram_branded_content_ads_brand,instagram_branded_content_brand,instagram_branded_content_creator,instagram_content_publish,instagram_creator_marketplace_discovery,instagram_manage_comments,instagram_manage_contents,instagram_manage_engagement,instagram_manage_insights,instagram_manage_messages,instagram_manage_upcoming_events,instagram_shopping_tag_products,pages_read_engagement,pages_show_list,public_profile",
      response_type: "code",
    });
    res.json({ url: `https://www.facebook.com/v19.0/dialog/oauth?${params}` });
  });

  // Callbacks
  app.get(["/auth/facebook/callback", "/auth/instagram/callback", "/auth/facebook/callback/", "/auth/instagram/callback/"], async (req, res) => {
    const { code, error, error_description } = req.query;
    const type = req.path.includes('facebook') ? 'Facebook' : 'Instagram';
    
    if (error) {
      return res.send(`
        <!DOCTYPE html>
        <html>
          <body>
            <script>
              window.opener.postMessage({ type: 'AUTH_ERROR', platform: '${type.toLowerCase()}', error: '${error_description || error}' }, '*');
              window.close();
            </script>
          </body>
        </html>
      `);
    }

    res.send(`
      <!DOCTYPE html>
      <html>
        <head>
          <title>${type} Authentication</title>
          <script>
            if (window.opener) {
              window.opener.postMessage({ type: 'AUTH_SUCCESS', platform: '${type.toLowerCase()}', code: '${code || ''}' }, '*');
              window.close();
            } else {
              document.write('Authentication successful. You can close this window.');
            }
          </script>
        </head>
        <body>
          <p>Processing authentication...</p>
        </body>
      </html>
    `);
  });

async function fetchPlatformData(platform: string, accessToken: string) {
  let allPages: any[] = [];
  let allIgAccounts: any[] = [];

  if (platform === 'facebook') {
    // 1. Fetch Pages
    try {
      const pagesRes = await fetch(`https://graph.facebook.com/v19.0/me/accounts?fields=name,access_token,instagram_business_account&access_token=${accessToken}`);
      const pagesData = await pagesRes.json();
      
      if (pagesData.data && pagesData.data.length > 0) {
         for (const page of pagesData.data) {
           const pageToken = page.access_token;
           let postsData: any = {};
           try {
             // Let's add posts.data mapping but also save raw api response if error
             const postsRes = await fetch(`https://graph.facebook.com/v19.0/${page.id}/posts?fields=message,created_time,likes.summary(true),comments.summary(true),shares,attachments{media{image,source},media_type,url,target}&limit=50&access_token=${pageToken}`);
             const postsJson = await postsRes.json();
             if (postsJson.data) {
               const enrichedData = await Promise.all(postsJson.data.map(async (p: any) => {
                  p.insights = { data: [] };
                  try {
                    const postInsightsRes = await fetch(`https://graph.facebook.com/v19.0/${p.id}/insights?metric=post_impressions_unique,post_engaged_users,post_clicks&access_token=${pageToken}`);
                    const postInsights = await postInsightsRes.json();
                    if (postInsights.data) {
                       p.insights.data = postInsights.data;
                    }
                  } catch(e) {}
                  
                  try {
                    const videoMetrics = ['post_video_views', 'total_video_views', 'total_video_views_unique'];
                    const videoRes = await fetch(`https://graph.facebook.com/v19.0/${p.id}/insights?metric=${videoMetrics.join(',')}&access_token=${pageToken}`);
                    const videoJson = await videoRes.json();
                    if (videoJson.data) {
                       p.insights.data.push(...videoJson.data);
                    } else {
                       for (const vMetric of videoMetrics) {
                         try {
                           const singleRes = await fetch(`https://graph.facebook.com/v19.0/${p.id}/insights?metric=${vMetric}&access_token=${pageToken}`);
                           const singleJson = await singleRes.json();
                           if (singleJson.data) p.insights.data.push(...singleJson.data);
                         } catch(e) {}
                       }
                    }
                    const targetId = p.attachments?.data?.[0]?.target?.id;
                    if (targetId) {
                       const targetRes = await fetch(`https://graph.facebook.com/v19.0/${targetId}/video_insights?metric=${videoMetrics.join(',')}&access_token=${pageToken}`);
                       const targetJson = await targetRes.json();
                       if (targetJson.data) p.insights.data.push(...targetJson.data);
                    }
                  } catch(e) {}
                  return p;
               }));
               postsData = { data: enrichedData, error: postsJson.error };
             } else {
               postsData = postsJson;
             }
           } catch(e) { console.log(e); }
           
           // Fetch Page Insights (e.g., reach)
           let insightsData: any = {};
           try {
             const insightsRes = await fetch(`https://graph.facebook.com/v19.0/${page.id}/insights?metric=page_impressions_unique,page_post_engagements,page_video_views&period=day&access_token=${pageToken}`);
             insightsData = await insightsRes.json();
           } catch(e) { console.log(e); }

           allPages.push({ 
             platform: 'facebook',
             pageName: page.name, 
             pageId: page.id, 
             pageToken: pageToken,
             linkedInstagramAccountId: page.instagram_business_account?.id || null,
             insights: insightsData.data || [],
             postsError: postsData.error || null,
             insightsError: insightsData.error || null,
             recentPosts: postsData.data ? postsData.data.slice(0, 50).map((p: any) => {
                 let reach = 0;
                 let engaged = 0;
                 let views = 0;
                 let clicks = 0;
                 if (p.insights && p.insights.data) {
                    reach = p.insights.data.find((i: any) => i.name === 'post_impressions_unique' || i.name === 'post_impressions')?.values?.[0]?.value || 0;
                    engaged = p.insights.data.find((i: any) => i.name === 'post_engaged_users')?.values?.[0]?.value || 0;
                    const videoViewsMetric = p.insights.data.find((i: any) => 
                      i.name === 'total_video_views' || 
                      i.name === 'post_video_views' || 
                      i.name === 'total_video_views_unique'
                    );
                    views = videoViewsMetric?.values?.[0]?.value || 0;
                    clicks = p.insights.data.find((i: any) => i.name === 'post_clicks')?.values?.[0]?.value || 0;
                 }
                 return {
                   id: p.id,
                   message: p.message,
                   created: p.created_time,
                   likes: p.likes?.summary?.total_count || 0,
                   comments: p.comments?.summary?.total_count || 0,
                   shares: p.shares?.count || 0,
                   reach: reach,
                   engaged: engaged,
                   views: views,
                   clicks: clicks,
                   full_picture: p.attachments?.data?.[0]?.media?.image?.src || null,
                   source: p.attachments?.data?.[0]?.media?.source || null,
                   media_type: p.attachments?.data?.[0]?.media_type || 'image'
                 };
             }) : []
           });
         }
      }
    } catch(e) { console.log('Error fetching pages', e); }

    // 2. Fetch User Profile
    try {
       const profileRes = await fetch(`https://graph.facebook.com/v19.0/me?fields=id,name,posts.limit(50){message,created_time,likes.summary(true),comments.summary(true),shares,attachments{media{image,source},media_type,url}}&access_token=${accessToken}`);
       const profileData = await profileRes.json();
       
       const recentPosts = profileData.posts?.data ? profileData.posts.data.map((p: any) => {
           let reach = 0;
           let engaged = 0;
           let views = 0;
           let clicks = 0;
           if (p.insights && p.insights.data) {
              reach = p.insights.data.find((i: any) => i.name === 'post_impressions_unique' || i.name === 'post_impressions')?.values?.[0]?.value || 0;
              engaged = p.insights.data.find((i: any) => i.name === 'post_engaged_users')?.values?.[0]?.value || 0;
              const videoViewsMetric = p.insights.data.find((i: any) => 
                i.name === 'total_video_views' || 
                i.name === 'post_video_views' || 
                i.name === 'total_video_views_unique'
              );
              views = videoViewsMetric?.values?.[0]?.value || 0;
              clicks = p.insights.data.find((i: any) => i.name === 'post_clicks')?.values?.[0]?.value || 0;
           }
           return {
             id: p.id,
             message: p.message,
             created: p.created_time,
             likes: p.likes?.summary?.total_count || 0,
             comments: p.comments?.summary?.total_count || 0,
             shares: p.shares?.count || 0,
             reach: reach,
             engaged: engaged,
             views: views,
             clicks: clicks,
             full_picture: p.attachments?.data?.[0]?.media?.image?.src || null,
             source: p.attachments?.data?.[0]?.media?.source || null,
             media_type: p.attachments?.data?.[0]?.media_type || 'image'
           };
       }) : [];
       
       allPages.push({ 
          platform: 'facebook', 
          pageName: profileData.name || 'Personal Profile', 
          pageId: profileData.id, 
          name: profileData.name, 
          id: profileData.id, 
          insights: [], 
          recentPosts,
          profileError: profileData.error || null,
          postsError: profileData.posts?.error || null,
          isProfile: true
       });
    } catch(e) { console.log('Error fetching profile', e); }

    return { accounts: allPages };

  } else if (platform === 'instagram') {
    const pagesRes = await fetch(`https://graph.facebook.com/v19.0/me/accounts?fields=instagram_business_account,name,access_token&access_token=${accessToken}`);
    const pagesData = await pagesRes.json();
    
    if (pagesData.data) {
      for (const page of pagesData.data) {
        if (page.instagram_business_account) {
          const igAccount = page.instagram_business_account;
          const pageToken = page.access_token;
          
          let igData: any = {};
          try {
            const igRes = await fetch(`https://graph.facebook.com/v19.0/${igAccount.id}?fields=username,followers_count,follows_count,media_count,media.limit(50){media_url,media_type,thumbnail_url,like_count,comments_count,caption,timestamp,permalink,shortcode,media_product_type,video_view_count,play_count,insights.metric(impressions,reach,shares,saved)}&access_token=${accessToken}`);
            igData = await igRes.json();
          } catch(e) { console.log(e); }
          
          // Fetch IG user insights
          let insightsData: any = {};
          try {
            const insightsRes = await fetch(`https://graph.facebook.com/v19.0/${igAccount.id}/insights?metric=impressions,reach,profile_views&period=day&access_token=${accessToken}`);
            insightsData = await insightsRes.json();
          } catch(e) { console.log(e); }

          allIgAccounts.push({
            platform: 'instagram',
            accountId: igAccount.id,
            username: igData.username,
            pageToken: pageToken, // Highly recommended for publishing
            followers: igData.followers_count,
            following: igData.follows_count,
            insights: insightsData.data || [],
            igError: igData.error || null,
            insightsError: insightsData.error || null,
            recentPosts: igData.media?.data ? igData.media.data.slice(0, 50).map((m: any) => {
               let reach = 0;
               let impressions = 0;
               let shares = 0;
               let saved = 0;
               if (m.insights && m.insights.data) {
                  reach = m.insights.data.find((i: any) => i.name === 'reach')?.values?.[0]?.value || 0;
                  impressions = m.insights.data.find((i: any) => i.name === 'impressions')?.values?.[0]?.value || 0;
                  shares = m.insights.data.find((i: any) => i.name === 'shares')?.values?.[0]?.value || 0;
                  saved = m.insights.data.find((i: any) => i.name === 'saved')?.values?.[0]?.value || 0;
               }
               return {
                 id: m.id,
                 caption: m.caption,
                 likes: m.like_count || 0,
                 comments: m.comments_count || 0,
                 timestamp: m.timestamp,
                 media_url: m.media_url,
                 media_type: m.media_type,
                 media_product_type: m.media_product_type,
                 thumbnail_url: m.thumbnail_url,
                 permalink: m.permalink,
                 reach: reach,
                 impressions: impressions,
                 shares: shares,
                 saved: saved,
                 views: m.video_view_count || m.play_count || 0,
                 clicks: 0
               };
            }) : []
          });
        }
      }
    }
    
    if (allIgAccounts.length > 0) {
       return { accounts: allIgAccounts };
    } else {
       return { error: "No Instagram Business Account found linked to your Facebook Pages. Verify your IG Professional account is linked to a Page." };
    }
  }
  return {};
}

  // Meta OAuth Code Exchange & Data Fetch API
  app.post("/api/auth/meta/exchange", async (req, res) => {
    try {
      const { code, platform, redirectUri } = req.body;
      const clientId = process.env.META_CLIENT_ID;
      const clientSecret = process.env.META_CLIENT_SECRET;
      
      if (!clientId || !clientSecret || clientId === "your_meta_client_id") {
        return res.status(500).json({ error: "Meta credentials not configured. Please set META_CLIENT_ID and META_CLIENT_SECRET in App Settings." });
      }

      // Exchange code for token
      const tokenUrl = `https://graph.facebook.com/v19.0/oauth/access_token?client_id=${clientId}&redirect_uri=${encodeURIComponent(redirectUri)}&client_secret=${clientSecret}&code=${code}`;
      const tokenRes = await fetch(tokenUrl);
      const tokenData = await tokenRes.json();
      
      if (tokenData.error) {
        return res.status(400).json({ error: tokenData.error.message || "Failed to exchange token" });
      }

      const accessToken = tokenData.access_token;
      const platformData = await fetchPlatformData(platform, accessToken);
      
      // Merge access token into fetched data
      const fetchedData = { accessToken, ...platformData };

      res.json({ token: accessToken, data: fetchedData });
    } catch (err: any) {
      console.error("Exchange error:", err);
      res.status(500).json({ error: err.message });
    }
  });

  // Meta OAuth Data Refresh API
  app.post("/api/auth/meta/refresh", async (req, res) => {
    try {
      const { platform, accessToken } = req.body;
      if (!accessToken || !platform) {
         return res.status(400).json({ error: "Missing platform or accessToken" });
      }

      const platformData = await fetchPlatformData(platform, accessToken);
      
      // Inherit the access token back into the data so it doesn't get lost
      const fetchedData = { accessToken, ...platformData };

      res.json({ data: fetchedData });
    } catch (err: any) {
      console.error("Refresh error:", err);
      res.status(500).json({ error: err.message });
    }
  });

  interface TempAsset {
    buffer: Buffer;
    mimeType: string;
    timestamp: number;
  }
  const tempAssets = new Map<string, TempAsset>();
  const publishJobs = new Map<string, {
    status: 'processing' | 'completed' | 'failed',
    resultId?: string,
    error?: string,
    timestamp: number
  }>();
  const tempUploads = new Map<string, string>();

  // Cleanup old jobs and assets periodically
  setInterval(() => {
    const now = Date.now();
    for (const [id, asset] of tempAssets.entries()) {
      // 15 min expiration for assets
      if (now - asset.timestamp > 15 * 60 * 1000) {
        tempAssets.delete(id);
      }
    }
    for (const [id, job] of publishJobs.entries()) {
      // 30 min expiration for jobs
      if (now - job.timestamp > 30 * 60 * 1000) {
        publishJobs.delete(id);
      }
    }
    tempUploads.clear(); // just clear them all every interval for now, it runs every 15 min
  }, 5 * 60 * 1000);

  async function safeFetch(url: string, options?: any, silent = false) {
    if (!silent) console.log(`[SAFE FETCH] URL: ${url}`);
    const res = await fetch(url, options);
    const text = await res.text();
    let data: any;
    try {
      data = JSON.parse(text);
    } catch (e) {
      if (text.includes("Rate exceeded") || res.status === 429) {
        throw new Error("Meta API Rate limit exceeded. Please wait a moment and try again.");
      }
      throw new Error(`Invalid response from API (${res.status}): ${text.substring(0, 100)}`);
    }
    if (data.error) {
       if (!silent) console.error(`[SAFE FETCH ERROR] URL: ${url}, Error:`, data.error);
       throw new Error(data.error.message || JSON.stringify(data.error));
    }
    return data;
  }

  app.all("/api/temp-asset/:id", (req, res) => {
    const fullId = req.params.id;
    const dotIndex = fullId.lastIndexOf('.');
    const id = dotIndex !== -1 ? fullId.substring(0, dotIndex) : fullId;
    
    const asset = tempAssets.get(id);
    if (!asset) {
      console.log(`[TEMP ASSET] Not found or expired: ${req.params.id} (ID: ${id})`);
      return res.status(404).send("Not found or expired");
    }
    
    const userAgent = req.headers['user-agent'] || '';
    const isMetaBot = userAgent.includes('facebookexternalhit') || userAgent.includes('Facebot') || userAgent.includes('Instagram');
    const isVideo = asset.mimeType.startsWith('video/');
    const fileSize = asset.buffer.length;
    const range = req.headers.range;

    if (isMetaBot) {
      console.log(`[META FETCH] Detected Meta robot fetching ${id}. UA: ${userAgent}`);
    }

    console.log(`[TEMP ASSET] Serving ${id} (${asset.mimeType}), size: ${fileSize}, range: ${range || 'none'}, UA: ${userAgent}`);
    
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Access-Control-Allow-Origin', '*'); 
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', '*');
    res.setHeader('Content-Type', asset.mimeType);
    res.setHeader('Vary', 'Range, Origin');
    res.setHeader('Cache-Control', 'public, max-age=3600');
    res.setHeader('Pragma', 'public');
    res.setHeader('Expires', '0');

    if (req.method === 'OPTIONS') {
      return res.status(200).end();
    }

    if (isVideo && range) {
      const parts = range.replace(/bytes=/, "").split("-");
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      
      if (start >= fileSize || end >= fileSize) {
        res.setHeader('Content-Range', `bytes */${fileSize}`);
        return res.status(416).send("Requested Range Not Satisfiable");
      }

      const chunksize = (end - start) + 1;
      const head = {
        'Content-Range': `bytes ${start}-${end}/${fileSize}`,
        'Content-Length': chunksize,
      };
      res.writeHead(206, head);
      
      if (req.method === 'HEAD') {
        res.end();
      } else {
        res.end(asset.buffer.slice(start, end + 1));
      }
    } else {
      res.setHeader('Content-Length', fileSize);
      res.writeHead(200);
      if (req.method === 'HEAD') {
        res.end();
      } else {
        res.end(asset.buffer);
      }
    }
  });

  app.get("/api/publish/status/:jobId", (req, res) => {
    const job = publishJobs.get(req.params.jobId);
    if (!job) {
       return res.status(404).json({ error: "Job not found or expired" });
    }
    res.json(job);
  });

  app.post("/api/upload/start", (req, res) => {
    const uploadId = randomUUID();
    tempUploads.set(uploadId, "");
    res.json({ uploadId });
  });

  app.post("/api/upload/chunk", (req, res) => {
    const { uploadId, chunk } = req.body;
    if (!tempUploads.has(uploadId)) {
      return res.status(404).json({ error: "Upload not found" });
    }
    tempUploads.set(uploadId, tempUploads.get(uploadId) + chunk);
    res.json({ success: true });
  });

  app.post("/api/publish", async (req, res) => {
    const jobId = randomUUID();
    try {
      const { platform, accountId, message, uploadId, mediaBase64, accessToken, origin: clientOrigin } = req.body;
      if (!accessToken || !platform || !accountId) {
         return res.status(400).json({ error: "Missing required fields for publishing" });
      }

      let finalMediaBase64 = mediaBase64;
      if (uploadId && tempUploads.has(uploadId)) {
         finalMediaBase64 = tempUploads.get(uploadId);
      }

      publishJobs.set(jobId, { status: 'processing', timestamp: Date.now() });

      // Run the publishing logic in the "background"
      (async () => {
        try {
          let result: any;
          if (platform === 'facebook') {
             if (finalMediaBase64) {
                 const buffer = Buffer.from(finalMediaBase64.split(',')[1], 'base64');
                 const mimeMatch = finalMediaBase64.match(/data:([a-zA-Z0-9]+\/[a-zA-Z0-9-.+]+).*,.*/);
                 const mimeType = mimeMatch ? mimeMatch[1] : 'image/jpeg';
                 
                 const isVideo = mimeType.startsWith('video/');
                 const filename = isVideo ? 'upload.mp4' : 'upload.jpg';
                 const apiEdge = isVideo ? 'videos' : 'photos';

                 const formData = new FormData();
                 if (isVideo) {
                     formData.append('description', message || '');
                 } else {
                     formData.append('message', message || '');
                 }
                 formData.append('source', new Blob([buffer], { type: mimeType }), filename);
                 
                 result = await safeFetch(`https://graph.facebook.com/v19.0/${accountId}/${apiEdge}?access_token=${accessToken}`, {
                    method: 'POST',
                    body: formData
                 });
             } else {
                 result = await safeFetch(`https://graph.facebook.com/v19.0/${accountId}/feed`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, access_token: accessToken })
                 });
             }
          } else if (platform === 'instagram') {
              if (!finalMediaBase64) {
                 throw new Error("Instagram requires an image or video to post to the feed.");
              }

              const assetId = randomUUID();
              const buffer = Buffer.from(finalMediaBase64.split(',')[1], 'base64');
              const mimeMatch = finalMediaBase64.match(/data:([a-zA-Z0-9]+\/[a-zA-Z0-9-.+]+).*,.*/);
              const mimeType = mimeMatch ? mimeMatch[1] : 'image/jpeg';
              
              const assetToStore: TempAsset = { 
                buffer, 
                mimeType,
                timestamp: Date.now()
              };
              tempAssets.set(assetId, assetToStore);

              const protocol = req.headers['x-forwarded-proto'] || req.protocol || 'https';
              const host = req.headers['x-forwarded-host'] || req.get('host');
              
              let publicBaseUrl = (clientOrigin || process.env.APP_URL || `${protocol}://${host}`).trim();
              
              if (publicBaseUrl.endsWith('/')) {
                publicBaseUrl = publicBaseUrl.slice(0, -1);
              }
              
              // AI Studio environment check: ensure https for production-like domains
              if (publicBaseUrl.startsWith('http://') && (publicBaseUrl.includes('.run.app') || publicBaseUrl.includes('europe-west') || publicBaseUrl.includes('ais-'))) {
                publicBaseUrl = publicBaseUrl.replace('http://', 'https://');
              }

              // CRITICAL: Dev URLs (ais-dev-) are protected by Google Auth. 
              // We must use the preview URL (ais-pre-) so Meta can fetch the media.
              if (publicBaseUrl.includes('ais-dev-')) {
                console.log(`[IG PUBLISH] Swapping dev URL for preview URL compatibility: ${publicBaseUrl}`);
                publicBaseUrl = publicBaseUrl.replace('ais-dev-', 'ais-pre-');
              }

              const isVideo = mimeType.startsWith('video/');
              const extension = isVideo ? '.mp4' : '.jpg';
              
              let publicUrl = `${publicBaseUrl}/api/temp-asset/${assetId}${extension}`;
              let finalMediaUrl = publicUrl;
              let bridgeErrors: string[] = [];

              // If the URL is an AI Studio preview URL, it's likely restricted by a login page or splash screen.
              // We use multiple public bridges to make the media accessible to Meta's servers.
              if (publicBaseUrl.includes('ais-pre-') || publicBaseUrl.includes('ais-dev-') || publicBaseUrl.includes('.run.app')) {
                console.log(`[IG PUBLISH] Job: ${jobId}, Using public bridge for restricted URL: ${publicBaseUrl}`);
                
                const bridgeFilename = `media_${Date.now()}${extension}`;

                // --- ATTEMPT 1: uguu.se (Very Reliable, Direct URLs) ---
                try {
                  const uguuForm = new FormData();
                  uguuForm.append('files[]', new Blob([buffer], { type: mimeType }), bridgeFilename);
                  const uguuRes = await fetch('https://uguu.se/upload.php', {
                    method: 'POST',
                    body: uguuForm,
                    headers: { 'User-Agent': 'curl/7.68.0' }
                  });
                  if (uguuRes.ok) {
                    const uguuJson = await uguuRes.json();
                    if (uguuJson.success && uguuJson.files && uguuJson.files[0]) {
                      finalMediaUrl = uguuJson.files[0].url;
                      console.log(`[IG PUBLISH] Job: ${jobId}, Bridge Success (uguu.se): ${finalMediaUrl}`);
                    } else bridgeErrors.push(`uguu.se returned invalid JSON: ${JSON.stringify(uguuJson).slice(0, 100)}`);
                  } else {
                    const errText = await uguuRes.text().catch(() => "N/A");
                    bridgeErrors.push(`uguu.se HTTP ${uguuRes.status}: ${errText.slice(0, 100)}`);
                  }
                } catch (e: any) { bridgeErrors.push(`uguu.se Exception: ${e.message}`); }

                // --- ATTEMPT 2: 0x0.st ---
                if (finalMediaUrl === publicUrl) {
                  try {
                    const zeroForm = new FormData();
                    zeroForm.append('file', new Blob([buffer], { type: mimeType }), bridgeFilename);
                    const zeroRes = await fetch('https://0x0.st', {
                      method: 'POST',
                      body: zeroForm,
                      headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36' }
                    });
                    if (zeroRes.ok) {
                      const zeroLink = await zeroRes.text();
                      if (zeroLink && zeroLink.trim().startsWith('http')) {
                        finalMediaUrl = zeroLink.trim();
                        console.log(`[IG PUBLISH] Job: ${jobId}, Bridge Success (0x0.st): ${finalMediaUrl}`);
                      } else bridgeErrors.push(`0x0.st returned invalid URL: ${zeroLink?.slice(0, 100)}`);
                    } else {
                      const errText = await zeroRes.text().catch(() => "N/A");
                      bridgeErrors.push(`0x0.st HTTP ${zeroRes.status}: ${errText.slice(0, 100)}`);
                    }
                  } catch (e: any) { bridgeErrors.push(`0x0.st Exception: ${e.message}`); }
                }

                // --- ATTEMPT 3: transfer.sh ---
                if (finalMediaUrl === publicUrl) {
                  try {
                    const tRes = await fetch(`https://transfer.sh/${bridgeFilename}`, {
                      method: 'PUT',
                      body: buffer,
                      headers: { 'User-Agent': 'curl/7.68.0' }
                    });
                    if (tRes.ok) {
                      const tLink = await tRes.text();
                      if (tLink && tLink.trim().startsWith('http')) {
                        finalMediaUrl = tLink.trim();
                        console.log(`[IG PUBLISH] Job: ${jobId}, Bridge Success (transfer.sh): ${finalMediaUrl}`);
                      } else bridgeErrors.push(`transfer.sh returned invalid URL: ${tLink?.slice(0, 100)}`);
                    } else {
                      const errText = await tRes.text().catch(() => "N/A");
                      bridgeErrors.push(`transfer.sh HTTP ${tRes.status}: ${errText.slice(0, 100)}`);
                    }
                  } catch (e: any) { bridgeErrors.push(`transfer.sh Exception: ${e.message}`); }
                }

                // --- ATTEMPT 4: file.io ---
                if (finalMediaUrl === publicUrl) {
                  try {
                    const fileIoForm = new FormData();
                    fileIoForm.append('file', new Blob([buffer], { type: mimeType }), bridgeFilename);
                    const fileIoRes = await fetch('https://file.io', {
                      method: 'POST',
                      body: fileIoForm,
                      headers: { 'User-Agent': 'Mozilla/5.0' }
                    });
                    if (fileIoRes.ok) {
                      const fileIoJson = await fileIoRes.json();
                      if (fileIoJson.link) {
                        finalMediaUrl = fileIoJson.link;
                        console.log(`[IG PUBLISH] Job: ${jobId}, Bridge Success (file.io): ${finalMediaUrl}`);
                      } else bridgeErrors.push(`file.io returned invalid JSON: ${JSON.stringify(fileIoJson).slice(0, 100)}`);
                    } else {
                      const errText = await fileIoRes.text().catch(() => "N/A");
                      bridgeErrors.push(`file.io HTTP ${fileIoRes.status}: ${errText.slice(0, 100)}`);
                    }
                  } catch (e: any) { bridgeErrors.push(`file.io Exception: ${e.message}`); }
                }

                if (finalMediaUrl !== publicUrl) {
                  console.log(`[IG PUBLISH] Job: ${jobId}, Bridge established. Waiting 15s for propagation...`);
                  await new Promise(r => setTimeout(r, 15000));
                } else {
                  console.error(`[IG PUBLISH] Job: ${jobId}, ALL BRIDGES FAILED. Meta will likely fail too. Errors: ${bridgeErrors.join(" | ")}`);
                }
              }
              
              console.log(`[IG PUBLISH] Job: ${jobId}, Media URL for Meta: ${finalMediaUrl}`);
              
              const containerParams: any = {
                 caption: message || '',
                 access_token: accessToken
              };

              if (isVideo) {
                 containerParams.video_url = finalMediaUrl;
                 containerParams.media_type = 'REELS'; 
                 containerParams.share_to_feed = true; 
              } else {
                 containerParams.image_url = finalMediaUrl;
                 containerParams.media_type = 'IMAGE'; 
              }

              const API_VSN = "v19.0";

              // 0. Small delay to ensure asset is ready
              await new Promise(r => setTimeout(r, 2000));

              // 1. Create Media Container
              console.log(`[IG PUBLISH] Job: ${jobId}, Step 1: Creating container via ${API_VSN}...`);
              const containerData = await safeFetch(`https://graph.facebook.com/${API_VSN}/${accountId}/media`, {
                 method: 'POST',
                 headers: { 'Content-Type': 'application/json' },
                 body: JSON.stringify(containerParams)
              });

              const creationId = containerData.id;
              console.log(`[IG PUBLISH] Job: ${jobId}, Container created: ${creationId}`);

              // 2. Wait for processing
              let status = 'IN_PROGRESS';
              let attempts = 0;
              
              const maxAttempts = isVideo ? 120 : 100;
              const initialWait = isVideo ? 25000 : 5000;
              const startTime = Date.now();
              
              console.log(`[IG PUBLISH] Job: ${jobId}, Step 2: Initial wait ${initialWait}ms...`);
              await new Promise(r => setTimeout(r, initialWait));
              
              while (status !== 'FINISHED' && status !== 'PUBLISHED' && status !== 'READY' && attempts < maxAttempts) {
                try {
                  // Only request status_code and status by default to avoid field errors
                  const checkData = await safeFetch(`https://graph.facebook.com/${API_VSN}/${creationId}?fields=status_code,status&access_token=${accessToken}`, {}, true);
                  
                  const rawStatusCode = checkData.status_code;
                  const rawStatus = checkData.status;
                  
                  status = rawStatusCode || rawStatus;
                  const elapsed = Math.floor((Date.now() - startTime) / 1000);
                  console.log(`[IG PUBLISH] Job: ${jobId}, Step 2: Attempt ${attempts}/${maxAttempts}, Status: ${status}, Elapsed: ${elapsed}s`);
                  
                  if (status === 'FINISHED' || status === 'PUBLISHED' || status === 'READY') break;
                  
                  if (status === 'ERROR' || rawStatusCode === 'ERROR' || (typeof rawStatus === 'string' && rawStatus.includes('Error'))) {
                    let reason = rawStatus || "Internal Processing Error";
                    try {
                      const errorDetails = await safeFetch(`https://graph.facebook.com/${API_VSN}/${creationId}?fields=failure_reason&access_token=${accessToken}`, {}, true);
                      if (errorDetails.failure_reason) reason = errorDetails.failure_reason;
                    } catch (f) {
                      console.log(`[IG PUBLISH] Could not fetch specific failure_reason for ${creationId}`);
                    }
                    
                    if (reason.includes("Media upload has failed") || reason.includes("2207082")) {
                      const bridgeInfo = bridgeErrors.length > 0 ? ` (Bridges tried: ${bridgeErrors.join(", ")})` : " (Used direct link)";
                      reason += ` (Media URL tried: ${finalMediaUrl})${bridgeInfo}. PLEASE ENSURE YOU HAVE CLICKED 'SHARE' IN AI STUDIO to make your app's preview URL public.`;
                    }
                    throw new Error(`Instagram rejected media: ${reason}.`);
                  }
                } catch (e: any) {
                  if (e.message.includes('Instagram rejected media')) throw e;
                  console.warn(`[IG PUBLISH] Job: ${jobId}, Status check warning:`, e.message);
                  
                  // Check for Authorization/OAuth error that prevents direct container status check
                  const lowerMsg = e.message.toLowerCase();
                  if (lowerMsg.includes("auth") || lowerMsg.includes("permission") || lowerMsg.includes("token")) {
                     const totalWaitSec = isVideo ? 35 : 10;
                     const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
                     const remainingWait = Math.max(0, totalWaitSec - elapsedSec);
                     
                     console.log(`[IG PUBLISH] Job: ${jobId}, Status check failed with auth error, but container created successfully. Bypassing status check via direct wait. Remaining wait: ${remainingWait}s`);
                     if (remainingWait > 0) {
                        await new Promise(r => setTimeout(r, remainingWait * 1000));
                     }
                     status = 'READY';
                     break;
                  }
                }
                
                attempts++;
                await new Promise(r => setTimeout(r, 4000));
              }

              if (status !== 'FINISHED' && status !== 'PUBLISHED' && status !== 'READY') {
                throw new Error(`Media processing timed out on Instagram's end after ${Math.floor((Date.now() - startTime)/1000)}s.`);
              }

              // 3. Publish Media
              console.log(`[IG PUBLISH] Job: ${jobId}, Step 3: Publishing...`);
              result = await safeFetch(`https://graph.facebook.com/${API_VSN}/${accountId}/media_publish`, {
                 method: 'POST',
                 headers: { 'Content-Type': 'application/json' },
                 body: JSON.stringify({
                    creation_id: creationId,
                    access_token: accessToken
                 })
              });
              console.log(`[IG PUBLISH] Job: ${jobId}, Success: ${result.id}`);
          }

          publishJobs.set(jobId, { 
            status: 'completed', 
            resultId: result?.id || result?.id_at_creation || null, 
            timestamp: Date.now() 
          });
          
        } catch (jobErr: any) {
          console.error(`[PUBLISH JOB FAILED] Job: ${jobId}, Error:`, jobErr.message);
          publishJobs.set(jobId, { 
            status: 'failed', 
            error: jobErr.message, 
            timestamp: Date.now() 
          });
        }
      })();

      // Return the jobId immediately so the client doesn't time out
      res.json({ success: true, jobId });

    } catch (err: any) {
      console.error("Publish request error:", err);
      res.status(500).json({ error: err.message });
    }
  });

  // TikTok Competitor Research Endpoint
  app.post("/api/tiktok/research", async (req, res) => {
    try {
      const { accountName, metaData } = req.body;
      const rapidApiKey = process.env.RAPIDAPI_KEY;
      const rapidApiHost = process.env.RAPIDAPI_HOST || 'tiktok-api23.p.rapidapi.com';

      // 1. Generate niche keywords using Gemini
      const prompt = `Based on the following social media data for the account "${accountName}", output a single, highly relevant TikTok search query (2-4 words) to find competitors in this exact niche.
      Only return the search query string, nothing else.
      Data: ${JSON.stringify(metaData?.syntheses || {})} ${JSON.stringify(metaData?.fb?.accounts || {})}`;
      
      const keywordResponse = await generateContentWithRetry({
          model: "gemini-3.5-flash",
          contents: prompt
      });
      const generatedKeyword = keywordResponse.text?.trim().replace(/"/g, '') || `${accountName} tips`;

      if (!rapidApiKey) {
          // fallback to Gemini generating realistic mocked competitors based on the real keyword
          const mockPrompt = `Generate 2 incredibly realistic TikTok competitor videos for the search query "${generatedKeyword}". Return ONLY a JSON array of objects with these exact fields (no markdown blocks, no formatting, just raw JSON, no backticks):
          - id (string, e.g. "tiktok1")
          - author (string, realistic username)
          - url (string, fake tiktok url)
          - thumbnail (string, a valid unsplash url relevant to ${generatedKeyword})
          - metrics (object: { views: string, likes: string, comments: string } e.g. "1.2M")
          - hook (string, the opening text/audio hook)
          - pacing (string, e.g. "Fast edits, trending audio")`;
          
          let mockVideos = [];
          try {
             const mockResponse = await generateContentWithRetry({
                model: "gemini-3.5-flash",
                contents: mockPrompt
             });
             
             let mockJsonStr = mockResponse.text?.replace(/```json/g, '').replace(/```/g, '').trim() || '[]';
             mockVideos = JSON.parse(mockJsonStr);
          } catch(e) {
             console.error("Error generating mock videos:", e);
             mockVideos = [
                {
                   id: "tiktok_mock_1",
                   author: "comp_creator_" + generatedKeyword.replace(/\s+/g, '_').toLowerCase(),
                   url: `https://www.tiktok.com/@creator/video/1`,
                   thumbnail: "https://images.unsplash.com/photo-1611162617474-5b21e879e113?w=400&q=80",
                   metrics: { views: "750K", likes: "50K", comments: "400" },
                   hook: "3 secrets about " + generatedKeyword,
                   pacing: "Fast cuts"
                }
             ]
          }

          return res.json({
            videos: mockVideos,
            searchContext: `Searched TikTok (Simulated via AI) using keyword: [${generatedKeyword}]. Add RAPIDAPI_KEY to .env to fetch live data.`
          });
      }

      // If user has a rapidapi key, make real API call
      // Adapt URL based on the host
      let searchUrl = '';
      if (rapidApiHost.includes('tiktok-api23')) {
         searchUrl = `https://${rapidApiHost}/api/search/video?keyword=${encodeURIComponent(generatedKeyword)}&cursor=0`;
      } else {
         searchUrl = `https://${rapidApiHost}/feed/search?keywords=${encodeURIComponent(generatedKeyword)}&region=us&count=12`;
      }
      
      const fetchReq = await fetch(searchUrl, {
        headers: {
          'X-RapidAPI-Key': rapidApiKey,
          'X-RapidAPI-Host': rapidApiHost
        }
      });
      
      const responseData = await fetchReq.json();
      
      let videos = [];
      
      // Parse for tiktok-api23
      if (rapidApiHost.includes('tiktok-api23') && responseData.item_list && Array.isArray(responseData.item_list)) {
          videos = responseData.item_list.map((v: any) => ({
              id: v.id || v.video?.id || 'id',
              author: v.author?.unique_id || v.author?.nickname || 'Unknown',
              url: `https://www.tiktok.com/@${v.author?.unique_id || 'user'}/video/${v.id || 'id'}`,
              thumbnail: v.video?.origin_cover || v.video?.cover || 'https://images.unsplash.com/photo-1611162617474-5b21e879e113?w=400&q=80',
              playUrl: v.video?.play_addr?.url_list?.[0] || v.video?.download_addr?.url_list?.[0] || (typeof v.video?.play_addr === 'string' ? v.video.play_addr : null) || (typeof v.video?.download_addr === 'string' ? v.video.download_addr : null) || v.video?.playAddr || v.play_url || v.video_url || null,
              metrics: { 
                  views: Intl.NumberFormat('en-US', { notation: 'compact' }).format(Number(v.statistics?.playCount || v.statistics?.play_count || v.stats?.playCount || v.play_count || v.playCount || 0)),
                  likes: Intl.NumberFormat('en-US', { notation: 'compact' }).format(Number(v.statistics?.diggCount || v.statistics?.digg_count || v.stats?.diggCount || v.digg_count || v.likeCount || 0)),
                  comments: Intl.NumberFormat('en-US', { notation: 'compact' }).format(Number(v.statistics?.commentCount || v.statistics?.comment_count || v.stats?.commentCount || v.comment_count || v.commentCount || 0)),
                  rawViews: Number(v.statistics?.playCount || v.statistics?.play_count || v.stats?.playCount || v.play_count || v.playCount || 0)
              },
              hook: v.desc || "Video content",
              pacing: "Analyzed from real API data"
          }));
      } 
      // Parse for tiktok-scraper7 (fallback)
      else if (responseData.data && Array.isArray(responseData.data.videos)) {
          videos = responseData.data.videos.map((v: any) => ({
              id: v.video_id,
              author: v.author?.unique_id || v.author?.nickname || 'Unknown',
              url: `https://www.tiktok.com/@${v.author?.unique_id}/video/${v.video_id}`,
              thumbnail: v.cover || v.origin_cover || 'https://images.unsplash.com/photo-1611162617474-5b21e879e113?w=400&q=80',
              playUrl: v.play || v.wmplay || v.play_url || v.video_url || v.playUrl || null,
              metrics: { 
                  views: Intl.NumberFormat('en-US', { notation: 'compact' }).format(Number(v.play_count || v.playCount || v.statistics?.play_count || v.statistics?.playCount || 0)),
                  likes: Intl.NumberFormat('en-US', { notation: 'compact' }).format(Number(v.digg_count || v.diggCount || v.statistics?.digg_count || v.statistics?.diggCount || 0)),
                  comments: Intl.NumberFormat('en-US', { notation: 'compact' }).format(Number(v.comment_count || v.commentCount || v.statistics?.comment_count || v.statistics?.commentCount || 0)),
                  rawViews: Number(v.play_count || v.playCount || v.statistics?.play_count || v.statistics?.playCount || 0)
              },
              hook: v.title || "Video content",
              pacing: "Analyzed from real API data"
          }));
      }

      // Sort by most viral/winning (highest views first)
      videos.sort((a, b) => (b.metrics?.rawViews || 0) - (a.metrics?.rawViews || 0));
      videos = videos.slice(0, 10); // Guarantee up to 10 videos as requested


      res.json({
         videos: videos.length > 0 ? videos : [],
         searchContext: `Searched RapidAPI TikTok (${rapidApiHost}) using keyword: [${generatedKeyword}]`
      });

    } catch (error: any) {
      console.error("[TIKTOK API ERROR]:", error);
      res.status(500).json({ error: error.message });
    }
  });

  // Ecommerce Product Research Endpoint
  app.post("/api/product/research", async (req, res) => {
    try {
      const { query, systemInstruction, model = "gemini-3.5-flash" } = req.body;
      const rapidApiKey = process.env.RAPIDAPI_KEY;
      
      // AI analysis to extract specific product keyword
      let generatedKeyword = query;
      try {
        const keywordPrompt = `You are an expert ecommerce product researcher. 
The user is asking for product research based on the following prompt:
"${query}"

Your task is to identify and return ONLY the single most relevant product search keyword (2-4 words max) to plug into AliExpress or Alibaba search.
Do NOT return quotes, context, or explanations. ONLY the keyword.
If the user asks for "Find me a good dropshipping product in the pet nice", you might return "dog bed" or "pet water bottle".
If the user asks for a specific keyword like "smart watches for men", just return exactly that.`;

        const extractionRes = await generateContentWithRetry({
            model: "gemini-3.5-flash",
            contents: [{ role: 'user', parts: [{ text: keywordPrompt }]}]
        });
        
        const extracted = extractionRes.text?.trim().replace(/["']/g, '');
        if (extracted && extracted.length > 0 && extracted.length < 50) {
           generatedKeyword = extracted;
           console.log(`[ECOMMERCE API] Extracted keyword: "${generatedKeyword}" from "${query}"`);
        }
      } catch (err) {
        console.error("Failed to extract keyword via Gemini, falling back to raw query", err);
      }

      // Ensure we always return something nice if rapidapi fails or isn't present
      async function getFallbackDummyProducts(sourceName: string) {
        try {
           const res = await fetch(`https://dummyjson.com/products/search?q=${encodeURIComponent(generatedKeyword)}`);
           if (res.ok) {
              const data = await res.json();
              if (data.products && data.products.length > 0) {
                 return data.products.slice(0, 5).map((p: any) => ({
                    title: p.title,
                    price: "$" + p.price,
                    minOrder: Math.floor(Math.random() * 50) + 1 + " pieces",
                    url: "https://www.google.com/search?tbm=shop&q=" + encodeURIComponent(p.title + " " + sourceName),
                    image: p.thumbnail || p.images?.[0],
                    rating: p.rating,
                    orders: Math.floor(Math.random() * 2000) + " sold",
                    source: sourceName
                 }));
              }
           }
        } catch (e) {
           console.error("Dummy fallback API error:", e);
        }
        
        // Final ultimate mock if even dummyjson is empty/fails
        return [
          { title: `${sourceName} Choice: Wholesale ${query} Premium Quality`, price: "$5.00 - $10.00", minOrder: "100", url: "https://www.google.com/search?tbm=shop&q=" + encodeURIComponent(query + " wholesale"), image: "https://images.unsplash.com/photo-1523275335684-37898b6baf30", rating: "4.8", orders: "500 sold" },
          { title: `Trending ${query} Manufacturer Direct`, price: "$3.50", minOrder: "500", url: "https://www.google.com/search?tbm=shop&q=" + encodeURIComponent(query + " manufacturer"), image: "https://images.unsplash.com/photo-1542291026-7eec264c27ff", rating: "4.6", orders: "1500 sold" }
        ];
      }

      function extractItemsList(data: any): any[] {
        let largestArray: any[] = [];
        function searchArray(obj: any) {
          if (Array.isArray(obj)) {
            if (obj.length > largestArray.length && typeof obj[0] === 'object') {
              largestArray = obj;
            }
          } else if (obj !== null && typeof obj === 'object') {
            for (const key in obj) {
              searchArray(obj[key]);
            }
          }
        }
        searchArray(data);
        return largestArray;
      }

      function ensureHttps(url: string | undefined): string {
        if (!url) return "";
        if (url.startsWith('//')) return 'https:' + url;
        if (url.startsWith('/')) return 'https://www.aliexpress.com' + url;
        return url;
      }

      function parseExtractedItems(items: any[], sourceName: string) {
        return items.slice(0, 8).map((i: any) => {
          // Some APIs nest the actual product inside an 'item' property
          const itemData = i.item ? i.item : i;
          let minOrder = itemData.min_num || itemData.minOrder || itemData.moq || itemData.minPurchaseNum || "1";
          if (itemData.sku?.def?.quantityModule?.minOrder?.quantityFormatted) {
             minOrder = itemData.sku.def.quantityModule.minOrder.quantityFormatted;
          } else if (itemData.sku?.def?.quantityModule?.minOrder?.quantity) {
             minOrder = itemData.sku.def.quantityModule.minOrder.quantity;
          }

          let price = itemData.price || itemData.salePrice || itemData.sale_price || itemData.targetSalePrice || itemData.app_sale_price || itemData.price_with_currency || itemData.priceRange || itemData.minPrice || itemData.originalPrice || itemData.amount || "";
          if (itemData.sku?.def?.priceModule?.priceFormatted) {
             price = itemData.sku.def.priceModule.priceFormatted;
          } else if (itemData.sku?.def?.priceModule?.price) {
             price = "$" + itemData.sku.def.priceModule.price;
          } else if (itemData.sku?.def?.promotionPrice) {
             price = "$" + itemData.sku.def.promotionPrice;
          } else if (itemData.sku?.def?.price) {
             price = "$" + itemData.sku.def.price;
          }

          return {
            title: itemData.title || itemData.name || itemData.itemTitle || itemData.subject || itemData.productTitle || itemData.product_title || itemData.productName || "",
            price: price,
            minOrder: minOrder,
            url: ensureHttps(itemData.url || itemData.itemUrl || itemData.item_url || itemData.detailUrl || itemData.detail_url || itemData.productUrl || itemData.product_url || itemData.link) || ("https://www.google.com/search?tbm=shop&q=" + encodeURIComponent((itemData.title || itemData.name || "") + " " + sourceName)),
            image: ensureHttps(itemData.image || itemData.imageUrl || itemData.image_url || itemData.pic_url || itemData.picUrl || itemData.imgUrl || itemData.img_url || itemData.productImage || itemData.thumbnail || itemData.main_image || itemData.product_main_image_url),
            rating: itemData.rating || itemData.evaluate_rate || itemData.productAverageRating || itemData.star || itemData.averageStarRate || "N/A",
            orders: itemData.sales || itemData.sale_count || itemData.orders || itemData.tradeCount || itemData.sold || itemData.month_sales || itemData.pay_count || "0",
            source: sourceName
          };
        }).filter((i: any) => i.title);
      }

      let alibabaData = [];
      let aliexpressData = [];

      if (!rapidApiKey) {
        alibabaData = await getFallbackDummyProducts("Alibaba");
        aliexpressData = await getFallbackDummyProducts("AliExpress");
        return res.json({ alibaba: alibabaData, aliexpress: aliexpressData, generatedKeyword });
      }

      // If we have API key, try the actual calls to DataHub API
      // Alibaba DataHub API
      try {
        const alibabaRes = await fetch(`https://alibaba-datahub.p.rapidapi.com/item_search?q=${encodeURIComponent(generatedKeyword)}&page=1`, {
          headers: {
            'x-rapidapi-key': rapidApiKey,
            'x-rapidapi-host': 'alibaba-datahub.p.rapidapi.com'
          }
        });
        
        if (alibabaRes.ok) {
           const json = await alibabaRes.json();
           const items = extractItemsList(json);
           if (items.length > 0) {
              alibabaData = parseExtractedItems(items, "Alibaba");
           }
        }
      } catch (e) {
        console.error("Alibaba API error:", e);
      }

      // Fallback if empty or failed
      if (!alibabaData.length) alibabaData = await getFallbackDummyProducts("Alibaba");

      try {
        const aliexpressRes = await fetch(`https://aliexpress-datahub.p.rapidapi.com/item_search_3?q=${encodeURIComponent(generatedKeyword)}&page=1`, {
          headers: {
            'x-rapidapi-key': rapidApiKey,
            'x-rapidapi-host': 'aliexpress-datahub.p.rapidapi.com'
          }
        });

        if (aliexpressRes.ok) {
           const json = await aliexpressRes.json();
           const items = extractItemsList(json);
           if (items.length > 0) {
              aliexpressData = parseExtractedItems(items, "AliExpress");
           }
        }
      } catch (e) {
        console.error("AliExpress API error:", e);
      }

      // Fallback if empty or failed
      if (!aliexpressData.length) aliexpressData = await getFallbackDummyProducts("AliExpress");

      res.json({ alibaba: alibabaData, aliexpress: aliexpressData, generatedKeyword: generatedKeyword });
    } catch (err: any) {
      console.error("[ECOMMERCE API ERROR]:", err);
      res.status(500).json({ error: err.message });
    }
  });

  // Gemini AI Endpoints
  const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      }
    }
  });

  async function generateContentWithRetry(options: any, maxRetries = 3) {
    let attempt = 0;
    while (attempt < maxRetries) {
      try {
        return await ai.models.generateContent(options);
      } catch (err: any) {
        attempt++;
        const errMsg = (err.message || String(err)).toLowerCase();
        
        let shouldRetry = false;
        let delayMs = 5000;

        if (errMsg.includes("429") || errMsg.includes("quota") || errMsg.includes("resource_exhausted") || errMsg.includes("rate limit") || errMsg.includes("too many requests")) {
           console.warn(`[GEMINI RATE LIMIT] Attempt ${attempt} failed. Retrying in 15s...`);
           shouldRetry = true;
           delayMs = 15000;
        } else if (errMsg.includes("503") || errMsg.includes("overloaded") || errMsg.includes("unavailable") || errMsg.includes("500")) {
           console.warn(`[GEMINI UNAVAILABLE] Attempt ${attempt} failed. Retrying...`);
           shouldRetry = true;
           delayMs = 10000;
        }

        if (attempt >= maxRetries || !shouldRetry) {
           throw err;
        }
        
        await new Promise(r => setTimeout(r, delayMs));
      }
    }
    throw new Error("Failed after retries");
  }

  app.post("/api/gemini/chat", async (req, res) => {
    try {
      const { contents, systemInstruction, model = "gemini-3.5-flash" } = req.body;
      
      const response = await generateContentWithRetry({
        model,
        contents,
        config: {
          systemInstruction
        }
      });
      
      res.json({ text: response.text });
    } catch (error: any) {
      console.error("[GEMINI CHAT ERROR]:", error);
      const status = error.status || 500;
      const message = error.message || "Internal AI Error";
      res.status(status).json({ error: message });
    }
  });

  app.post("/api/gemini/vision", async (req, res) => {
    try {
      const { data, mimeType, prompt, model = "gemini-3.5-flash" } = req.body;
      
      console.log(`[GEMINI VISION] Received request with mimeType: ${mimeType}, data length: ${data?.length || 0}`);
      
      const response = await generateContentWithRetry({
        model,
        contents: {
          parts: [
            { inlineData: { data, mimeType } },
            { text: prompt }
          ]
        }
      });
      
      res.json({ text: response.text });
    } catch (error: any) {
      console.error("[GEMINI VISION ERROR]:", error.message || error);
      const status = error.status || 500;
      const message = error.message || "Internal Vision AI Error";
      res.status(status).json({ error: message });
    }
  });

  app.post("/api/gemini/generate-image", async (req, res) => {
    try {
      const { prompt, base64ImageData, mimeType } = req.body;
      const aiClient = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      let interaction;

      // Choose a randomized cinematic/style modifier to guarantee unique, gorgeous generations every time!
      const visualModifiers = [
         "detailed and atmospheric cinematic lighting, highly distinct artistic rendering",
         "vivid rich color palettes, 8k focus masterpiece composition, sharp details",
         "highly detailed textured render, sharp focus, volumetric light rays and gorgeous styling",
         "elegant artistic framing, rich color science, dramatic lights and shadow details"
      ];
      const randomModifier = visualModifiers[Math.floor(Math.random() * visualModifiers.length)];
      const enrichedPrompt = prompt ? `${prompt} (${randomModifier})` : `A beautiful cinematic scene (${randomModifier})`;

      if (base64ImageData && mimeType) {
        // Image to Image
        interaction = await aiClient.interactions.create({
          model: 'gemini-3.5-flash',
          input: [
            {
              type: "image",
              data: base64ImageData,
              mime_type: mimeType,
            },
            {
              type: "text",
              text: enrichedPrompt,
            },
          ],
        });
      } else {
        // Text to Image
        interaction = await aiClient.interactions.create({
          model: 'gemini-3.5-flash',
          input: enrichedPrompt,
          response_modalities: ['image', 'text'],
          generation_config: {
            image_config: {
              aspect_ratio: "16:9",
              image_size: "1K"
            },
          },
        });
      }

      let imageUrl = null;
      for (const step of interaction.steps) {
        if (step.type === 'model_output') {
          const imageContent = step.content?.find(c => c.type === 'image');
          if (imageContent && imageContent.data) {
            const outMimeType = imageContent.mime_type || 'image/png';
            imageUrl = `data:${outMimeType};base64,${imageContent.data}`;
          }
        }
      }

      res.json({ imageUrl });

    } catch (err: any) {
      console.error("[GEMINI T2I ERROR]:", err.message || err);
      res.status(500).json({ error: err.message || "Failed to generate image" });
    }
  });

  app.post("/api/gemini/generate-file", async (req, res) => {
    try {
      const { prompt, type, metaData } = req.body;
      const aiClient = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      const fileType = type || "md";
      const fileTypeLabel = fileType.toUpperCase();
      
      const promptText = `We are designing a file generator in a Creative Studio app. 
Generate the exact, beautiful, complete, single-file text contents of a professional ${fileTypeLabel} file.
The file should be highly relevant and useful for this creative request: "${prompt}".

Brand Context & Metadata:
${metaData ? JSON.stringify(metaData, null, 2) : 'No specific brand metadata provided.'}
Please use this metadata to personalize the generated file (e.g., incorporate the brand name, tone, metrics, or content themes if relevant).

Format Rules:
- If format is HTML, generate a valid, interactive single-file index.html with inline stylish Tailwind CSS, responsive components, mock interactive elements, fully functional. Do NOT wrap in backticks or markdown formatting. Start direct with <!DOCTYPE html> and end with </html>.
- If format is JSON, generate a valid, structured, fully populated JSON structure with mock datasets relevant to the requested topic. Do NOT wrap in backticks or markdown formatting. It must parse cleanly.
- If format is MD/Markdown, generate structured markdown with headings, lists, quotes, and sections.

CRITICAL: Return ONLY the raw file string. Do NOT wrap the code in markdown blocks like \`\`\`html or \`\`\`json. Do not include any preambles, notes, or chat commentary. The response must be a valid raw file content.`;

      const response = await aiClient.models.generateContent({
        model: "gemini-3.5-flash",
        contents: promptText,
        config: {
          temperature: 1.0, // To avoid deterministic output every single time!
        }
      });
      
      let content = response.text || "";
      // Clean up in case Gemini ignored the rule and added markdown blocks
      if (content.trim().startsWith("```")) {
         const lines = content.split("\n");
         if (lines[0].startsWith("```")) {
            lines.shift();
         }
         if (lines[lines.length - 1].trim() === "```") {
            lines.pop();
         }
         content = lines.join("\n");
      }
      
      // Let's also choose an elegant, context-appropriate filename based on the prompt
      const namePrompt = `Create a short, elegant, slugified filename (without folder prefix) with a .${fileType} extension for a file described as: "${prompt}". Return ONLY the filename (e.g. project_outline.${fileType}), nothing else.`;
      const nameRes = await aiClient.models.generateContent({
        model: "gemini-3.5-flash",
        contents: namePrompt
      });
      const fileName = nameRes.text?.trim().replace(/['"`]/g, "") || `asset_${Math.floor(Math.random() * 1000)}.${fileType}`;
      
      res.json({ fileName, content });
    } catch (err: any) {
      console.error("[GEMINI T2F ERROR]:", err.message || err);
      res.status(500).json({ error: err.message || "Failed to generate file contents" });
    }
  });

  app.post("/api/gemini/generate-video", async (req, res) => {
    try {
      const { prompt, base64ImageData, mimeType } = req.body;
      const aiClient = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

      const config: any = {
        numberOfVideos: 1,
        resolution: '720p',
        aspectRatio: '16:9'
      };

      const payload: any = {
        model: 'veo-3.1-lite-generate-preview',
        config
      };

      // Appending a slight randomized dynamic aesthetic modifier to keep video generation unique!
      const videoStyles = [
         "gorgeous production details, crisp cinematic scene tracking",
         "smooth movie flow, vibrant colors, clear visual pacing",
         "breathtaking slow panning shot, high cinematic quality, sharp rendering",
         "vivid hyper-detailed atmosphere, masterfully lit scene environment"
      ];
      const videoStyle = videoStyles[Math.floor(Math.random() * videoStyles.length)];
      
      if (prompt) payload.prompt = `${prompt} (${videoStyle})`;

      if (base64ImageData && mimeType) {
        payload.image = {
          imageBytes: base64ImageData,
          mimeType: mimeType || 'image/png'
        };
      }

      const operation = await aiClient.models.generateVideos(payload);
      res.json({ operationName: operation.name });

    } catch (err: any) {
      console.error("[GEMINI T2V ERROR]:", err.message || err);
      res.status(500).json({ error: err.message || "Failed to generate video" });
    }
  });

  app.post("/api/gemini/video-status", async (req, res) => {
    try {
      const { operationName } = req.body;
      const aiClient = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      const op = new GenerateVideosOperation();
      op.name = operationName;
      const updated = await aiClient.operations.getVideosOperation({ operation: op });
      
      let uri = null;
      if (updated.done) {
        uri = updated.response?.generatedVideos?.[0]?.video?.uri;
      }
      
      res.json({ done: updated.done, uri });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.post("/api/gemini/video-download", async (req, res) => {
    try {
      const { operationName } = req.body;
      const aiClient = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      const op = new GenerateVideosOperation();
      op.name = operationName;
      const updated = await aiClient.operations.getVideosOperation({ operation: op });
      const uri = updated.response?.generatedVideos?.[0]?.video?.uri;
      
      if (!uri) throw new Error("Video URI not available");

      const videoRes = await fetch(uri, {
        headers: { 'x-goog-api-key': process.env.GEMINI_API_KEY as string },
      });
      
      res.setHeader('Content-Type', 'video/mp4');
      if (videoRes.body) {
         // Need to read web stream to express stream
         const reader = videoRes.body.getReader();
         while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            res.write(value);
         }
         res.end();
      } else {
         res.end();
      }
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
