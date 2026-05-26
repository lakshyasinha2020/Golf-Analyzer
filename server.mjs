import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL(".", import.meta.url)));
const distRoot = join(root, "dist");
const port = Number(process.env.PORT || 4173);

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webm": "video/webm",
  ".mp4": "video/mp4",
};

const securityHeaders = {
  "Content-Security-Policy": [
    "default-src 'self'",
    "base-uri 'self'",
    "object-src 'none'",
    "frame-ancestors 'none'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com data:",
    "img-src 'self' data: blob: https:",
    "media-src 'self' blob: https:",
    "connect-src 'self' blob: https://*.supabase.co wss://*.supabase.co https://storage.googleapis.com https://tfhub.dev",
    "worker-src 'self' blob:",
  ].join("; "),
  "Cross-Origin-Opener-Policy": "same-origin",
  "Permissions-Policy": "camera=(self), microphone=(), geolocation=(), payment=(), usb=(), browsing-topics=()",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
};

function withSecurityHeaders(headers = {}) {
  return { ...securityHeaders, ...headers };
}

function send(res, status, body, headers = {}) {
  res.writeHead(status, withSecurityHeaders(headers));
  res.end(body);
}

createServer((req, res) => {
  if (!existsSync(distRoot)) {
    send(
      res,
      503,
      "SwingGrade is now a Vite app. Run `npm install` and `npm run dev`, or run `npm run build` before using this preview server.",
      { "Content-Type": "text/plain; charset=utf-8" },
    );
    return;
  }

  let pathname = "/";
  try {
    const url = new URL(req.url || "/", `http://${req.headers.host}`);
    pathname = decodeURIComponent(url.pathname);
  } catch {
    send(res, 400, "Bad request", { "Content-Type": "text/plain; charset=utf-8" });
    return;
  }

  const safePath = normalize(pathname).replace(/^[/\\]+/, "");
  let filePath = resolve(distRoot, safePath);

  if (filePath !== distRoot && !filePath.startsWith(`${distRoot}${sep}`)) {
    send(res, 403, "Forbidden", { "Content-Type": "text/plain; charset=utf-8" });
    return;
  }

  if (pathname === "/" || !extname(filePath)) {
    filePath = join(distRoot, "index.html");
  }

  if (!existsSync(filePath) || !statSync(filePath).isFile()) {
    send(res, 404, "Not found", { "Content-Type": "text/plain; charset=utf-8" });
    return;
  }

  const ext = extname(filePath);
  const cacheControl = filePath.includes(`${sep}assets${sep}`)
    ? "public, max-age=31536000, immutable"
    : "no-store";

  res.writeHead(200, withSecurityHeaders({
    "Cache-Control": cacheControl,
    "Content-Type": mimeTypes[ext] || "application/octet-stream",
  }));
  createReadStream(filePath).pipe(res);
}).listen(port, () => {
  console.log(`SwingGrade preview server running at http://localhost:${port}`);
});
