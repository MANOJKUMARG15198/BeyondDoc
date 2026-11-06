import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Built files are served by FastAPI at "/" (src/medical_rag/api/static/app). `npm run dev` proxies the API.
// base "./" and relative API paths let the same build run under a sub-path (e.g. /medrag/ behind nginx).
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "../src/medical_rag/api/static/app", emptyOutDir: true },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000", "/info": "http://127.0.0.1:8000", "/chunks": "http://127.0.0.1:8000" },
  },
});
