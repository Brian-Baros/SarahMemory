import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";
import { imagetools } from "vite-imagetools";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  // SARAHMEMORY_PATCH_NOTE 2026-06-24:
  // Dev server remains loopback-bound by default. Use SARAH_UI_DEV_HOST to
  // expose it intentionally; do not bind to all interfaces during local-first
  // development. The Flask backend remains the production served path.
  server: {
    host: process.env.SARAH_UI_DEV_HOST || "127.0.0.1",
    port: Number(process.env.SARAH_UI_DEV_PORT || 8080),
    strictPort: false,
    proxy: {
      "/api": {
        target: process.env.SARAH_API_BASE || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  plugins: [
    react(),
    imagetools(),
    mode === "development" && componentTagger()
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
