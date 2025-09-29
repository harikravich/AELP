import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  define: {
    'process.env.NODE_ENV': JSON.stringify('development'),
  },
  optimizeDeps: {
    force: true,
  },
  server: {
    host: "::",
    port: 8080,
    proxy: {
      // Avoid CORS in dev: call /api/* on Vite and proxy to Next.js (3000)
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  // Disable Lovable tagger to avoid JSX dev runtime (_jsxDEV) errors
  plugins: [react({ jsxImportSource: 'react', fastRefresh: true })],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
