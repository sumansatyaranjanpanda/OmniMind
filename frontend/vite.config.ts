import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    // Every backend route the app calls needs a rule here. Anything missing is
    // served by Vite itself and comes back as a 404 with an empty body, which the
    // client can only report as a generic failure — `/auth` was missing, so signup
    // and login failed with "Could not create that account" no matter what the
    // API would actually have said, and every request went out unauthenticated.
    // Keep this list in sync with api/main.py's router registrations.
    proxy: Object.fromEntries(
      [
        '/api',
        '/auth',
        '/chat',
        '/conversations',
        '/documents',
        '/graph',
        '/search',
        '/health',
        '/voice',
      ].map((route) => [
        route,
        {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
          // SSE must not be buffered or the answer arrives in one lump at the end.
          ...(route === '/chat' ? { ws: false } : {}),
          // Voice is a WebSocket, so the proxy has to forward the HTTP upgrade
          // handshake. Without this the socket is served by Vite itself and the
          // connection closes immediately with no usable error.
          ...(route === '/voice' ? { ws: true } : {}),
          ...(route === '/api'
            ? { rewrite: (p: string) => p.replace(/^\/api/, '') }
            : {}),
        },
      ])
    ),
  },
});
