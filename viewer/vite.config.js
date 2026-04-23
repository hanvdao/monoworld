import { defineConfig } from 'vite';
import { resolve } from 'path';

// MonoWorld viewer config.
// We serve generated scene files (.ply, scenes.json) directly from
// ../data/outputs/ via Vite's publicDir mechanism. That way the python
// pipeline's outputs are accessible at the URL root without any copy step.
export default defineConfig({
  root: '.',
  publicDir: resolve(__dirname, '../data/outputs'),
  server: {
    port: 5173,
    open: true,
    fs: {
      // allow Vite to read files outside the viewer dir (project root)
      allow: [resolve(__dirname, '..')],
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
