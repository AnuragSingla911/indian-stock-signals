// Copies the pipeline-generated predictions.json into public/ so it can be bundled into a
// static build (e.g. GitHub Pages) and served without a live backend.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, '../../backend/app/data/predictions.json');
const dest = resolve(here, '../public/predictions.json');

if (!existsSync(src)) {
  console.error(`predictions.json not found at ${src} — run the ML pipeline first.`);
  process.exit(1);
}

mkdirSync(dirname(dest), { recursive: true });
copyFileSync(src, dest);
console.log(`Copied ${src} -> ${dest}`);
