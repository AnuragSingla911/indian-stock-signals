const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

// Static mode (e.g. GitHub Pages): load a pre-generated predictions.json bundled with the
// site instead of calling the live backend.
const STATIC_MODE =
  import.meta.env.VITE_STATIC === '1' || import.meta.env.VITE_STATIC === 'true';

export async function fetchPredictions() {
  const url = STATIC_MODE
    ? `${import.meta.env.BASE_URL}predictions.json`
    : `${API_BASE}/api/predictions`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

export { API_BASE, STATIC_MODE };
