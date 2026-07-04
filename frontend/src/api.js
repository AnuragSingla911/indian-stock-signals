const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export async function fetchPredictions() {
  const res = await fetch(`${API_BASE}/api/predictions`);
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

export { API_BASE };
