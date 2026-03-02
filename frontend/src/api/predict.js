const API_BASE = "http://localhost:8000";

export async function checkHealth() {
  const resp = await fetch(`${API_BASE}/health`);
  if (!resp.ok) throw new Error("Backend unreachable");
  return resp.json();
}

export async function predictImage(file, topK = 3, threshold = 0.5) {
  const form = new FormData();
  form.append("file", file);

  const resp = await fetch(
    `${API_BASE}/predict?top_k=${topK}&threshold=${threshold}`,
    { method: "POST", body: form }
  );

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${resp.status}`);
  }

  return resp.json();
}
