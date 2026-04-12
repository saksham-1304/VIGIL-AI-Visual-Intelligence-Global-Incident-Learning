import type { Alert, HealthStatus, IncidentEvent } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}

export function getHealth(): Promise<HealthStatus> {
  return fetchJson<HealthStatus>(`${API_BASE}/api/v1/health`);
}

export function getEvents(limit = 50): Promise<IncidentEvent[]> {
  return fetchJson<IncidentEvent[]>(`${API_BASE}/api/v1/events?limit=${limit}`);
}

export function getAlerts(limit = 50): Promise<Alert[]> {
  return fetchJson<Alert[]>(`${API_BASE}/api/v1/alerts?limit=${limit}`);
}

export function startStream(payload: { source: string; camera_id: string }) {
  return fetchJson(`${API_BASE}/api/v1/stream/start`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function stopStream() {
  return fetchJson(`${API_BASE}/api/v1/stream/stop`, { method: "POST" });
}

export function websocketUrl(): string {
  const url = new URL(API_BASE);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = "/ws/events";
  url.search = "";
  return url.toString();
}
