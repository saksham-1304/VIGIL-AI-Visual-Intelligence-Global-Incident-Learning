export type Severity = "low" | "medium" | "high" | "critical";

export interface Detection {
  label: string;
  confidence: number;
  bbox: number[];
  track_id?: number;
}

export interface IncidentEvent {
  id: string;
  timestamp: string;
  camera_id: string;
  event_type: string;
  severity: Severity;
  description: string;
  detections: Detection[];
  actions: string[];
  anomaly_score: number | null;
  metadata: Record<string, unknown>;
}

export interface Alert {
  id: string;
  created_at: string;
  event_id: string;
  title: string;
  severity: Severity;
  message: string;
}

export interface HealthStatus {
  status: string;
  app: string;
  stream_running: boolean;
  websocket_clients: number;
}
