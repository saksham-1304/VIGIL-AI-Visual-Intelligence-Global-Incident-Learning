import type { Alert, HealthStatus, IncidentEvent } from "@/lib/types";

interface MetricsStripProps {
  health: HealthStatus | null;
  events: IncidentEvent[];
  alerts: Alert[];
}

export default function MetricsStrip({ health, events, alerts }: MetricsStripProps) {
  const criticalCount = events.filter((event) => event.severity === "critical").length;
  const avgAnomaly =
    events.length > 0
      ? events.reduce((sum, event) => sum + (event.anomaly_score ?? 0), 0) / events.length
      : 0;

  return (
    <section className="metric-strip">
      <article className="metric">
        <h4>System Status</h4>
        <p>{health?.status ?? "offline"}</p>
      </article>

      <article className="metric">
        <h4>Live Stream</h4>
        <p>{health?.stream_running ? "running" : "stopped"}</p>
      </article>

      <article className="metric">
        <h4>Critical Incidents</h4>
        <p>{criticalCount}</p>
      </article>

      <article className="metric">
        <h4>Avg Anomaly</h4>
        <p>{avgAnomaly.toFixed(2)} ({alerts.length} alerts)</p>
      </article>
    </section>
  );
}
