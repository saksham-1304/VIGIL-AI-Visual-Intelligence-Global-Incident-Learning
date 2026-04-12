import type { Alert } from "@/lib/types";

interface AlertRailProps {
  alerts: Alert[];
}

export default function AlertRail({ alerts }: AlertRailProps) {
  const latest = alerts.slice(0, 12);

  return (
    <aside className="alert-rail">
      <h2 className="panel-title">Alert Rail</h2>
      {latest.length === 0 ? (
        <p className="timeline-item__meta">No alerts yet.</p>
      ) : (
        <ul className="alert-list">
          {latest.map((alert) => (
            <li key={alert.id} className="alert-item">
              <p className="alert-item__meta">
                {new Date(alert.created_at).toLocaleTimeString()} | {alert.title}
              </p>
              <p className="alert-item__msg">{alert.message}</p>
              <span className={`severity severity--${alert.severity}`}>{alert.severity}</span>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
