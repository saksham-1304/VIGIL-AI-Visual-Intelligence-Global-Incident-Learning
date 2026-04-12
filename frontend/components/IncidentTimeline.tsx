import type { IncidentEvent } from "@/lib/types";

interface IncidentTimelineProps {
  events: IncidentEvent[];
}

export default function IncidentTimeline({ events }: IncidentTimelineProps) {
  const timeline = events.slice(0, 30);

  return (
    <section className="timeline">
      <h2 className="panel-title">Incident Timeline</h2>
      {timeline.length === 0 ? (
        <p className="timeline-item__meta">No incident events captured yet.</p>
      ) : (
        <ul className="timeline-list">
          {timeline.map((event) => (
            <li key={event.id} className="timeline-item">
              <p className="timeline-item__meta">
                {new Date(event.timestamp).toLocaleString()} | {event.camera_id} | {event.event_type}
              </p>
              <p className="timeline-item__desc">{event.description}</p>
              {event.actions?.length ? (
                <p className="timeline-item__meta">Actions: {event.actions.join(", ")}</p>
              ) : null}
              <span className={`severity severity--${event.severity}`}>{event.severity}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
