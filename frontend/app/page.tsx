"use client";

import { useEffect, useMemo, useState } from "react";
import AlertRail from "@/components/AlertRail";
import IncidentTimeline from "@/components/IncidentTimeline";
import LiveFeedCard from "@/components/LiveFeedCard";
import MetricsStrip from "@/components/MetricsStrip";
import {
  getAlerts,
  getEvents,
  getHealth,
  startStream,
  stopStream,
  websocketUrl,
} from "@/lib/api";
import type { Alert, HealthStatus, IncidentEvent } from "@/lib/types";

export default function HomePage() {
  const [events, setEvents] = useState<IncidentEvent[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [statusText, setStatusText] = useState("Monitoring pipeline idle");
  const [error, setError] = useState<string | null>(null);

  const latestEvent = useMemo(() => events[0] ?? null, [events]);

  async function refresh() {
    try {
      const [healthRes, eventsRes, alertsRes] = await Promise.all([
        getHealth(),
        getEvents(40),
        getAlerts(40),
      ]);
      setHealth(healthRes);
      setEvents(eventsRes);
      setAlerts(alertsRes);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to fetch backend data");
    }
  }

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const ws = new WebSocket(websocketUrl());

    ws.onopen = () => setStatusText("Realtime channel connected");
    ws.onerror = () => setStatusText("Realtime channel unavailable");
    ws.onclose = () => setStatusText("Realtime channel disconnected");

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as {
          type: string;
          payload?: IncidentEvent;
          alerts?: Alert[];
        };

        if (payload.type === "event" && payload.payload) {
          setEvents((prev) => [payload.payload as IncidentEvent, ...prev].slice(0, 80));
        }

        if (payload.alerts?.length) {
          setAlerts((prev) => [...payload.alerts as Alert[], ...prev].slice(0, 80));
        }
      } catch {
        // Ignore malformed messages from non-dashboard clients.
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  async function handleStart() {
    try {
      await startStream({ source: "webcam", camera_id: "cam-01" });
      setStatusText("Live stream inference running");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start stream");
    }
  }

  async function handleStop() {
    try {
      await stopStream();
      setStatusText("Stream processing stopped");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to stop stream");
    }
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <div className="hero__badge">Real-Time Multimodal Incident Intelligence</div>
        <h1 className="hero__title">Control Room for Safety-Critical Visual Intelligence</h1>
        <p className="hero__text">
          Live detection, identity tracking, action understanding, anomaly scoring, and
          natural-language incident narratives in one operational cockpit.
        </p>
        <div className="hero__actions">
          <button className="btn btn--primary" onClick={handleStart}>
            Start Live Inference
          </button>
          <button className="btn btn--ghost" onClick={handleStop}>
            Stop Stream
          </button>
        </div>
        <p className="hero__status">{statusText}</p>
        {error ? <p className="hero__error">{error}</p> : null}
      </section>

      <MetricsStrip health={health} events={events} alerts={alerts} />

      <section className="layout-grid">
        <LiveFeedCard latestEvent={latestEvent} />
        <AlertRail alerts={alerts} />
      </section>

      <IncidentTimeline events={events} />
    </main>
  );
}
