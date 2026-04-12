"use client";

import { useEffect, useMemo, useRef } from "react";
import type { IncidentEvent } from "@/lib/types";

interface LiveFeedCardProps {
  latestEvent: IncidentEvent | null;
}

function drawOverlay(canvas: HTMLCanvasElement, event: IncidentEvent | null) {
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }

  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!event?.detections?.length) {
    return;
  }

  const maxX = Math.max(640, ...event.detections.map((d) => d.bbox[2] || 0));
  const maxY = Math.max(360, ...event.detections.map((d) => d.bbox[3] || 0));
  const scaleX = canvas.width / maxX;
  const scaleY = canvas.height / maxY;

  event.detections.forEach((detection) => {
    const [x1, y1, x2, y2] = detection.bbox;
    const x = x1 * scaleX;
    const y = y1 * scaleY;
    const width = (x2 - x1) * scaleX;
    const height = (y2 - y1) * scaleY;

    context.strokeStyle = "#2ad3b3";
    context.lineWidth = 2;
    context.strokeRect(x, y, width, height);

    const label = `${detection.label} ${(detection.confidence * 100).toFixed(0)}% #${detection.track_id ?? "-"}`;
    context.font = "12px IBM Plex Mono";
    const metrics = context.measureText(label);
    context.fillStyle = "rgba(4, 26, 22, 0.85)";
    context.fillRect(x, y - 18, metrics.width + 10, 18);
    context.fillStyle = "#9ef4dd";
    context.fillText(label, x + 5, y - 5);
  });
}

export default function LiveFeedCard({ latestEvent }: LiveFeedCardProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const eventSummary = useMemo(() => {
    if (!latestEvent) {
      return "Waiting for incidents";
    }
    return `${latestEvent.event_type} | ${latestEvent.description}`;
  }, [latestEvent]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !navigator.mediaDevices?.getUserMedia) {
      return;
    }

    let stream: MediaStream | null = null;
    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
      .then((mediaStream) => {
        stream = mediaStream;
        video.srcObject = mediaStream;
      })
      .catch(() => {
        // If webcam access is blocked, card still renders backend metadata and overlays.
      });

    return () => {
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    drawOverlay(canvas, latestEvent);
  }, [latestEvent]);

  return (
    <article className="live-card">
      <h2 className="panel-title">Live Feed + Detection Overlay</h2>
      <div className="live-card__stage">
        <video ref={videoRef} autoPlay muted playsInline />
        <canvas ref={canvasRef} width={1280} height={720} />
      </div>
      <p className="timeline-item__meta" style={{ marginTop: 10 }}>
        {eventSummary}
      </p>
    </article>
  );
}
