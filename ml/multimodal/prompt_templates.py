INCIDENT_TEMPLATE = """
You are a safety analyst.
Given a scene caption, detections, actions, and anomaly score, produce one concise incident explanation.
Output format:
- what_happened
- why_it_matters
- urgency_level
""".strip()


def build_prompt(caption: str, detections: list[str], actions: list[str], anomaly_score: float) -> str:
    return (
        f"Scene caption: {caption}\n"
        f"Detections: {', '.join(detections) if detections else 'none'}\n"
        f"Actions: {', '.join(actions) if actions else 'none'}\n"
        f"Anomaly score: {anomaly_score:.2f}\n"
        "Generate an incident explanation."
    )
