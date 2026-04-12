from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute project quality gate score from research and ops artifacts")
    parser.add_argument("--eval-report", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--load-report", type=str, default="")
    parser.add_argument("--feedback-report", type=str, default="")
    return parser.parse_args()


def read_json(path: str) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def clamp_score(value: float) -> float:
    return float(np.clip(value, 0.0, 10.0))


def main() -> None:
    args = parse_args()

    eval_report = read_json(args.eval_report)
    if eval_report is None:
        raise FileNotFoundError(f"Evaluation report not found: {args.eval_report}")

    load_report = read_json(args.load_report)
    feedback_report = read_json(args.feedback_report)

    best_model = eval_report.get("best_model", {})
    best_name = str(best_model.get("name", "unknown"))

    ablations = eval_report.get("ablations", [])
    top = ablations[0] if ablations else {}
    best_f1 = float(top.get("f1", 0.0))
    best_pr_auc = float(top.get("pr_auc", 0.0))

    cross_scene = eval_report.get("diagnostics", {}).get("cross_scene", {})
    cross_std = float(cross_scene.get("summary", {}).get("f1_std", 0.35))

    research_score = (
        4.2 * min(1.0, best_f1 / 0.80)
        + 3.2 * min(1.0, best_pr_auc / 0.82)
        + 2.6 * max(0.0, 1.0 - min(cross_std, 0.35) / 0.35)
    )

    practical_score = 5.5
    load_checks = {
        "present": False,
        "p95_ms": None,
        "throughput_fps": None,
        "slo_pass": False,
    }
    if load_report is not None:
        summary = load_report.get("summary", {})
        p95 = float(summary.get("latency_ms", {}).get("p95", 9999.0))
        fps = float(summary.get("throughput_fps", 0.0))
        slo_pass = bool(summary.get("slo", {}).get("pass", False))

        load_checks = {
            "present": True,
            "p95_ms": p95,
            "throughput_fps": fps,
            "slo_pass": slo_pass,
        }
        practical_score += 2.0 * min(1.0, fps / 20.0)
        practical_score += 2.5 * max(0.0, 1.0 - min(p95, 220.0) / 220.0)
        if slo_pass:
            practical_score += 0.8
    else:
        practical_score -= 1.8

    feedback_checks = {
        "present": False,
        "f1_delta": None,
    }
    if feedback_report is not None:
        f1_delta = float(feedback_report.get("improvement", {}).get("f1_delta", 0.0))
        feedback_checks = {
            "present": True,
            "f1_delta": f1_delta,
        }
        practical_score += 0.7 if f1_delta >= 0.01 else 0.2
        research_score += 0.4 if f1_delta >= 0.01 else 0.0
    else:
        practical_score -= 0.8

    research_score = clamp_score(research_score)
    practical_score = clamp_score(practical_score)
    overall = clamp_score(0.55 * research_score + 0.45 * practical_score)

    gates = {
        "best_f1_gte_0_75": bool(best_f1 >= 0.75),
        "best_pr_auc_gte_0_78": bool(best_pr_auc >= 0.78),
        "cross_scene_f1_std_lte_0_15": bool(cross_std <= 0.15),
        "load_slo_pass": bool(load_checks.get("slo_pass", False)),
        "feedback_f1_delta_gte_0_01": bool((feedback_checks.get("f1_delta") or 0.0) >= 0.01),
    }

    result = {
        "scores": {
            "research": research_score,
            "practical": practical_score,
            "overall": overall,
        },
        "best_model": {
            "name": best_name,
            "f1": best_f1,
            "pr_auc": best_pr_auc,
        },
        "cross_scene": {
            "f1_std": cross_std,
        },
        "checks": {
            "load": load_checks,
            "feedback": feedback_checks,
            "quality_gates": gates,
        },
        "recommendation": {
            "ready_for_interviews": bool(overall >= 8.8),
            "ready_for_research_submission": bool(research_score >= 8.5),
            "ready_for_pilot_deployment": bool(practical_score >= 8.5),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"Quality gate complete | overall={overall:.2f}/10 research={research_score:.2f}/10 practical={practical_score:.2f}/10",
        flush=True,
    )
    print(f"Saved quality gate report to {output_path}", flush=True)


if __name__ == "__main__":
    main()
