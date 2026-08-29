#!/usr/bin/env python3
"""Score model responses with the official VSI-Bench aggregation logic."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import copy
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSE_DIR = REPO_ROOT / "gemini_response"
DEFAULT_GROUND_TRUTH_DIR = REPO_ROOT / "scannet_questions"

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.0,
    "MRA:.5:.95:.05": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute VSI-Bench benchmark numbers from saved model responses."
    )
    parser.add_argument(
        "--response-dir",
        type=Path,
        default=DEFAULT_RESPONSE_DIR,
        help=f"Directory containing response JSON files. Default: {DEFAULT_RESPONSE_DIR}",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_DIR,
        help=f"Directory containing ground-truth question exports. Default: {DEFAULT_GROUND_TRUTH_DIR}",
    )
    parser.add_argument(
        "--scene",
        action="append",
        help="Optional scene id to score. Repeat the flag to score multiple scenes.",
    )
    parser.add_argument(
        "--save-json",
        nargs="?",
        const="auto",
        help="Save the score report. Pass no value to auto-name the file, or pass a path explicitly.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def fuzzy_matching(pred: str) -> str:
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred: str, target: str) -> float:
    return 1.0 if pred.lower() == target.lower() else 0.0


def to_float(pred: str) -> Optional[float]:
    try:
        return float(pred)
    except Exception:
        return None


def abs_dist_norm(pred: float, target: float) -> float:
    return abs(pred - target) / target


def mean_relative_accuracy(pred: float, target: float, start: float = 0.5, end: float = 0.95, interval: float = 0.05) -> float:
    thresholds = []
    current = start
    while current <= end + 1e-9:
        thresholds.append(round(current, 10))
        current += interval
    thresholds.append(1.0)
    normalized_error = abs_dist_norm(pred, target)
    hits = [normalized_error <= 1 - threshold for threshold in thresholds]
    return sum(hits) / len(hits)


def vsibench_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    doc["prediction"] = results[0]
    if doc["question_type"] in MCA_QUESTION_TYPES:
        doc["accuracy"] = exact_match(fuzzy_matching(doc["prediction"]), doc["ground_truth"])
    elif doc["question_type"] in NA_QUESTION_TYPES:
        try:
            doc["MRA:.5:.95:.05"] = mean_relative_accuracy(
                to_float(fuzzy_matching(doc["prediction"])),
                to_float(doc["ground_truth"]),
            )
        except TypeError:
            doc["MRA:.5:.95:.05"] = WORST_CASE_FOR_METRICS["MRA:.5:.95:.05"]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")
    return {"vsibench_score": doc}


def vsibench_aggregate_results(results: List[Dict[str, Any]]) -> OrderedDict:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["question_type"]].append(row)

    output: Dict[str, float] = {}
    for question_type, rows in grouped.items():
        if question_type in MCA_QUESTION_TYPES:
            output[f"{question_type}_accuracy"] = statistics.mean(row["accuracy"] for row in rows)
        elif question_type in NA_QUESTION_TYPES:
            output[f"{question_type}_MRA:.5:.95:.05"] = statistics.mean(row["MRA:.5:.95:.05"] for row in rows)
        else:
            raise ValueError(f"Unknown question type: {question_type}")

    direction_keys = [
        key
        for key in [
            "object_rel_direction_easy_accuracy",
            "object_rel_direction_medium_accuracy",
            "object_rel_direction_hard_accuracy",
        ]
        if key in output
    ]
    if direction_keys:
        output["object_rel_direction_accuracy"] = (
            sum(output[key] for key in direction_keys) / len(direction_keys)
        )

    output["overall"] = sum(output.values()) / len(output)

    results_out: OrderedDict[str, Any] = OrderedDict()
    results_out["overall"] = output["overall"] * 100.0
    for question_type in [
        "object_counting",
        "object_abs_distance",
        "object_size_estimation",
        "room_size_estimation",
        "object_rel_distance",
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
        "object_rel_direction",
        "route_planning",
        "obj_appearance_order",
    ]:
        for metric in ["accuracy", "MRA:.5:.95:.05"]:
            key = f"{question_type}_{metric}"
            if key in output:
                results_out[key] = output[key] * 100.0

    tabulated_keys = ", ".join(results_out.keys())
    tabulated_results = ", ".join(f"{value:.3f}" for value in results_out.values())
    results_out["tabulated_keys"] = tabulated_keys
    results_out["tabulated_results"] = tabulated_results
    return results_out


def scene_id_from_response_file(path: Path) -> str:
    return path.name.replace("_gemini_response.json", "")


def ground_truth_path_for_scene(scene_id: str, ground_truth_dir: Path) -> Path:
    return ground_truth_dir / f"{scene_id}_questions.json"


def score_scene(scene_id: str, response_path: Path, ground_truth_path: Path) -> Dict[str, Any]:
    response_data = load_json(response_path)
    ground_truth_data = load_json(ground_truth_path)

    response_by_question = {
        item["question"]: item.get("gemini_answer", "")
        for item in response_data.get("questions", [])
    }

    scored_docs: List[Dict[str, Any]] = []
    missing_predictions: List[str] = []
    for doc in ground_truth_data.get("questions", []):
        prediction = response_by_question.get(doc["question"])
        if prediction is None:
            missing_predictions.append(doc["question"])
            continue
        processed = vsibench_process_results(copy.deepcopy(doc), [prediction])["vsibench_score"]
        scored_docs.append(processed)

    aggregate = vsibench_aggregate_results(scored_docs) if scored_docs else {}
    return {
        "scene_name": scene_id,
        "response_file": str(response_path),
        "ground_truth_file": str(ground_truth_path),
        "num_questions_in_ground_truth": len(ground_truth_data.get("questions", [])),
        "num_questions_scored": len(scored_docs),
        "num_missing_predictions": len(missing_predictions),
        "missing_predictions": missing_predictions,
        "benchmark_scores": dict(aggregate),
    }


def build_report(response_dir: Path, ground_truth_dir: Path, scenes: Optional[List[str]]) -> Dict[str, Any]:
    response_files = sorted(response_dir.glob("*_gemini_response.json"))
    if scenes:
        wanted = set(scenes)
        response_files = [path for path in response_files if scene_id_from_response_file(path) in wanted]

    scene_reports: List[Dict[str, Any]] = []
    all_scored_docs: List[Dict[str, Any]] = []

    for response_path in response_files:
        scene_id = scene_id_from_response_file(response_path)
        ground_truth_path = ground_truth_path_for_scene(scene_id, ground_truth_dir)
        if not ground_truth_path.exists():
            scene_reports.append(
                {
                    "scene_name": scene_id,
                    "response_file": str(response_path),
                    "ground_truth_file": str(ground_truth_path),
                    "error": "Ground-truth file not found.",
                }
            )
            continue

        scene_report = score_scene(scene_id, response_path, ground_truth_path)
        scene_reports.append(scene_report)

        if "error" not in scene_report:
            response_data = load_json(response_path)
            ground_truth_data = load_json(ground_truth_path)
            response_by_question = {
                item["question"]: item.get("gemini_answer", "")
                for item in response_data.get("questions", [])
            }
            for doc in ground_truth_data.get("questions", []):
                prediction = response_by_question.get(doc["question"])
                if prediction is None:
                    continue
                all_scored_docs.append(
                    vsibench_process_results(copy.deepcopy(doc), [prediction])["vsibench_score"]
                )

    overall_scores = dict(vsibench_aggregate_results(all_scored_docs)) if all_scored_docs else {}
    return {
        "response_dir": str(response_dir),
        "ground_truth_dir": str(ground_truth_dir),
        "num_scenes_scored": sum(1 for report in scene_reports if "benchmark_scores" in report),
        "num_total_questions_scored": len(all_scored_docs),
        "scene_reports": scene_reports,
        "overall_benchmark_scores": overall_scores,
    }


def dump_json(data: Dict[str, Any], pretty: bool) -> None:
    if pretty:
        print(json.dumps(data, indent=2, ensure_ascii=True))
    else:
        print(json.dumps(data, ensure_ascii=True))


def print_summary(report: Dict[str, Any]) -> None:
    print("VSI-Bench benchmark summary")
    print()
    for scene_report in report["scene_reports"]:
        print(f"Scene: {scene_report['scene_name']}")
        if "error" in scene_report:
            print(f"  Error: {scene_report['error']}")
            print()
            continue

        print(
            f"  Coverage: {scene_report['num_questions_scored']}/"
            f"{scene_report['num_questions_in_ground_truth']} questions scored"
        )
        for key, value in scene_report["benchmark_scores"].items():
            if key.startswith("tabulated_"):
                continue
            print(f"  {key}: {value}")
        print()

    print("Overall")
    for key, value in report["overall_benchmark_scores"].items():
        if key.startswith("tabulated_"):
            continue
        print(f"  {key}: {value}")


def save_report(report: Dict[str, Any], pretty: bool, save_json: Optional[str]) -> Optional[Path]:
    if not save_json:
        return None
    if save_json == "auto":
        response_dir = Path(report["response_dir"])
        output_path = response_dir / "gemini_vsibench_scores.json"
    else:
        output_path = Path(save_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    else:
        output_path.write_text(json.dumps(report, ensure_ascii=True) + "\n")
    return output_path


def main() -> int:
    args = parse_args()
    report = build_report(args.response_dir, args.ground_truth_dir, args.scene)
    print_summary(report)
    print()
    dump_json(report, args.pretty)
    saved_path = save_report(report, args.pretty, args.save_json)
    if saved_path:
        print()
        print(f"Saved JSON to: {saved_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
