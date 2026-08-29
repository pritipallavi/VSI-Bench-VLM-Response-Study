#!/usr/bin/env python3
"""Compare model responses against VSI-Bench ground truth exports."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSE_DIR = REPO_ROOT / "data" / "vsi_bench" / "gemini_response"
DEFAULT_GROUND_TRUTH_DIR = REPO_ROOT / "data" / "vsi_bench" / "exports"

MCQ_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Gemini VSI-Bench answers against ground-truth exports."
    )
    parser.add_argument(
        "--response-dir",
        type=Path,
        default=DEFAULT_RESPONSE_DIR,
        help=f"Directory containing model response JSON files. Default: {DEFAULT_RESPONSE_DIR}",
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
        help="Optional scene id to compare. Repeat the flag to compare multiple scenes.",
    )
    parser.add_argument(
        "--save-json",
        nargs="?",
        const="auto",
        help="Save the comparison report. Pass no value to auto-name the file, or pass a path explicitly.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--save-csv",
        nargs="?",
        const="auto",
        help="Save per-question comparison rows as CSV. Pass no value to auto-name the file, or pass a path explicitly.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def extract_number(text: Any) -> Optional[float]:
    if text is None:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", str(text))
    if not match:
        return None
    return float(match.group(0))


def extract_option_letter(text: Any) -> Optional[str]:
    if text is None:
        return None
    match = re.search(r"\b([A-Z])\b", str(text).upper())
    if not match:
        return None
    return match.group(1)


def scene_id_from_response_file(path: Path) -> str:
    return path.name.replace("_gemini_response.json", "")


def ground_truth_path_for_scene(scene_id: str, ground_truth_dir: Path) -> Path:
    return ground_truth_dir / f"{scene_id}_questions.json"


def compare_question(response_q: Dict[str, Any], gt_q: Dict[str, Any]) -> Dict[str, Any]:
    question_type = gt_q.get("question_type")
    row: Dict[str, Any] = {
        "question": gt_q.get("question"),
        "question_type": question_type,
        "gemini_answer": response_q.get("gemini_answer"),
        "ground_truth": gt_q.get("ground_truth"),
    }

    if question_type in MCQ_QUESTION_TYPES:
        pred = extract_option_letter(response_q.get("gemini_answer"))
        target = extract_option_letter(gt_q.get("ground_truth"))
        row["evaluation_type"] = "mcq"
        row["parsed_prediction"] = pred
        row["parsed_ground_truth"] = target
        row["is_correct"] = pred == target if pred and target else False
        return row

    pred_num = extract_number(response_q.get("gemini_answer"))
    target_num = extract_number(gt_q.get("ground_truth"))
    absolute_error = None if pred_num is None or target_num is None else round(abs(pred_num - target_num), 4)
    row["evaluation_type"] = "numeric"
    row["parsed_prediction"] = pred_num
    row["parsed_ground_truth"] = target_num
    row["absolute_error"] = absolute_error
    return row


def summarize_scene(question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric = [row for row in question_results if row["evaluation_type"] == "numeric"]
    mcq = [row for row in question_results if row["evaluation_type"] == "mcq"]
    numeric_errors = [row["absolute_error"] for row in numeric if row["absolute_error"] is not None]
    mcq_correct = sum(1 for row in mcq if row.get("is_correct"))

    by_type: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in question_results:
        grouped[row["question_type"]].append(row)

    for question_type, rows in sorted(grouped.items()):
        numeric_rows = [row for row in rows if row["evaluation_type"] == "numeric"]
        mcq_rows = [row for row in rows if row["evaluation_type"] == "mcq"]
        entry: Dict[str, Any] = {"num_questions": len(rows)}
        if numeric_rows:
            errors = [row["absolute_error"] for row in numeric_rows if row["absolute_error"] is not None]
            entry["evaluation_type"] = "numeric"
            entry["mean_absolute_error"] = round(statistics.mean(errors), 4) if errors else None
            entry["max_absolute_error"] = round(max(errors), 4) if errors else None
        else:
            correct = sum(1 for row in mcq_rows if row.get("is_correct"))
            entry["evaluation_type"] = "mcq"
            entry["num_correct"] = correct
            entry["accuracy"] = round(correct / len(mcq_rows), 4) if mcq_rows else None
        by_type[question_type] = entry

    summary = {
        "num_questions": len(question_results),
        "num_numeric_questions": len(numeric),
        "num_mcq_questions": len(mcq),
        "numeric_mean_absolute_error": round(statistics.mean(numeric_errors), 4) if numeric_errors else None,
        "numeric_median_absolute_error": round(statistics.median(numeric_errors), 4) if numeric_errors else None,
        "numeric_max_absolute_error": round(max(numeric_errors), 4) if numeric_errors else None,
        "numeric_num_exact_matches": sum(1 for err in numeric_errors if err == 0),
        "mcq_num_correct": mcq_correct,
        "mcq_accuracy": round(mcq_correct / len(mcq), 4) if mcq else None,
        "per_question_type": by_type,
    }
    return summary


def compare_scene(scene_id: str, response_path: Path, ground_truth_path: Path) -> Dict[str, Any]:
    response_data = load_json(response_path)
    gt_data = load_json(ground_truth_path)

    gt_by_question = {item["question"]: item for item in gt_data.get("questions", [])}
    question_results: List[Dict[str, Any]] = []
    unmatched_response_questions: List[str] = []

    for response_q in response_data.get("questions", []):
        question = response_q.get("question")
        gt_q = gt_by_question.get(question)
        if gt_q is None:
            unmatched_response_questions.append(question)
            continue
        question_results.append(compare_question(response_q, gt_q))

    matched_questions = {row["question"] for row in question_results}
    missing_response_questions = [
        item["question"] for item in gt_data.get("questions", []) if item["question"] not in matched_questions
    ]

    return {
        "scene_name": scene_id,
        "response_file": str(response_path),
        "ground_truth_file": str(ground_truth_path),
        "summary": summarize_scene(question_results),
        "question_results": question_results,
        "num_unmatched_response_questions": len(unmatched_response_questions),
        "num_missing_response_questions": len(missing_response_questions),
        "unmatched_response_questions": unmatched_response_questions,
        "missing_response_questions": missing_response_questions,
    }


def build_report(response_dir: Path, ground_truth_dir: Path, scenes: Optional[List[str]]) -> Dict[str, Any]:
    response_files = sorted(response_dir.glob("*_gemini_response.json"))
    if scenes:
        wanted = set(scenes)
        response_files = [path for path in response_files if scene_id_from_response_file(path) in wanted]

    scene_reports = []
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
        scene_reports.append(compare_scene(scene_id, response_path, ground_truth_path))

    valid_reports = [report for report in scene_reports if "summary" in report]
    numeric_errors = []
    total_mcq = 0
    total_mcq_correct = 0
    total_questions = 0
    for report in valid_reports:
        total_questions += report["summary"]["num_questions"]
        total_mcq += report["summary"]["num_mcq_questions"]
        total_mcq_correct += report["summary"]["mcq_num_correct"]
        for row in report["question_results"]:
            if row["evaluation_type"] == "numeric" and row["absolute_error"] is not None:
                numeric_errors.append(row["absolute_error"])

    overall_summary = {
        "num_scenes_compared": len(valid_reports),
        "num_questions_compared": total_questions,
        "overall_numeric_mean_absolute_error": round(statistics.mean(numeric_errors), 4) if numeric_errors else None,
        "overall_numeric_median_absolute_error": round(statistics.median(numeric_errors), 4) if numeric_errors else None,
        "overall_numeric_max_absolute_error": round(max(numeric_errors), 4) if numeric_errors else None,
        "overall_mcq_accuracy": round(total_mcq_correct / total_mcq, 4) if total_mcq else None,
        "overall_mcq_num_correct": total_mcq_correct,
        "overall_mcq_num_questions": total_mcq,
    }

    return {
        "response_dir": str(response_dir),
        "ground_truth_dir": str(ground_truth_dir),
        "scene_reports": scene_reports,
        "overall_summary": overall_summary,
    }


def dump_json(data: Dict[str, Any], pretty: bool) -> None:
    if pretty:
        print(json.dumps(data, indent=2, ensure_ascii=True))
    else:
        print(json.dumps(data, ensure_ascii=True))


def print_text_summary(report: Dict[str, Any]) -> None:
    print("Scene comparison summary")
    print()
    for scene_report in report["scene_reports"]:
        print(f"Scene: {scene_report['scene_name']}")
        if "error" in scene_report:
            print(f"  Error: {scene_report['error']}")
            print()
            continue

        summary = scene_report["summary"]
        print(
            "  Stats: "
            f"{summary['num_questions']} questions, "
            f"{summary['num_numeric_questions']} numeric, "
            f"{summary['num_mcq_questions']} mcq"
        )
        if summary["num_numeric_questions"]:
            print(
                "  Numeric: "
                f"mean abs error={summary['numeric_mean_absolute_error']}, "
                f"median abs error={summary['numeric_median_absolute_error']}, "
                f"max abs error={summary['numeric_max_absolute_error']}, "
                f"exact matches={summary['numeric_num_exact_matches']}"
            )
        if summary["num_mcq_questions"]:
            print(
                "  MCQ: "
                f"{summary['mcq_num_correct']}/{summary['num_mcq_questions']} correct "
                f"({summary['mcq_accuracy']:.2%})"
            )

        print("  Per-question results:")
        for index, row in enumerate(scene_report["question_results"], start=1):
            if row["evaluation_type"] == "numeric":
                print(
                    f"    {index}. [{row['question_type']}] abs_error={row['absolute_error']} "
                    f"| pred={row['gemini_answer']} | gt={row['ground_truth']}"
                )
            else:
                status = "correct" if row["is_correct"] else "wrong"
                print(
                    f"    {index}. [{row['question_type']}] {status} "
                    f"| pred={row['gemini_answer']} | gt={row['ground_truth']}"
                )
        print()

    overall = report["overall_summary"]
    print("Overall")
    print(
        f"  Scenes={overall['num_scenes_compared']}, "
        f"Questions={overall['num_questions_compared']}, "
        f"Numeric mean abs error={overall['overall_numeric_mean_absolute_error']}, "
        f"MCQ accuracy={overall['overall_mcq_accuracy']:.2%}" if overall["overall_mcq_accuracy"] is not None
        else f"  Scenes={overall['num_scenes_compared']}, Questions={overall['num_questions_compared']}"
    )


def save_report(report: Dict[str, Any], pretty: bool, save_json: Optional[str]) -> Optional[Path]:
    if not save_json:
        return None
    if save_json == "auto":
        output_path = REPO_ROOT / "gemini_response" / "gemini_comparison_report.json"
    else:
        output_path = Path(save_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    else:
        output_path.write_text(json.dumps(report, ensure_ascii=True) + "\n")
    return output_path


def flatten_question_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene_report in report["scene_reports"]:
        if "question_results" not in scene_report:
            continue
        for index, question_result in enumerate(scene_report["question_results"], start=1):
            rows.append(
                {
                    "scene_name": scene_report["scene_name"],
                    "question_index": index,
                    "question_type": question_result.get("question_type"),
                    "evaluation_type": question_result.get("evaluation_type"),
                    "question": question_result.get("question"),
                    "gemini_answer": question_result.get("gemini_answer"),
                    "ground_truth": question_result.get("ground_truth"),
                    "parsed_prediction": question_result.get("parsed_prediction"),
                    "parsed_ground_truth": question_result.get("parsed_ground_truth"),
                    "absolute_error": question_result.get("absolute_error"),
                    "is_correct": question_result.get("is_correct"),
                }
            )
    return rows


def save_csv_report(report: Dict[str, Any], save_csv: Optional[str]) -> Optional[Path]:
    if not save_csv:
        return None
    if save_csv == "auto":
        output_path = REPO_ROOT / "gemini_response" / "gemini_comparison_report.csv"
    else:
        output_path = Path(save_csv).expanduser()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = flatten_question_rows(report)
    fieldnames = [
        "scene_name",
        "question_index",
        "question_type",
        "evaluation_type",
        "question",
        "gemini_answer",
        "ground_truth",
        "parsed_prediction",
        "parsed_ground_truth",
        "absolute_error",
        "is_correct",
    ]
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> int:
    args = parse_args()
    report = build_report(args.response_dir, args.ground_truth_dir, args.scene)
    print_text_summary(report)
    print()
    dump_json(report, args.pretty)
    saved_json_path = save_report(report, args.pretty, args.save_json)
    saved_csv_path = save_csv_report(report, args.save_csv)
    if saved_json_path:
        print()
        print(f"Saved JSON to: {saved_json_path}")
    if saved_csv_path:
        print(f"Saved CSV to: {saved_csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
