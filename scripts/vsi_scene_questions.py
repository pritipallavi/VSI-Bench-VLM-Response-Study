#!/usr/bin/env python3
"""Inspect VSI-Bench questions for a ScanNet scene.

This utility supports three paths:
1. Load exact benchmark questions from a local JSON/JSONL export.
2. Load exact benchmark questions from Hugging Face (`nyu-visionx/VSI-Bench`).
3. Fall back to repo metadata to tell you whether a ScanNet scene is part of the
   released VSI benchmark split.

For scenes outside the benchmark split, this script does not fabricate "official"
questions. Instead, it explains the mismatch and suggests benchmarked scenes that
already exist in the same video directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNET_META_INFO = REPO_ROOT / "data" / "meta_info" / "scannet_meta_info_val.json"
HF_DATASET_NAME = "nyu-visionx/VSI-Bench"
DEFAULT_EXPORT_DIR = REPO_ROOT / "data" / "vsi_bench" / "exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the VSI-Bench questions for a ScanNet scene when available."
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        help="Path to the scene video, e.g. /path/to/scene0251_00.mp4",
    )
    parser.add_argument(
        "--scene-id",
        help="Scene id such as scene0251_00. If omitted, it is inferred from --video-path.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        help=(
            "Optional local JSON/JSONL file containing VSI-Bench annotations. "
            "If provided, the script reads questions from this file before trying Hugging Face."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to query when using Hugging Face. Default: test",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "local", "hf", "meta"],
        default="auto",
        help=(
            "Question source. 'auto' tries local file, then Hugging Face, then repo metadata. "
            "'meta' only checks whether the scene belongs to the released benchmark split."
        ),
    )
    parser.add_argument(
        "--list-nearby-benchmark-scenes",
        action="store_true",
        help="Also list benchmark ScanNet videos found next to the provided video.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List ScanNet scenes that have questions in the benchmark question file.",
    )
    parser.add_argument(
        "--print-scene-names",
        action="store_true",
        help="Print just the scene names, one per line.",
    )
    parser.add_argument(
        "--omit-ground-truth",
        action="store_true",
        help="Remove ground-truth answers from question outputs.",
    )
    parser.add_argument(
        "--save-json",
        nargs="?",
        const="auto",
        help=(
            "Save the JSON output to disk. Pass no value to auto-name the file, "
            "or pass a file path explicitly."
        ),
    )
    return parser.parse_args()


def infer_scene_id(scene_id: Optional[str], video_path: Optional[Path]) -> str:
    if scene_id:
        return scene_id
    if video_path:
        return video_path.stem
    raise SystemExit("Provide either --scene-id or --video-path.")


def load_scannet_meta() -> Dict[str, Any]:
    if not SCANNET_META_INFO.exists():
        raise FileNotFoundError(f"Missing metadata file: {SCANNET_META_INFO}")
    return json.loads(SCANNET_META_INFO.read_text())


def normalize_doc(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    dataset = doc.get("dataset")
    scene_name = doc.get("scene_name") or doc.get("scene_id")
    question = doc.get("question") or doc.get("Q")
    question_type = doc.get("question_type") or doc.get("task") or doc.get("type")
    ground_truth = doc.get("ground_truth") or doc.get("answer") or doc.get("A")
    options = doc.get("options")

    if not scene_name or not question:
        return None

    normalized = {
        "dataset": dataset,
        "scene_name": scene_name,
        "question": question,
        "question_type": question_type,
        "ground_truth": ground_truth,
    }
    if options is not None:
        normalized["options"] = options
    return normalized


def iter_local_docs(questions_file: Path) -> Iterable[Dict[str, Any]]:
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    if questions_file.suffix == ".jsonl":
        for line in questions_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            doc = normalize_doc(json.loads(line))
            if doc:
                yield doc
        return

    data = json.loads(questions_file.read_text())
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                doc = normalize_doc(item)
                if doc:
                    yield doc
        return

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        doc = normalize_doc(item)
                        if doc:
                            yield doc
            elif isinstance(value, dict):
                merged = dict(value)
                merged.setdefault("scene_name", key)
                doc = normalize_doc(merged)
                if doc:
                    yield doc


def load_questions_from_local(scene_id: str, questions_file: Path) -> List[Dict[str, Any]]:
    return [doc for doc in iter_local_docs(questions_file) if doc["scene_name"] == scene_id]


def list_scannet_scenes_from_local(questions_file: Path) -> Dict[str, Any]:
    scene_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"num_questions": 0, "question_types": set()}
    )
    total_scannet_questions = 0

    for raw_doc in iter_local_docs(questions_file):
        if raw_doc.get("dataset") != "scannet":
            continue
        scene_name = raw_doc["scene_name"]
        scene_stats[scene_name]["num_questions"] += 1
        total_scannet_questions += 1
        question_type = raw_doc.get("question_type")
        if question_type:
            scene_stats[scene_name]["question_types"].add(question_type)

    scenes = []
    for scene_name in sorted(scene_stats):
        stats = scene_stats[scene_name]
        scenes.append(
            {
                "scene_name": scene_name,
                "num_questions": stats["num_questions"],
                "question_types": sorted(stats["question_types"]),
            }
        )

    return {
        "source": f"local:{questions_file}",
        "dataset": "scannet",
        "num_scenes": len(scenes),
        "num_questions": total_scannet_questions,
        "scenes": scenes,
    }


def load_questions_from_hf(scene_id: str, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'datasets' package is required for Hugging Face access. "
            "Install it with: pip install datasets"
        ) from exc

    dataset = load_dataset(HF_DATASET_NAME, split=split)
    matches: List[Dict[str, Any]] = []
    for doc in dataset:
        if doc.get("scene_name") == scene_id:
            normalized = normalize_doc(dict(doc))
            if normalized:
                matches.append(normalized)
    return matches


def list_benchmark_scenes_in_directory(video_path: Path, scannet_meta: Dict[str, Any]) -> List[str]:
    directory = video_path.parent
    available = []
    for file_path in sorted(directory.glob("scene*.mp4")):
        if file_path.stem in scannet_meta:
            available.append(file_path.name)
    return available


def build_meta_only_report(scene_id: str, video_path: Optional[Path], scannet_meta: Dict[str, Any]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "scene_name": scene_id,
        "video_path": str(video_path) if video_path else None,
        "benchmark_dataset": HF_DATASET_NAME,
        "benchmark_split_checked": "ScanNet validation metadata released in this repo",
        "is_scannet_benchmark_scene": scene_id in scannet_meta,
    }

    if scene_id in scannet_meta:
        meta = scannet_meta[scene_id]
        report["message"] = (
            "This scene is present in the released ScanNet VSI metadata. "
            "You can retrieve its exact question set from the benchmark annotations."
        )
        report["meta_summary"] = {
            "dataset": meta.get("dataset"),
            "num_object_categories": len(meta.get("object_counts", {})),
            "room_size": meta.get("room_size"),
        }
    else:
        report["message"] = (
            "This scene is not part of the released ScanNet VSI benchmark metadata in this repo, "
            "so there is no official benchmark question set available locally for it."
        )
        if video_path:
            nearby = list_benchmark_scenes_in_directory(video_path, scannet_meta)
            report["benchmark_scenes_found_next_to_video"] = nearby[:25]
    return report


def build_questions_report(
    scene_id: str,
    video_path: Optional[Path],
    source: str,
    questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    report = {
        "scene_name": scene_id,
        "video_path": str(video_path) if video_path else None,
        "source": source,
        "num_questions": len(questions),
        "questions": questions,
    }
    return report


def maybe_strip_ground_truth(report: Dict[str, Any], omit_ground_truth: bool) -> Dict[str, Any]:
    if not omit_ground_truth or "questions" not in report:
        return report

    stripped_questions = []
    for question in report["questions"]:
        item = dict(question)
        item.pop("ground_truth", None)
        stripped_questions.append(item)

    updated_report = dict(report)
    updated_report["questions"] = stripped_questions
    updated_report["ground_truth_included"] = False
    return updated_report


def dump_output(data: Dict[str, Any], pretty: bool) -> None:
    if pretty:
        print(json.dumps(data, indent=2, ensure_ascii=True))
    else:
        print(json.dumps(data, ensure_ascii=True))


def print_scene_names(data: Dict[str, Any]) -> None:
    for scene in data.get("scenes", []):
        scene_name = scene.get("scene_name")
        if scene_name:
            print(scene_name)


def auto_output_path(report: Dict[str, Any], omit_ground_truth: bool) -> Path:
    DEFAULT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    if "questions" in report and report.get("scene_name"):
        suffix = "questions_no_ground_truth" if omit_ground_truth else "questions"
        return DEFAULT_EXPORT_DIR / f"{report['scene_name']}_{suffix}.json"

    if report.get("dataset") == "scannet" and "scenes" in report:
        suffix = "scene_names_with_counts"
        return DEFAULT_EXPORT_DIR / f"scannet_vsi_bench_{suffix}.json"

    return DEFAULT_EXPORT_DIR / "vsi_bench_output.json"


def save_json_output(data: Dict[str, Any], pretty: bool, save_json: Optional[str], omit_ground_truth: bool) -> Optional[Path]:
    if not save_json:
        return None

    output_path = auto_output_path(data, omit_ground_truth) if save_json == "auto" else Path(save_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")
    else:
        output_path.write_text(json.dumps(data, ensure_ascii=True) + "\n")
    return output_path


def main() -> int:
    args = parse_args()
    if args.list_scenes:
        if not args.questions_file:
            raise SystemExit("--list-scenes requires --questions-file.")
        report = list_scannet_scenes_from_local(args.questions_file)
        dump_output(report, args.pretty)
        if args.print_scene_names:
            print()
            print_scene_names(report)
        saved_path = save_json_output(report, args.pretty, args.save_json, args.omit_ground_truth)
        if saved_path:
            print()
            print(f"Saved JSON to: {saved_path}")
        return 0

    scene_id = infer_scene_id(args.scene_id, args.video_path)
    scannet_meta = load_scannet_meta()

    if args.source in {"auto", "local"} and args.questions_file:
        questions = load_questions_from_local(scene_id, args.questions_file)
        if questions:
            report = build_questions_report(scene_id, args.video_path, f"local:{args.questions_file}", questions)
            report = maybe_strip_ground_truth(report, args.omit_ground_truth)
            dump_output(report, args.pretty)
            saved_path = save_json_output(report, args.pretty, args.save_json, args.omit_ground_truth)
            if saved_path:
                print(f"Saved JSON to: {saved_path}")
            return 0
        if args.source == "local":
            report = {
                "scene_name": scene_id,
                "source": f"local:{args.questions_file}",
                "num_questions": 0,
                "message": "No questions for this scene were found in the provided local file.",
            }
            dump_output(report, args.pretty)
            saved_path = save_json_output(report, args.pretty, args.save_json, args.omit_ground_truth)
            if saved_path:
                print(f"Saved JSON to: {saved_path}")
            return 1

    if args.source in {"auto", "hf"}:
        try:
            questions = load_questions_from_hf(scene_id, args.split)
        except Exception as exc:
            if args.source == "hf":
                dump_output(
                    {
                        "scene_name": scene_id,
                        "source": "huggingface",
                        "num_questions": 0,
                        "error": str(exc),
                    },
                    args.pretty,
                )
                return 1
        else:
            if questions:
                report = build_questions_report(scene_id, args.video_path, f"hf:{HF_DATASET_NAME}:{args.split}", questions)
                report = maybe_strip_ground_truth(report, args.omit_ground_truth)
                dump_output(report, args.pretty)
                saved_path = save_json_output(report, args.pretty, args.save_json, args.omit_ground_truth)
                if saved_path:
                    print(f"Saved JSON to: {saved_path}")
                return 0

    report = build_meta_only_report(scene_id, args.video_path, scannet_meta)
    if args.list_nearby_benchmark_scenes and args.video_path:
        report["benchmark_scenes_found_next_to_video"] = list_benchmark_scenes_in_directory(
            args.video_path, scannet_meta
        )
    dump_output(report, args.pretty)
    saved_path = save_json_output(report, args.pretty, args.save_json, args.omit_ground_truth)
    if saved_path:
        print(f"Saved JSON to: {saved_path}")
    return 0 if report["is_scannet_benchmark_scene"] else 2


if __name__ == "__main__":
    sys.exit(main())
