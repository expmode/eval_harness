from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from .io import iter_jsonl, load_jsonl_index


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def validate_run_artifacts(
    *,
    responses_path: Path | None = None,
    judged_path: Path | None = None,
    generation_manifest_path: Path | None = None,
    judging_manifest_path: Path | None = None,
) -> dict:
    issues: list[str] = []
    summary: dict[str, object] = {
        "ok": True,
        "issues": issues,
    }

    response_rows = None
    judged_rows = None

    if responses_path is not None:
        if not responses_path.exists():
            issues.append(f"Missing responses file: {responses_path}")
        else:
            response_rows = load_jsonl_index(responses_path, "row_id")
            duplicate_check = list(iter_jsonl(responses_path))
            if len(duplicate_check) != len(response_rows):
                issues.append("Duplicate row_id values detected in responses file")
            for row_id, row in response_rows.items():
                if not row.get("prompt"):
                    issues.append(f"Response row {row_id} missing prompt")
                if row.get("status") == "success" and not row.get("response"):
                    issues.append(f"Response row {row_id} marked success but has empty response")
                if row.get("status") != "success" and not row.get("error"):
                    issues.append(f"Response row {row_id} marked failure but missing error")
                prompt_hash = row.get("prompt_hash")
                prompt = row.get("prompt", "")
                if prompt_hash and prompt_hash != _sha1_text(prompt):
                    issues.append(f"Response row {row_id} has mismatched prompt_hash")
            summary["responses"] = {
                "path": str(responses_path),
                "row_count": len(response_rows),
            }

    if generation_manifest_path is not None:
        if not generation_manifest_path.exists():
            issues.append(f"Missing generation manifest: {generation_manifest_path}")
        else:
            manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
            if responses_path is not None and response_rows is not None:
                if manifest.get("row_count_written") is not None and manifest.get("row_count_written") != len(response_rows):
                    issues.append("Generation manifest row_count_written does not match responses row count")
                if manifest.get("row_count_total") is not None and manifest.get("row_count_total") < len(response_rows):
                    issues.append("Generation manifest row_count_total is smaller than written responses")
            summary["generation_manifest"] = manifest

    if judged_path is not None:
        if not judged_path.exists():
            issues.append(f"Missing judged file: {judged_path}")
        else:
            judged_rows = load_jsonl_index(judged_path, "row_id")
            duplicate_check = list(iter_jsonl(judged_path))
            if len(duplicate_check) != len(judged_rows):
                issues.append("Duplicate row_id values detected in judged file")
            for row_id, row in judged_rows.items():
                parsing_status = row.get("parsing_status")
                if parsing_status == "success" and row.get("judge_label") not in {"refused", "complied"}:
                    issues.append(f"Judged row {row_id} has invalid success label")
                if parsing_status != "success" and not row.get("raw_judge_output"):
                    issues.append(f"Judged row {row_id} missing raw_judge_output for failure case")
            summary["judged"] = {
                "path": str(judged_path),
                "row_count": len(judged_rows),
            }

    if judging_manifest_path is not None:
        if not judging_manifest_path.exists():
            issues.append(f"Missing judging manifest: {judging_manifest_path}")
        else:
            manifest = json.loads(judging_manifest_path.read_text(encoding="utf-8"))
            if judged_path is not None and judged_rows is not None:
                pending = manifest.get("response_count_pending")
                completed_existing = manifest.get("response_count_completed_existing")
                if pending is not None and completed_existing is not None:
                    expected_min = pending + completed_existing
                    if expected_min < len(judged_rows):
                        issues.append("Judging manifest counts are inconsistent with judged row count")
            summary["judging_manifest"] = manifest

    if response_rows is not None and judged_rows is not None:
        missing_in_judged = sorted(set(response_rows) - set(judged_rows))
        extra_in_judged = sorted(set(judged_rows) - set(response_rows))
        if missing_in_judged:
            issues.append(f"Judged file missing row_ids present in responses: {missing_in_judged[:5]}")
        if extra_in_judged:
            issues.append(f"Judged file has row_ids not present in responses: {extra_in_judged[:5]}")

    summary["ok"] = not issues
    return summary


def validate_run_and_exit_code(**kwargs) -> int:
    payload = validate_run_artifacts(**kwargs)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(validate_run_and_exit_code())