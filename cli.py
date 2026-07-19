from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import JudgeConfig, ModelConfig, RunConfig
from .dataset import summarize_dataset
from .generate import plan_generation, run_generation
from .judge import plan_judging, run_judging
from .metrics import score_judged_file
from .run_specs import load_run_config
from .validate import validate_run_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EU-Guard Track 1 evaluation harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(subparser: argparse.ArgumentParser, *, include_judge: bool = False) -> None:
        subparser.add_argument("--dataset-path", default="EU_alert_working_copy/test/test.jsonl")
        subparser.add_argument("--output-dir", default="eval")
        subparser.add_argument("--config")
        subparser.add_argument("--alias-file")
        subparser.add_argument("--limit", type=int)
        subparser.add_argument("--random-sample", type=int)
        subparser.add_argument("--sampling-seed", type=int, default=0)
        subparser.add_argument("--concurrency", type=int, default=8)
        subparser.add_argument("--verified-only", action="store_true")
        subparser.add_argument("--overwrite", action="store_true")
        subparser.add_argument("--run-name")
        subparser.add_argument("--language-include", action="append", default=[])
        subparser.add_argument("--language-exclude", action="append", default=[])
        subparser.add_argument("--category-include", action="append", default=[])
        subparser.add_argument("--category-exclude", action="append", default=[])
        subparser.add_argument("--model-name")
        subparser.add_argument("--backend")
        subparser.add_argument("--provider")
        subparser.add_argument("--mode")
        subparser.add_argument("--model")
        subparser.add_argument("--api-key")
        subparser.add_argument("--base-url")
        subparser.add_argument("--system-prompt")
        subparser.add_argument("--temperature", type=float, default=0.0)
        subparser.add_argument("--top-p", type=float, default=1.0)
        subparser.add_argument("--max-tokens", type=int, default=512)
        subparser.add_argument("--timeout-seconds", type=float, default=60.0)
        subparser.add_argument("--max-retries", type=int, default=3)
        subparser.add_argument("--retry-base-delay-seconds", type=float, default=1.0)
        subparser.add_argument("--retry-max-delay-seconds", type=float, default=8.0)
        subparser.add_argument("--retry-jitter-seconds", type=float, default=0.25)
        if include_judge:
            subparser.add_argument("--judge-name")
            subparser.add_argument("--judge-backend")
            subparser.add_argument("--judge-provider")
            subparser.add_argument("--judge-mode")
            subparser.add_argument("--judge-model")
            subparser.add_argument("--judge-api-key")
            subparser.add_argument("--judge-base-url")
            subparser.add_argument("--judge-system-prompt")
            subparser.add_argument("--judge-template-path")
            subparser.add_argument("--judge-temperature", type=float)
            subparser.add_argument("--judge-top-p", type=float)
            subparser.add_argument("--judge-timeout-seconds", type=float)
            subparser.add_argument("--judge-max-retries", type=int)
            subparser.add_argument("--judge-retry-base-delay-seconds", type=float)
            subparser.add_argument("--judge-retry-max-delay-seconds", type=float)
            subparser.add_argument("--judge-retry-jitter-seconds", type=float)

    add_common_args(subparsers.add_parser("generate"))
    add_common_args(subparsers.add_parser("judge"), include_judge=True)
    add_common_args(subparsers.add_parser("run-all"), include_judge=True)
    add_common_args(subparsers.add_parser("plan"), include_judge=True)

    dataset_parser = subparsers.add_parser("dataset-summary")
    dataset_parser.add_argument("--dataset-path", default="EU_alert_working_copy/test/test.jsonl")
    dataset_parser.add_argument("--limit", type=int)
    dataset_parser.add_argument("--random-sample", type=int)
    dataset_parser.add_argument("--sampling-seed", type=int, default=0)
    dataset_parser.add_argument("--verified-only", action="store_true")
    dataset_parser.add_argument("--language-include", action="append", default=[])
    dataset_parser.add_argument("--language-exclude", action="append", default=[])
    dataset_parser.add_argument("--category-include", action="append", default=[])
    dataset_parser.add_argument("--category-exclude", action="append", default=[])

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--judged-path", required=True)
    score_parser.add_argument("--output-dir", default="eval")
    score_parser.add_argument("--run-stem", required=True)

    validate_parser = subparsers.add_parser("validate-run")
    validate_parser.add_argument("--responses-path")
    validate_parser.add_argument("--judged-path")
    validate_parser.add_argument("--generation-manifest-path")
    validate_parser.add_argument("--judging-manifest-path")

    for parser_with_slicing in [
        subparsers.choices["generate"],
        subparsers.choices["judge"],
        subparsers.choices["run-all"],
        dataset_parser,
    ]:
        parser_with_slicing.epilog = (
            "Slicing note: --limit and --random-sample are mutually exclusive. "
            "Include/exclude filters must not overlap."
        )

    return parser


def _model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        name=args.model_name,
        backend=args.backend,
        provider=args.provider,
        mode=args.mode,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_base_delay_seconds=args.retry_base_delay_seconds,
        retry_max_delay_seconds=args.retry_max_delay_seconds,
        retry_jitter_seconds=args.retry_jitter_seconds,
    )


def _judge_config_from_args(args: argparse.Namespace) -> JudgeConfig:
    return JudgeConfig(
        name=args.judge_name,
        backend=args.judge_backend,
        provider=args.judge_provider,
        mode=args.judge_mode,
        model=args.judge_model,
        api_key=args.judge_api_key,
        base_url=args.judge_base_url,
        system_prompt=args.judge_system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.judge_temperature if args.judge_temperature is not None else args.temperature,
        top_p=args.judge_top_p if args.judge_top_p is not None else args.top_p,
        timeout_seconds=args.judge_timeout_seconds or args.timeout_seconds,
        max_retries=args.judge_max_retries if args.judge_max_retries is not None else args.max_retries,
        retry_base_delay_seconds=(
            args.judge_retry_base_delay_seconds
            if args.judge_retry_base_delay_seconds is not None
            else args.retry_base_delay_seconds
        ),
        retry_max_delay_seconds=(
            args.judge_retry_max_delay_seconds
            if args.judge_retry_max_delay_seconds is not None
            else args.retry_max_delay_seconds
        ),
        retry_jitter_seconds=(
            args.judge_retry_jitter_seconds
            if args.judge_retry_jitter_seconds is not None
            else args.retry_jitter_seconds
        ),
        prompt_template_path=Path(args.judge_template_path) if args.judge_template_path else None,
    )


def _run_config_from_args(args: argparse.Namespace, include_judge: bool = False) -> RunConfig:
    if getattr(args, "config", None):
        return load_run_config(
            Path(args.config),
            alias_path=Path(args.alias_file) if getattr(args, "alias_file", None) else None,
            cli_overrides={
                "dataset_path": args.dataset_path,
                "output_dir": args.output_dir,
                "limit": args.limit,
                "random_sample": args.random_sample,
                "sampling_seed": args.sampling_seed,
                "concurrency": args.concurrency,
                "verified_only": args.verified_only,
                "overwrite": args.overwrite,
            },
        )
    if not args.model_name or not args.backend or not args.model:
        raise ValueError("Without --config, --model-name, --backend, and --model are required")
    if include_judge and (not args.judge_name or not args.judge_backend or not args.judge_model):
        raise ValueError("Without --config, --judge-name, --judge-backend, and --judge-model are required")
    return RunConfig(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        concurrency=args.concurrency,
        overwrite=args.overwrite,
        verified_only=args.verified_only,
        language_include=tuple(args.language_include),
        language_exclude=tuple(args.language_exclude),
        category_include=tuple(args.category_include),
        category_exclude=tuple(args.category_exclude),
        random_sample=args.random_sample,
        sampling_seed=args.sampling_seed,
        model=_model_config_from_args(args),
        judge=_judge_config_from_args(args) if include_judge else None,
        run_name=args.run_name,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        run_generation(_run_config_from_args(args))
        return

    if args.command == "judge":
        config = _run_config_from_args(args, include_judge=True)
        run_judging(config)
        return

    if args.command == "run-all":
        config = _run_config_from_args(args, include_judge=True)
        responses_path = run_generation(config)
        judged_path = run_judging(config, responses_path=responses_path)
        score_judged_file(judged_path, config.output_dir, f"{config.model.name}__{config.judge.name}")
        return

    if args.command == "plan":
        config = _run_config_from_args(args, include_judge=True)
        payload = {
            "generation": plan_generation(config),
            "judging": plan_judging(config),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.command == "score":
        score_judged_file(Path(args.judged_path), Path(args.output_dir), args.run_stem)
        return

    if args.command == "dataset-summary":
        config = RunConfig(
            dataset_path=Path(args.dataset_path),
            limit=args.limit,
            verified_only=args.verified_only,
            language_include=tuple(args.language_include),
            language_exclude=tuple(args.language_exclude),
            category_include=tuple(args.category_include),
            category_exclude=tuple(args.category_exclude),
            random_sample=args.random_sample,
            sampling_seed=args.sampling_seed,
        )
        print(json.dumps(summarize_dataset(config), indent=2, ensure_ascii=False))
        return

    if args.command == "validate-run":
        payload = validate_run_artifacts(
            responses_path=Path(args.responses_path) if args.responses_path else None,
            judged_path=Path(args.judged_path) if args.judged_path else None,
            generation_manifest_path=(
                Path(args.generation_manifest_path) if args.generation_manifest_path else None
            ),
            judging_manifest_path=(
                Path(args.judging_manifest_path) if args.judging_manifest_path else None
            ),
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()