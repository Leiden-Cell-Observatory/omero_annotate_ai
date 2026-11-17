"""Command line interface: ``omero-annotate-ai train``."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from omero_annotate_ai.processing.trainers import TRAINERS, _deep_merge, train


def _parse_value(raw: str) -> Any:
    """JSON first (so ``1e-4`` is a float; YAML reads it as a string), then YAML."""
    try:
        return json.loads(raw)
    except ValueError:
        return yaml.safe_load(raw)


def parse_set(items: list[str]) -> dict[str, Any]:
    """Turn ``KEY=VALUE`` items into a dict; values are YAML, dotted keys nest under ``config``."""
    params: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        value = _parse_value(raw)
        if "." in key:
            for part in reversed(key.split(".")):
                value = {part: value}
            params = _deep_merge(params, {"config": value})
        else:
            params[key] = value
    return params


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omero-annotate-ai")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "train", help="Train a model on an exported training-data folder"
    )
    p.add_argument(
        "export_dir", type=Path, help="Folder from prepare_training_data_from_table"
    )
    p.add_argument("--trainer", required=True, choices=sorted(TRAINERS))
    p.add_argument("--output", required=True, type=Path, help="Output directory")
    p.add_argument("--params", type=Path, help="YAML file with trainer parameters")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a parameter (repeatable); dotted keys go to the trainer config",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        params: dict[str, Any] = {}
        if args.params:
            params = yaml.safe_load(args.params.read_text()) or {}
        params = _deep_merge(params, parse_set(args.set))
        result = train(args.export_dir, args.trainer, args.output, **params)
        (args.output / "training_result.json").write_text(
            json.dumps(result.to_json(), indent=2)
        )
        print(result.model_path)  # noqa: T201
    except Exception as e:  # noqa: BLE001 - any failure is exit code 1 with its message
        print(f"Error: {e}", file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
