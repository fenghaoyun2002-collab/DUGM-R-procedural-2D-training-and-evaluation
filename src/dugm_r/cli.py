"""Public 2D training, evaluation, and visualization commands."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dugm_r_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dugm_r_cache")

from dugm_r.evaluation_2d.evaluate import evaluate_2d
from dugm_r.nominal.train import train_nominal
from dugm_r.recovery.rollouts import collect_rollouts
from dugm_r.recovery.train import train_recovery
from dugm_r.recovery.train_rvf import train_rvf
from dugm_r.visualization.plots import render_curriculum, render_observation, render_rvf


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dugm-r", description="DUGM-R public procedural 2D pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    nominal = sub.add_parser("train-nominal")
    nominal.add_argument("--config", default="nominal_train_2d.yaml")
    nominal.add_argument("--output", type=Path, required=True)
    nominal.add_argument("--smoke", action="store_true")
    rollouts = sub.add_parser("collect-nominal-rollouts")
    rollouts.add_argument("--config", default="nominal_rollouts_2d.yaml")
    rollouts.add_argument("--nominal", type=Path, required=True)
    rollouts.add_argument("--output", type=Path, required=True)
    rollouts.add_argument("--smoke", action="store_true")
    rvf = sub.add_parser("train-rvf")
    rvf.add_argument("--config", default="rvf_h40.yaml")
    rvf.add_argument("--rollouts-manifest", type=Path, required=True)
    rvf.add_argument("--output", type=Path, required=True)
    rvf.add_argument("--smoke", action="store_true")
    recovery = sub.add_parser("train-recovery")
    recovery.add_argument("--config", default="recovery_train_2d.yaml")
    recovery.add_argument("--nominal", type=Path, required=True)
    recovery.add_argument("--rvf", type=Path, required=True)
    recovery.add_argument("--thresholds", type=Path, required=True)
    recovery.add_argument("--warm-start-recovery", type=Path)
    recovery.add_argument("--output", type=Path, required=True)
    recovery.add_argument("--smoke", action="store_true")
    for name in ("eval-2d", "render-2d"):
        p = sub.add_parser(name)
        p.add_argument("--method", required=True, choices=("dugm_nominal", "dugm_r"))
        p.add_argument("--config", default="eval_2d.yaml")
        p.add_argument("--nominal", type=Path, required=True)
        p.add_argument("--recovery", type=Path)
        p.add_argument("--rvf", type=Path)
        p.add_argument("--thresholds", type=Path)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--smoke", action="store_true")
    curriculum = sub.add_parser("render-curriculum")
    curriculum.add_argument("--config", default="nominal_train_2d.yaml")
    curriculum.add_argument("--output", type=Path, required=True)
    observation = sub.add_parser("render-observation")
    observation.add_argument("--config", default="nominal_train_2d.yaml")
    observation.add_argument("--output", type=Path, required=True)
    observation.add_argument("--level", type=int, default=8)
    observation.add_argument("--seed", type=int, default=2000001)
    rvf_plot = sub.add_parser("render-rvf")
    rvf_plot.add_argument("--config", default="nominal_train_2d.yaml")
    rvf_plot.add_argument("--rvf", type=Path, required=True)
    rvf_plot.add_argument("--output", type=Path, required=True)
    rvf_plot.add_argument("--level", type=int, default=8)
    rvf_plot.add_argument("--seed", type=int, default=3000800)
    rvf_plot.add_argument("--state", type=float, nargs=4,
                          help="Optional normalized [distance, bearing, previous linear, previous angular] state.")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "train-nominal":
        print(train_nominal(args.config, args.output, smoke=args.smoke))
    elif args.command == "collect-nominal-rollouts":
        print(collect_rollouts(args.config, args.nominal, args.output, smoke=args.smoke))
    elif args.command == "train-rvf":
        print(train_rvf(args.config, args.rollouts_manifest, args.output, smoke=args.smoke))
    elif args.command == "train-recovery":
        print(train_recovery(args.config, args.nominal, args.rvf, args.thresholds, args.output,
                             smoke=args.smoke, warm_start_checkpoint=args.warm_start_recovery))
    elif args.command in {"eval-2d", "render-2d"}:
        print(evaluate_2d(args.method, args.nominal, args.output, recovery_checkpoint=args.recovery,
                          rvf_checkpoint=args.rvf, thresholds_path=args.thresholds, config_name=args.config,
                          render=args.command == "render-2d", smoke=args.smoke))
    elif args.command == "render-curriculum":
        print(*render_curriculum(args.output, config_name=args.config), sep="\n")
    elif args.command == "render-observation":
        print(render_observation(args.output, config_name=args.config, level=args.level, seed=args.seed))
    elif args.command == "render-rvf":
        print(render_rvf(args.rvf, args.output, level=args.level, seed=args.seed,
                         state_override=args.state, config_name=args.config))
    elif args.command == "smoke":
        root = args.output
        nominal = train_nominal("nominal_train_2d.yaml", root / "nominal", smoke=True)
        rollout_manifest = collect_rollouts("nominal_rollouts_2d.yaml", nominal, root / "rollouts", smoke=True)
        rvf, thresholds = train_rvf("rvf_h40.yaml", rollout_manifest, root / "rvf", smoke=True)
        recovery = train_recovery("recovery_train_2d.yaml", nominal, rvf, thresholds, root / "recovery", smoke=True)
        evaluate_2d("dugm_nominal", nominal, root, render=True, smoke=True)
        evaluate_2d("dugm_r", nominal, root, recovery_checkpoint=recovery, rvf_checkpoint=rvf,
                    thresholds_path=thresholds, render=True, smoke=True)
        render_observation(root / "visualization" / "observation")
        render_rvf(rvf, root / "visualization" / "rvf")
        print(root.resolve())


if __name__ == "__main__":
    main()
