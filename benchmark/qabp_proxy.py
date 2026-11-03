"""Phase B of the quality-aware policy: does a signal at the cut predict what pruning costs
the task? See :mod:`vitprune.pipeline.benchmark.proxy`.

Run on a finished policy study whose AMA step had the annotations (``run_qabp.sh selecao``):
the blocks up to the cut on every clip (a few minutes of GPU), then the analysis on the CPU.
Exploratory: nothing here tunes a parameter.

    python benchmark/qabp_proxy.py --run benchmark/results/qabp/selecao --weights W --model-source S
"""

from __future__ import annotations

import argparse

from vitprune.core import ClipGeometry, PrunedEncoder, PruningPlan, check_backbone
from vitprune.pipeline.benchmark.proxy import analyse, collect_cut_features
from vitprune.pipeline.benchmark.settings import RunRecord
from vitprune.pipeline.checkpoint import disable_activation_checkpointing, load_videomae


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="results folder of the policy study")
    parser.add_argument("--weights")
    parser.add_argument("--model-source")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--analyse-only", action="store_true",
                        help="the signals at the cut are already collected")
    args = parser.parse_args()

    record = RunRecord.read(args.run)
    if not args.analyse_only:
        geometry = ClipGeometry.videomae_giant()
        model = disable_activation_checkpointing(
            load_videomae(args.weights, args.model_source, geometry, device=args.device))
        print(check_backbone(model, geometry, reference=model.forward_features).describe())
        encoder = PrunedEncoder(model, geometry, PruningPlan(prune_layer=record.prune_layer))
        collect_cut_features(encoder, args.run, args.device, record.settings["precision"],
                             record.settings["batch"])

    report = analyse(args.run)
    for variant, found in report["variants"].items():
        lost = found["actions"]
        print(f"\n{variant}: {found['labelled_clips']} clips inside one action, "
              f"{found.get('flips', 0)} right in dense and wrong here; actions found "
              f"{lost.get('found')} of {lost.get('annotated')}, lost {lost.get('lost')}")
        for signal, values in found["signals"].items():
            print(f"  {signal:>10}: rho with delta_ce {values['rho_delta_ce']:+.3f} "
                  f"(p within videos {values['p_delta_ce_within_videos']:.3f}), "
                  f"with delta_p {values['rho_delta_p']:+.3f}, AUC of flips "
                  f"{values['auc_flip']:.3f}")
        for control in ("g", "motion_redundancy"):
            beyond = found.get(f"probe_beyond_{control}", {})
            print(f"  probe beyond {control}: "
                  f"delta_ce {beyond.get('delta_ce', float('nan')):+.3f}, "
                  f"delta_p {beyond.get('delta_p', float('nan')):+.3f}")
    print(f"\nwritten to {args.run}/proxy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
