import argparse
import json

from .experiment import evaluate, train
from .synthetic import generate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Corvus: source-only astronomical CNN, frozen ECG test")
    commands = parser.add_subparsers(dest="command", required=True)
    synth = commands.add_parser("synthetic", help="Generate explicitly synthetic smoke fixtures")
    synth.add_argument("--output", required=True)
    synth.add_argument("--seed", type=int, default=42)
    fit = commands.add_parser("train", help="Train and calibrate using an astro-only manifest")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--run-dir", required=True)
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--length", type=int, default=1024)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--learning-rate", type=float, default=0.001)
    fit.add_argument("--source-fpr", type=float, default=0.05)
    fit.add_argument("--shuffle-labels", action="store_true", help="Source training label-permutation negative control")
    test = commands.add_parser("evaluate", help="Evaluate a frozen source checkpoint on ECG only")
    test.add_argument("--checkpoint", required=True)
    test.add_argument("--manifest", required=True)
    test.add_argument("--output", required=True)
    test.add_argument("--bootstrap", type=int, default=500)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "synthetic":
            print(generate(**args))
        elif command == "train":
            result = train(**args)
            print(json.dumps({"best_epoch": result["best_epoch"], "source_validation": result["source_validation"]}, indent=2))
        else:
            args["checkpoint_path"] = args.pop("checkpoint")
            result = evaluate(**args)
            print(json.dumps({k: {m: v[m] for m in ["auroc", "average_precision", "sensitivity", "specificity"]}
                              for k, v in result["models"].items()}, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
