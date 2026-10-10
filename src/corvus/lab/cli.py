import argparse


def main():
    p = argparse.ArgumentParser(description="Corvus: supervised baseline, stress testing and inference")
    commands = p.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--manifest", default="data/prepared/ecg-lab/manifest.csv")
    train.add_argument("--output", default="runs/ecg-lab")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--seed", type=int, default=42)
    build = commands.add_parser("build")
    build.add_argument("--manifest", default="data/prepared/ecg-lab/manifest.csv")
    build.add_argument("--training", default="runs/ecg-lab")
    build.add_argument("--astro-checkpoint", default="runs/real_seed42/checkpoint.pt")
    build.add_argument("--output", default="artifacts/lab")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--artifacts", default="artifacts/lab")
    args = vars(p.parse_args())
    command = args.pop("command")
    if command == "train":
        from .train import train_ecg
        train_ecg(**args)
    elif command == "build":
        from .build import build_bundle
        build_bundle(**args)
    else:
        import uvicorn
        from .api import create_app
        artifacts = args.pop("artifacts")
        uvicorn.run(create_app(artifacts), **args, access_log=False)


if __name__ == "__main__":
    main()
