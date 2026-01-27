import argparse
from pathlib import Path
from human3d.pipeline import Human3DPipeline
from human3d.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--input", required=True, type=str)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pipe = Human3DPipeline(cfg)
    out = pipe.run(args.input)
    print(f"\n[OK] Outputs saved to: {out}\n")

if __name__ == "__main__":
    main()
