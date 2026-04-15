#!/usr/bin/env python3
"""Upload experiment output to HuggingFace.

Usage:
    python upload.py --name lora_baseline_4A100_main
    python upload.py --name abl_seg_only_2A100 --repo JasonXF/SynthUrbanSAT
"""
import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def load_env():
    """Load .env file from script directory."""
    env_path = SCRIPT_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def main():
    parser = argparse.ArgumentParser(description="Upload experiment output to HuggingFace")
    parser.add_argument("--name", required=True,
                        help="Experiment name (matches output/<name>/ folder)")
    parser.add_argument("--repo", default="JasonXF/SynthUrbanSAT-Output",
                        help="HuggingFace repo ID (default: JasonXF/SynthUrbanSAT-Output)")
    parser.add_argument("--path-in-repo", default=None,
                        help="Remote path prefix in the repo (default: output/<name>)")
    args = parser.parse_args()

    load_env()

    token = os.environ.get("HF_TOKEN_WRITE", "")
    if not token:
        print("ERROR: HF_TOKEN_WRITE not found. Set it in .env or environment.", file=sys.stderr)
        sys.exit(1)

    local_dir = SCRIPT_DIR / "output" / args.name
    if not local_dir.exists():
        print(f"ERROR: output/{args.name}/ does not exist.", file=sys.stderr)
        print(f"Available experiments:", file=sys.stderr)
        output_dir = SCRIPT_DIR / "output"
        if output_dir.exists():
            for d in sorted(output_dir.iterdir()):
                if d.is_dir() and not d.name.startswith("."):
                    print(f"  - {d.name}", file=sys.stderr)
        sys.exit(1)

    path_in_repo = args.path_in_repo or f"output/{args.name}"

    # Disable XET storage backend
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Ensure repo exists
    try:
        api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"WARN: create_repo: {e}", file=sys.stderr)

    file_count = sum(1 for _ in local_dir.rglob("*") if _.is_file())
    print(f"Uploading {file_count} files from output/{args.name}/ → {args.repo}/{path_in_repo}")
    print(f"Token: ...{token[-4:]}")

    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_type="model",
    )

    print(f"\n✓ Upload complete: https://huggingface.co/{args.repo}/tree/main/{path_in_repo}")


if __name__ == "__main__":
    main()
