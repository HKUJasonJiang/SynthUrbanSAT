#!/usr/bin/env python3
"""Upload experiment output to HuggingFace.

Usage (CLI):
    python upload.py --name lora_baseline_4A100_main
    python upload.py --name abl_seg_only_2A100 --repo JasonXF/SynthUrbanSAT-Output

Programmatic (from train_script.py):
    from upload import upload_to_hf
    upload_to_hf("output/my_run", repo="JasonXF/SynthUrbanSAT-Output",
                 path_in_repo="output/my_run",
                 ignore_patterns=["checkpoint_*/**"])
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


def upload_to_hf(local_dir, repo="JasonXF/SynthUrbanSAT-Output",
                 path_in_repo=None, token=None, ignore_patterns=None):
    """Upload a local directory to a HuggingFace repo.

    Args:
        local_dir: Path to the local directory to upload.
        repo: HuggingFace repo ID (e.g. "JasonXF/SynthUrbanSAT-Output").
        path_in_repo: Remote path prefix (default: basename of local_dir).
        token: HF write token. If None, reads from HF_TOKEN_WRITE env / .env.
        ignore_patterns: List of glob patterns to exclude (e.g. ["checkpoint_*/**"]).

    Returns:
        URL string on success.

    Raises:
        ValueError: if no token is found.
    """
    if token is None:
        load_env()
        token = os.environ.get("HF_TOKEN_WRITE", "")
    if not token:
        raise ValueError("HF_TOKEN_WRITE not found. Set it in .env or environment.")

    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Directory not found: {local_dir}")

    if path_in_repo is None:
        path_in_repo = local_dir.name

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    try:
        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"WARN: create_repo: {e}", file=sys.stderr)

    kwargs = dict(
        repo_id=repo,
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_type="model",
    )
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns

    api.upload_folder(**kwargs)

    url = f"https://huggingface.co/{repo}/tree/main/{path_in_repo}"
    return url


def main():
    parser = argparse.ArgumentParser(description="Upload experiment output to HuggingFace")
    parser.add_argument("--name", required=True,
                        help="Experiment name (matches output/<name>/ folder)")
    parser.add_argument("--repo", default="JasonXF/SynthUrbanSAT-Output",
                        help="HuggingFace repo ID (default: JasonXF/SynthUrbanSAT-Output)")
    parser.add_argument("--path-in-repo", default=None,
                        help="Remote path prefix in the repo (default: output/<name>)")
    args = parser.parse_args()

    local_dir = SCRIPT_DIR / "output" / args.name
    if not local_dir.exists():
        print(f"ERROR: output/{args.name}/ does not exist.", file=sys.stderr)
        print("Available experiments:", file=sys.stderr)
        output_dir = SCRIPT_DIR / "output"
        if output_dir.exists():
            for d in sorted(output_dir.iterdir()):
                if d.is_dir() and not d.name.startswith("."):
                    print(f"  - {d.name}", file=sys.stderr)
        sys.exit(1)

    path_in_repo = args.path_in_repo or f"output/{args.name}"
    file_count = sum(1 for _ in local_dir.rglob("*") if _.is_file())
    print(f"Uploading {file_count} files from output/{args.name}/ → {args.repo}/{path_in_repo}")

    url = upload_to_hf(local_dir, repo=args.repo, path_in_repo=path_in_repo)
    print(f"\n✓ Upload complete: {url}")


if __name__ == "__main__":
    main()
