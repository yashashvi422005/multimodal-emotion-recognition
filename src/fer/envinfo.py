"""Environment/version reporting, used in verification reports now and
in MLflow run logging from Phase 3 onward."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def get_git_commit(repo_root: str = ".") -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def get_package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def get_env_report(repo_root: str = ".") -> dict:
    packages = [
        "pandas", "numpy", "av", "soundfile", "librosa", "scikit-learn", "scipy",
        "torch", "torchvision", "torchaudio", "timm", "transformers",
        "mediapipe", "retina-face", "opensmile", "mlflow", "wandb", "gradio",
    ]
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": get_git_commit(repo_root),
        "package_versions": {p: get_package_version(p) for p in packages},
    }
