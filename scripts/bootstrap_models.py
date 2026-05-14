"""Post-uv-sync model bootstrap.

Handles the three runtime artifacts that uv cannot lock:
  - coreferee's English model (if `nlp-extra` group is installed)
  - NLTK corpora (punkt, wordnet, averaged_perceptron_tagger, brown)
  - quantulum3 classifier retrain (`quantulum3-training -s`)

Run after `uv sync`:
    uv run python scripts/bootstrap_models.py

The spaCy `en_core_web_lg` model is NOT downloaded here — it's a URL dep
in pyproject.toml, so `uv sync` installs it directly.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import ssl
import subprocess
import sys
from typing import Callable

logger = logging.getLogger("dackar.bootstrap")

# Step name -> function. Ordered for predictable output; --only picks one.
STEPS: dict[str, Callable[[], None]] = {}


def _has_module(name: str) -> bool:
    """True if module is importable in the current environment."""
    return importlib.util.find_spec(name) is not None


def _apply_insecure_ssl() -> None:
    """Disable SSL cert verification globally for downloads behind MITM proxies.

    Same trick as the old nltkDownloader.py. Off by default; opt in with
    --insecure-ssl.
    """
    try:
        _create_unverified = ssl._create_unverified_context  # type: ignore[attr-defined]
    except AttributeError:
        return
    ssl._create_default_https_context = _create_unverified


def install_coreferee() -> None:
    """`python -m coreferee install en` — only if coreferee is importable."""
    if not _has_module("coreferee"):
        logger.info("coreferee not installed (nlp-extra group missing); skipping")
        return
    logger.info("Installing coreferee English model")
    subprocess.check_call([sys.executable, "-m", "coreferee", "install", "en"])


STEPS["coreferee"] = install_coreferee


def install_nltk_corpora() -> None:
    """Download NLTK corpora used by dackar.similarity / dackar.text_processing."""
    import nltk

    corpora = ["punkt", "wordnet", "averaged_perceptron_tagger", "brown"]
    for name in corpora:
        logger.info("nltk.download(%s)", name)
        nltk.download(name, quiet=True)


STEPS["nltk"] = install_nltk_corpora


def retrain_quantulum() -> None:
    """`quantulum3-training -s` — retrains classifier against installed sklearn."""
    logger.info("Retraining quantulum3 classifier")
    subprocess.check_call(["quantulum3-training", "-s"])


STEPS["quantulum"] = retrain_quantulum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap runtime models/corpora for DACKAR.",
    )
    parser.add_argument(
        "--only",
        choices=sorted(STEPS),
        help="Run a single step instead of all (useful for partial CI envs).",
    )
    parser.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable SSL cert verification for downloads (MITM proxy workaround).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    args = build_parser().parse_args(argv)

    if args.insecure_ssl:
        logger.warning("SSL verification disabled for this run")
        _apply_insecure_ssl()

    selected = [args.only] if args.only else list(STEPS)
    for name in selected:
        logger.info("=== %s ===", name)
        STEPS[name]()

    logger.info("Bootstrap complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
