#!/usr/bin/env python3
"""Visualize the MedQA-USMLE LangGraph architecture.

Usage:
    python scripts/visualize_graph.py              # V3 full (default)
    python scripts/visualize_graph.py v2           # No memory
    python scripts/visualize_graph.py v4           # No verifier
    python scripts/visualize_graph.py all          # All 3 variants
    python scripts/visualize_graph.py v3 --png     # Also export PNG (needs grandalf)
    python scripts/visualize_graph.py v3 --save-mermaid  # Save Mermaid .mmd file
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from medqa_usmle.configs import load_config
from medqa_usmle.agents.graph import build_graph as build_core_graph
from medqa_usmle.variants.v3_full_system import build_variant_graph


# ── Variant definitions ──────────────────────────────────────────────────

VARIANTS: dict[str, dict] = {
    "v2": {"with_memory": False, "with_verifier": True, "label": "V2 — No Memory"},
    "v3": {"with_memory": True,  "with_verifier": True, "label": "V3 — Full System"},
    "v4": {"with_memory": True,  "with_verifier": False, "label": "V4 — No Verifier"},
}


def show_variant(label: str, graph, args: argparse.Namespace, variant_key: str):
    """Display graph visualization for one variant."""
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"{sep}\n")

    g = graph.get_graph()

    # ── ASCII ────────────────────────────────────────────────────────
    print("─" * 40)
    print("  ASCII")
    print("─" * 40)
    try:
        g.print_ascii()
    except ImportError as e:
        print(f"  [SKIP] {e} — install grandalf for ASCII layout")
        print("  $ pip install grandalf")

    # ── Mermaid ──────────────────────────────────────────────────────
    mermaid_code = g.draw_mermaid()
    print(f"\n{'─' * 40}")
    print("  Mermaid (copy to https://mermaid.live)")
    print(f"{'─' * 40}\n")
    print(mermaid_code)

    if args.save_mermaid:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        mermaid_path = out_dir / f"graph_{variant_key}.mmd"
        with open(mermaid_path, "w") as f:
            f.write(mermaid_code)
        print(f"\n  → Saved Mermaid: {mermaid_path}")

    # ── PNG (needs grandalf) ─────────────────────────────────────────
    if args.png:
        try:
            png_bytes = g.draw_mermaid_png()
            out_dir = args.output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            png_path = out_dir / f"graph_{variant_key}.png"
            with open(png_path, "wb") as f:
                f.write(png_bytes)
            print(f"\n  → Saved PNG: {png_path}")
        except ImportError as e:
            print(f"\n  [SKIP] PNG export needs grandalf: pip install grandalf")
        except Exception as e:
            print(f"\n  [WARN] PNG export failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Visualize MedQA-USMLE LangGraph")
    parser.add_argument(
        "variant", nargs="?", default="v3",
        choices=["v2", "v3", "v4", "all"],
        help="Variant to visualize (default: v3)",
    )
    parser.add_argument(
        "--png", action="store_true",
        help="Export PNG via Mermaid (requires grandalf)",
    )
    parser.add_argument(
        "--save-mermaid", action="store_true",
        help="Save Mermaid .mmd file to output directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(REPO_ROOT / "scripts" / "output"),
        help="Output directory for saved files (default: scripts/output/)",
    )
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)

    config = load_config()

    def build_one(key: str) -> object:
        """Return a compiled graph for the given variant key."""
        params = VARIANTS[key]
        return build_variant_graph(
            config,
            with_memory=params["with_memory"],
            with_verifier=params["with_verifier"],
        )

    if args.variant == "all":
        for key in ("v2", "v3", "v4"):
            graph = build_one(key)
            show_variant(VARIANTS[key]["label"], graph, args, key)
    else:
        graph = build_one(args.variant)
        show_variant(VARIANTS[args.variant]["label"], graph, args, args.variant)


if __name__ == "__main__":
    main()
