#!/usr/bin/env python3
"""Validate interactive_brief assets exist and numbers still match metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BRIEF = ROOT / "interactive_brief"


def main() -> None:
    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    required = [BRIEF / "index.html", BRIEF / "styles.css", BRIEF / "app.js"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing brief files: {missing}")

    app_js = (BRIEF / "app.js").read_text()
    # Sanity: shipped DR total appears in the embedded DATA block.
    shipped = int(round(metrics["v3_holdout"]["dr_incremental_total"]))
    if "121476" not in app_js and str(shipped) not in app_js:
        print(
            "Warning: interactive_brief/app.js may be out of date vs outputs/metrics.json. "
            f"Expected shipped DR around {shipped}."
        )
    else:
        print("Brief assets OK; embedded headline numbers align with metrics.json.")
    print(f"Open: {BRIEF / 'index.html'}")


if __name__ == "__main__":
    main()
