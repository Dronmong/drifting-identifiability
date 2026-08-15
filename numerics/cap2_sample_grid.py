"""Fixed uncurated sample grids for visual comparison.

Draws from a fixed seed and writes the *first* rows x columns images with no
selection, sorting or retries -- ``save_fixed_grid`` enforces that, and the
one-call forward count is asserted by ``generate`` rather than assumed. A grid
assembled by picking good samples would be worthless for judging a generator,
so nothing here is allowed to choose.

Both arms of a comparison must be given the same seed and the same count, which
is why the seed is an explicit argument with one default rather than something
each invocation picks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from numerics.encoder_independent_drifting.stage_cap.config import (  # noqa: E402
    CAPModelConfig,
)
from numerics.encoder_independent_drifting.stage_cap.evaluation import (  # noqa: E402
    generate,
)
from numerics.encoder_independent_drifting.stage_cap.model import (  # noqa: E402
    CAPPixelTransformer,
)
from numerics.encoder_independent_drifting.stage_cap2.preview import (  # noqa: E402
    save_fixed_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--columns", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_804)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_config = CAPModelConfig(**payload["profile"]["model"])
    model = CAPPixelTransformer(model_config, seed=1).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    needed = args.rows * args.columns
    images, calls = generate(model, needed, model_config, args.seed, device,
                             batch=min(500, needed))
    digest = save_fixed_grid(images, args.out, rows=args.rows, columns=args.columns)

    record = {
        "label": args.label or args.checkpoint.stem,
        "checkpoint": str(args.checkpoint),
        "step": payload.get("step"),
        "kind": payload.get("kind"),
        "grid": str(args.out),
        "grid_sha256": digest,
        "rows": args.rows,
        "columns": args.columns,
        "samples": needed,
        "seed": args.seed,
        "model_calls": calls,
        "one_call_per_batch": calls == (needed + min(500, needed) - 1)
        // min(500, needed),
        "selection": "none; the first rows*columns draws in generation order",
        "embedding_scale": model_config.scalar_embedding_scale,
    }
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
