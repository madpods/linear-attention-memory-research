"""Stage 3: kernel feature-map ablation (plan section 17).

    python scripts/stage3_sweep.py --preset ablation --count
    python scripts/stage3_sweep.py --preset ablation --only-index 0 --out parts/x.csv

Imports ``FIXED`` from ``stage2_sweep`` rather than restating it, so the frozen
hyperparameters cannot drift between stages (plan principle 3), and writes the
same CSV schema so ``analyze_stage2.py`` reads Stage 3 output unchanged -- the
cross-cutting metrics harness is built once and invoked identically.

Two arm types, and the second is the one that decides anything
--------------------------------------------------------------
``phi`` arms
    ``d_model=64`` (``d_k=16``) with a feature map. Parameter-free, so every arm
    has an identical parameter count; what grows is the state (``d_phi x d_v``)
    and the cost of touching it.

``wide`` arms -- THE CONTROL
    ``identity`` with ``d_model`` widened until ``d_k == d_phi``, reaching the same
    key-space dimension by simply making the projections bigger. Section 17 claims
    phi buys capacity "without inflating the base vector width", so the question is
    not whether phi beats the d_k=16 baseline -- it will, its key space is wider --
    but whether it **matches the wide arm at equal d_phi while using far fewer
    parameters**. If plain d_k=64 beats dpfp2, phi's only advantage is parameter
    count, which is not the capacity claim.

    Note the wide arm is *advantaged*, not exactly matched: widening ``d_model``
    grows ``d_v`` along with ``d_k`` (both are ``d_model/num_heads``), so its
    state is ``d_k' x d_k'`` against phi's ``d_phi x 16``. It gets more state and
    more parameters. That makes it a conservative control -- phi matching it is a
    strong result; phi losing to it is only weak evidence against phi.

What actually bounds capacity (a correction to section 17 as written)
--------------------------------------------------------------------
Section 17 says the ceiling is ``min(d_phi, d_v)``. Taken literally that would
make this whole stage pointless here: ``d_v = 16`` is fixed, so ``min(64, 16)``
is still 16 and expanding ``d_phi`` could change nothing.

But Stage 2 **measured** capacity 44.0 at ``d_k = d_v = 16``, and 44 > 16. So
``min(d_phi, d_v)`` is not what bounds the association count -- it bounds
``rank(S)``, which is a different quantity. Storing N associations means solving
``k_i^T S = v_i^T`` for N equations, which stays solvable while the N keys remain
linearly independent, and that needs ``d_phi >= N``. The binding constraint is
the **key-space dimension**, and 44 at ``d_k=16`` is ~2.75x, consistent with the
``N ~ 2*d_k`` counting in the addendum 03 notes.

Extrapolating that 2.75x, and remembering kv <= 124 is the hard measurement
ceiling at ``seq_len=256``:

    identity 16 -> ~44 (measurable)   dpfp1 32 -> ~88 (measurable)
    dpfp2 64 -> ~176   dpfp3 96 -> ~264   dpfp4 128 -> ~352   (all off-scale)

So **dpfp2 and above are expected to saturate the grid**. If they do, that is a
positive result that the sweep cannot quantify, and ``seq_len`` has to rise
before the larger arms mean anything. Read dpfp1 as the informative arm and
treat "dpfp2+ pins the top of the range" as the signal to extend the range.

Predictions, registered before running (from the interference measurement in
``tests/test_feature_maps.py``): mean off-diagonal |cos| between 512 random
L2-normalized keys at d_k=16 is 0.203, and the maps move it to

    elu 0.954 (4.7x WORSE)   relu 0.328 (1.6x worse)
    dpfp1 0.125   dpfp2 0.121   dpfp3 0.118   dpfp4 0.116

so: (1) ``elu`` and ``relu`` should HURT the delta rule, because an all-positive
code is mutually aligned and interference is what a rank-limited state fights;
(2) DPFP's benefit **saturates almost immediately** -- dpfp1 already captures
1.62x of the 1.75x available at dpfp4 -- so capacity should be roughly flat in
``nu`` while cost grows linearly with ``d_phi``. If that holds, the answer to
"smallest d_phi that closes the gap" is 2*d_k and the larger arms are waste.

Run on ``delta``, not ``gated_delta``. Stage 2 established the two are
indistinguishable over 420 runs, so the effect transfers, and the fixed
comparison baseline (Stage 2's gated_delta curve) is numerically the same.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from lamr.train import TrainConfig, train
from stage2_sweep import FIXED  # noqa: E402  (sibling script, same directory)

#: Feature maps to ablate. identity is the control at d_k=16.
PHI_ARMS = ("identity", "elu", "relu", "dpfp1", "dpfp2", "dpfp3", "dpfp4")

#: d_phi -> d_model that reaches the same d_k without a feature map, at
#: num_heads=4. These are the matched-state controls.
MATCHED_STATE_D_MODEL = {32: 128, 64: 256, 96: 384, 128: 512}

PRESETS = {
    # r=0 only: isolates the capacity question before spending anything on the
    # redundancy axis. Whichever d_phi wins here gets the full r sweep after.
    "ablation": dict(
        kv_pairs=(4, 8, 16, 32, 48, 64, 96),
        redundancies=(0.0,),
        steps=8000,
    ),
    # The winner, crossed with redundancy. Set --feature-maps to the arm that won.
    "redundancy": dict(
        kv_pairs=(4, 8, 16, 32, 48, 64, 96),
        redundancies=(0.0, 0.25, 0.5, 0.75, 0.9),
        steps=8000,
    ),
    "quick": dict(kv_pairs=(4, 16), redundancies=(0.0,), steps=200),
}


def build_grid(preset: dict, phi_arms, include_wide: bool) -> list[tuple]:
    """``(feature_map, d_model, r, kv)`` tuples. Wide arms always use identity."""
    arms: list[tuple[str, int]] = [(fm, FIXED["d_model"]) for fm in phi_arms]
    if include_wide:
        seen = set()
        for fm in phi_arms:
            if not fm.startswith("dpfp"):
                continue
            d_phi = 2 * (FIXED["d_model"] // FIXED["num_heads"]) * int(fm[len("dpfp"):])
            d_model = MATCHED_STATE_D_MODEL.get(d_phi)
            if d_model and d_model not in seen:
                seen.add(d_model)
                arms.append(("identity", d_model))
    return [
        (fm, dm, r, kv)
        for (fm, dm), r, kv in itertools.product(
            arms, preset["redundancies"], preset["kv_pairs"]
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="ablation")
    parser.add_argument("--out", default="results/parts_stage3/stage3.csv")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="delta", help="delta|gated_delta|linear")
    parser.add_argument("--backend", default="chunked")
    parser.add_argument("--feature-maps", nargs="+", default=list(PHI_ARMS))
    parser.add_argument(
        "--no-wide",
        action="store_true",
        help="skip the matched-state control arms. They are the point; only skip "
        "for a smoke test.",
    )
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--only-index", type=int, default=None)
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    steps = args.steps or preset["steps"]
    out = Path(args.out)
    grid = build_grid(preset, args.feature_maps, include_wide=not args.no_wide)

    if args.count:
        print(len(grid))
        return
    if args.only_index is not None:
        if not 0 <= args.only_index < len(grid):
            raise SystemExit(
                f"--only-index {args.only_index} out of range for {len(grid)} runs"
            )
        grid = [grid[args.only_index]]

    print(f"preset={args.preset} runs={len(grid)} steps={steps} -> {out}")

    for i, (feature_map, d_model, r, kv) in enumerate(grid, start=1):
        fixed = {**FIXED, "d_model": d_model}
        cfg = TrainConfig(
            mode=args.mode,
            feature_map=feature_map,
            redundancy_r=r,
            num_kv_pairs=kv,
            num_queries=min(8, kv),
            steps=steps,
            seed=args.seed,
            backend=args.backend,
            results_csv=str(out),
            eval_every=0,
            log_every=0,
            tag=f"stage3-{args.preset}",
            **fixed,
        )
        label = f"{feature_map} d_model={d_model} r={r} kv={kv}"
        if already_done_stage3(out, cfg):
            print(f"[{i}/{len(grid)}] skip (done): {label}")
            continue
        print(f"[{i}/{len(grid)}] {label}")
        train(cfg, verbose=True)


def already_done_stage3(path: Path, cfg: TrainConfig) -> bool:
    """Like stage2's check, plus the two axes Stage 3 adds.

    Without ``feature_map`` and ``d_model`` in the key, every arm at a given
    (mode, r, kv, steps, seed) would look already-done after the first one ran,
    and the sweep would silently record a single arm seven times.
    """
    if not path.exists():
        return False
    keys = ("mode", "redundancy_r", "num_kv_pairs", "steps", "seed",
            "feature_map", "d_model")
    want = {k: str(getattr(cfg, k)) for k in keys}
    with path.open(newline="", encoding="utf-8") as fh:
        return any(
            all(row.get(k) == v for k, v in want.items()) for row in csv.DictReader(fh)
        )


if __name__ == "__main__":
    main()
