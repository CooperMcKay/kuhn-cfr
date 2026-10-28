"""Train CFR, evaluate it, and write results/ (convergence.png, results.json, results.md)."""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kuhn.baselines import BASELINES  # noqa: E402
from kuhn.cfr import train  # noqa: E402
from kuhn.evaluate import exploitability, game_value, head_to_head, simulate_win_rate  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHECKPOINTS = (1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000)
FINAL_ITERATIONS = CHECKPOINTS[-1]
SIM_HANDS = 100_000
SEED = 0
GAME_VALUE = -1 / 18


def convergence_curve():
    # Training is deterministic, so retraining per checkpoint matches one long run.
    return [(n, exploitability(train(n))) for n in CHECKPOINTS]


def plot_convergence(curve, path):
    iterations, exploits = zip(*curve)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(iterations, exploits, marker="o")
    ax.set_xlabel("CFR iterations")
    ax.set_ylabel("Exploitability (chips/hand)")
    ax.set_title("Kuhn poker: CFR convergence")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_baselines(strategy):
    rows = {}
    for name, build in BASELINES.items():
        opponent = build()
        rows[name] = {
            "avg_chips_per_hand": head_to_head(strategy, opponent),
            "win_rate": simulate_win_rate(strategy, opponent, SIM_HANDS, SEED),
        }
    return rows


def write_markdown(results, path):
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Iterations | {results['iterations']:,} |",
        f"| Training runtime | {results['train_seconds']:.2f} s |",
        f"| Final exploitability | {results['final_exploitability']:.6f} chips/hand |",
        f"| Game value (P1) | {results['game_value']:.5f} (exact: {GAME_VALUE:.5f}) |",
        "",
        f"| Opponent | Avg chips/hand (exact) | Win rate ({SIM_HANDS:,} hands) |",
        "|---|---|---|",
    ]
    for name, row in results["baselines"].items():
        lines.append(f"| {name} | {row['avg_chips_per_hand']:+.4f} | {row['win_rate']:.1%} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    start = time.perf_counter()
    strategy = train(FINAL_ITERATIONS)
    train_seconds = time.perf_counter() - start

    curve = convergence_curve()
    plot_convergence(curve, RESULTS_DIR / "convergence.png")

    results = {
        "iterations": FINAL_ITERATIONS,
        "train_seconds": train_seconds,
        "final_exploitability": exploitability(strategy),
        "game_value": game_value(strategy, strategy),
        "convergence": curve,
        "baselines": evaluate_baselines(strategy),
        "strategy": strategy,
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, RESULTS_DIR / "results.md")
    print((RESULTS_DIR / "results.md").read_text())


if __name__ == "__main__":
    main()
