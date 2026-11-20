"""Plot paired accuracy gains for the candidate-only CPU router across the three evaluated pools."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from route_query import ROOT

REPORT = ROOT / "reports/routing/capability"


def main():
    face = Path("C:/Windows/Fonts/msjh.ttc")
    if face.is_file():
        font_manager.fontManager.addfont(str(face))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=face).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    rows = []
    for name in ("audited_13", "expanded_37"):
        file = REPORT / f"confirmation/equal_200_{name}_paired.csv"
        result = pd.read_csv(file).query("split=='confirmation'").set_index("reference")
        rows.append(
            [result.loc[ref] for ref in ("deterministic", "full-deterministic")]
        )
    folder = REPORT / "release_confirmation/coverage"
    rows.append(
        [
            pd.read_csv(folder / file)
            .query("split=='official' and reference=='deterministic'")
            .iloc[0]
            for file in ("equal_200_paired.csv", "paired.csv")
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.65), sharex=True, sharey=True)
    colors = ("#126265", "#8C4929")
    largest = 100 * max(
        max(abs(r.micro_low), abs(r.micro_high)) for pair in rows for r in pair
    )
    limit = max(8, int(2 * np.ceil((largest + 1) / 2)))
    for index, (ax, title, color) in enumerate(
        zip(axes, ("雙方使用相同 200 題", "EmbedLLM 使用完整訓練資料"), colors)
    ):
        for y, pair in enumerate(rows):
            r = pair[index]
            score, low, high = 100 * np.asarray([r.gain, r.micro_low, r.micro_high])
            ax.errorbar(
                score,
                y,
                xerr=[[score - low], [high - score]],
                fmt="o",
                color=color,
                ms=6,
                lw=2,
                capsize=4,
            )
            ax.annotate(
                f"{score:+.2f}",
                (score, y),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=10,
            )
        ax.axvline(0, color="#56646B", lw=0.9)
        ax.grid(axis="x", color="#D8DFE1", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(2.5, -0.57)
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("正確率差（百分點）")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#B9C3C6")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(
        [0, 1, 2], ["13 模型 · 4,985 題", "37 模型 · 4,985 題", "112 模型 · 3,000 題"]
    )
    fig.suptitle(
        "200 題能力向量相對於確定性 EmbedLLM 的正確率",
        x=0.51,
        y=0.99,
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.12,
        0.02,
        "點為 15 份題單的平均差。線段為同時重抽題單與題組的 95% 配對區間。",
        fontsize=9,
        color="#46535A",
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.77, bottom=0.22, wspace=0.14)
    fig.savefig(REPORT / "paired_accuracy.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
