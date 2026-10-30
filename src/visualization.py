"""Exportação dos gráficos para README, LinkedIn e portfólio."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_figures(df: pd.DataFrame, probabilities: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sample = probabilities.loc[probabilities["grid"].between(1, 10)].set_index("grid")
    ax = sample[["vitoria_pct", "podio_pct", "pontos_pct"]].plot(
        kind="bar", figsize=(12, 6), color=["#dd3322", "#2b567d", "#d1a044"], width=.8)
    ax.set(title="Desempenho por posição de largada | 2004–2017", xlabel="Posição no grid",
           ylabel="Proporção dos resultados (%)", ylim=(0, 105))
    ax.legend(["Vitória", "Pódio", "Pontuou"])
    ax.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(output / "probabilidades_por_grid.png", dpi=170)
    plt.close()

    subset = df[df["grid"].between(1, 20) & df["finish_position"].between(1, 20)]
    crosstab = pd.crosstab(subset["grid"], subset["finish_position"])
    crosstab = crosstab.reindex(index=range(1, 21), columns=range(1, 21), fill_value=0)
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    image = ax.imshow(crosstab.to_numpy(), origin="upper", cmap="inferno", aspect="auto")
    ax.set(xticks=range(20), xticklabels=range(1, 21), yticks=range(20),
           yticklabels=range(1, 21), xlabel="Posição final registrada", ylabel="Posição no grid",
           title="Frequência de grid × resultado | 2004–2017")
    fig.colorbar(image, ax=ax, label="Quantidade de resultados")
    fig.tight_layout()
    fig.savefig(output / "mapa_calor_grid_chegada.png", dpi=170)
    plt.close(fig)

    if not paired.empty:
        by_gp = paired.groupby("grand_prix")["diferenca_2014_menos_2013_seg"].median().sort_values()
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(by_gp.index, by_gp.to_numpy(), color="#b23b3b")
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(title="2014 − 2013: mediana da diferença na melhor volta por GP",
               xlabel="Segundos (positivo = volta mais lenta em 2014)")
        fig.tight_layout()
        fig.savefig(output / "comparacao_2013_2014.png", dpi=170)
        plt.close(fig)
