"""Testes de integridade dos PNGs exportados em outputs/figures.

Condicionais a disponibilidade dos arquivos fisicos (nao regenerados por estetica).
"""

from pathlib import Path

import pytest

FIGURES_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"

ESPERADOS = [
    "comparacao_2013_2014.png",
    "mapa_calor_grid_chegada.png",
    "modelo_curva_calibracao.png",
    "modelo_curva_pr.png",
    "modelo_curva_roc.png",
    "modelo_distribuicao_erros.png",
    "modelo_mae_comparacao.png",
    "modelo_observado_vs_previsto.png",
    "modelo_validacao_temporal_treino.png",
    "probabilidades_por_grid.png",
]


@pytest.mark.skipif(not FIGURES_DIR.exists(), reason="outputs/figures nao disponível no ambiente")
@pytest.mark.parametrize("nome", ESPERADOS)
def test_png_existe_nao_vazio_com_assinatura(nome):
    caminho = FIGURES_DIR / nome
    assert caminho.is_file(), f"PNG esperado ausente: {nome}"
    dados = caminho.read_bytes()
    assert len(dados) > 0, f"PNG vazio: {nome}"
    assert dados[:8] == b"\x89PNG\r\n\x1a\n", f"Assinatura PNG invalida: {nome}"


@pytest.mark.skipif(not FIGURES_DIR.exists(), reason="outputs/figures nao disponível no ambiente")
def test_pngs_sao_dez_e_decodaveis():
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow nao instalado; nao e possivel decodificar PNGs")
    nomes = {p.name for p in FIGURES_DIR.glob("*.png")}
    assert set(ESPERADOS).issubset(nomes), f"Faltam PNGs: {set(ESPERADOS) - nomes}"
    for nome in ESPERADOS:
        imagem = Image.open(FIGURES_DIR / nome)
        imagem.load()
        largura, altura = imagem.size
        assert largura > 0 and altura > 0, f"Dimensoes invalidas em {nome}"
