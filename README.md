# F1 Data Lab

**Análise histórica da Fórmula 1 com Python, estatística, machine learning e Streamlit**

O F1 Data Lab transforma um estudo exploratório que começou no Google Colab em uma aplicação local, modular e auditável. A versão atual separa ingestão e validação, análise, modelagem, visualizações e interface. A pasta `notebooks/` conserva o contexto histórico; não é a fonte de verdade da implementação atual.

> **Autorização para redistribuição dos dados ainda não confirmada.** Os cinco CSVs históricos locais correspondem byte a byte aos membros comparados na Fase 7C da distribuição *Formula 1 Race Data*, publicada por cjgdev no [Kaggle](https://www.kaggle.com/datasets/cjgdev/formula-1-race-data-19502017); a página declara CC BY-NC-SA 4.0. Essa correspondência e declaração não confirmam a fonte primária anterior ao Kaggle, a versão/release exato, a cadeia de direitos nem autorização para redistribuição pública. Dados climáticos, dados de terceiros e resultados derivados também não são automaticamente liberados. Este repositório não deve incluir `data/` ou `outputs/` até haver confirmação documentada dos direitos aplicáveis. Consulte [Proveniência e licenças](docs/PROVENIENCIA_E_LICENCAS.md).

> **Uma cópia apenas do código não reproduz a aplicação completa.** O dashboard exige os cinco CSVs históricos locais para iniciar. A área climática requer arquivos locais adicionais. A distribuição Kaggle correspondente aos CSVs foi identificada, mas isso não equivale a confirmar autorização de aquisição ou redistribuição para este projeto; os arquivos não são distribuídos aqui.

## O projeto

O estudo inicial foi desenvolvido em Google Colab, com caminhos e etapas exploratórias vinculados àquele ambiente. A versão atual organiza o trabalho como um projeto Python executável localmente: o pipeline integra tabelas por IDs, as regras de qualidade ficam em módulos próprios, a análise e os modelos são separados da interface e o Streamlit orquestra as áreas do dashboard. Os notebooks permanecem como registro histórico, não como substitutos do código em `src/`.

### Seis áreas e modos de leitura

O dashboard oferece seis áreas e um seletor de explicação global:

| Área | O que apresenta |
|---|---|
| **Visão Geral** | Escopo padrão, indicadores e frequências observadas por posição de largada. |
| **Explorador de Corridas** | Recortes por temporada, corrida, piloto, construtora e grid, com frequências, mapa de calor e tabela. |
| **Comparação Histórica 2013–2014** | Pares de voltas rápidas do mesmo piloto, equipe e circuito, mais análise de sensibilidade. |
| **Laboratório de Machine Learning** | Regressão e classificação com divisão temporal, baselines e diagnósticos. |
| **Dados e Metodologia** | Cobertura, auditoria dos dados locais, achados e limitações de proveniência. |
| **Contexto Climático 2013–2014** | Observações de estações regionais, flags de qualidade NOAA e evidências documentais separadas. |

Os modos **Explicação simples** e **Visão técnica** acompanham as áreas, com glossário compartilhado. Os filtros do Explorador são locais àquela área; não alteram o período de avaliação dos modelos.

## Resultados de referência

Os valores a seguir descrevem os artefatos locais existentes; não são resultados de uma nova execução nesta etapa documental. A execução que geraria tabelas, gráficos ou métricas pode sobrescrever arquivos de `outputs/`.

### Frequências por grid

O recorte padrão é 2004–2017, com `grid > 0`. Há 5.657 registros mantidos depois de contabilizar 14 registros com `grid = 0`. Para P1, a base contém 263 resultados:

| Medida para P1 | Frequência observada |
|---|---:|
| Vitória pela ordenação registrada | 53,23% |
| Pódio pela ordenação registrada | 79,85% |
| Top 10 pela ordenação registrada | 88,97% |
| Pontuação registrada (`points > 0`) | 87,83% |

Essas são frequências empíricas da amostra, não efeitos causais nem garantias para corridas futuras. Cada grid usa seu próprio denominador. `grid` é a posição efetiva de largada registrada, não a posição da qualificação; `finish_position` deriva de `positionOrder` e não equivale necessariamente à classificação oficial ou à chegada física.

### Voltas rápidas: 2013 versus 2014

A comparação atual contém 165 pares por piloto, construtora e circuito. A diferença é **2014 − 2013**: média **+3,256 s** e mediana **+2,100 s**. Excluindo o Japão, restam 156 pares; a média é **+2,476 s** e a mediana **+2,000 s**.

A exclusão do Japão é uma análise de sensibilidade, não uma validação da amostra. A auditoria documenta um problema de identidade dos tempos de volta rápida na Austrália 2014, e os artefatos de reconciliação disponíveis não estabelecem um conjunto de pares confirmado em precisão oficial. Estes números descrevem os CSVs locais; a comparação não é homologada e não identifica o efeito causal do regulamento, clima, pista ou estratégia.

### Machine learning

A entrada dos modelos é somente `grid`. O treino usa 2004–2014; o teste temporal reservado usa 2015–2017 (1.233 registros).

| Modelo ou referência | Métrica no teste | Valor |
|---|---|---:|
| Regressão linear | MAE | 3,7748 posições |
| Baseline: prever posição final igual ao grid | MAE | 3,6926 posições |
| Baseline: mediana do treino | MAE | 5,1736 posições |
| Regressão linear | RMSE / R² | 4,7096 / 0,3830 |
| Classificador de pódio | Precisão / recall | 0,4624 / 0,9222 |
| Classificador de pódio | ROC-AUC / PR-AUC | 0,9399 / 0,7208 |

A regressão linear **não superou** o baseline `posição prevista = grid` na métrica principal MAE. O classificador usa limiar 0,2 escolhido somente no treino; sua ROC-AUC coincide com a AUC da ordenação pelo grid isolado. Assim, a AUC alta não demonstra ganho de ranking sobre esse sinal simples, e a precisão observada implica muitos falsos positivos. Os modelos são exploratórios, não previsões validadas para uso prático.

### Contexto climático

Os artefatos locais registram dez consultas de evento e 90 observações NOAA selecionadas em janelas ao redor do horário programado. Na política QC documentada, 90/90 valores de temperatura do ar e 89/90 valores de ponto de orvalho são aceitos. As estações são regionais, os horários não são a largada real e a documentação narrativa não é uma medição instrumental. Não há pareamento temporal com as voltas rápidas; portanto, o dashboard não estima efeito de clima sobre o desempenho.

## Arquitetura

- `app.py`: configura e despacha a aplicação Streamlit.
- `src/pipeline.py`, `src/data_quality.py` e `src/audit_data.py`: leitura, integração, auditoria e validação dos CSVs históricos.
- `src/analysis.py` e `src/visualization.py`: frequências, comparação histórica e gráficos da análise.
- `src/model.py` e `src/model_figures.py`: modelos, baselines, avaliação temporal e figuras de diagnóstico.
- `src/climate_data.py` e `src/historical_reconciliation.py`: coleta/processamento local documentado e reconciliação histórica; não implicam autorização para redistribuir os dados usados.
- `src/dashboard/`: carregamento, gráficos, explicações educativas e renderização das seis áreas.
- `tests/`: testes unitários, de contrato e de interface; as fixtures compartilhadas se identificam como sintéticas.
- `data/` e `outputs/`: material de trabalho local, não presumido publicável.

## Requisitos e execução local

A versão de referência documentada é Python 3.12; `requirements.txt` declara intervalos de versões das dependências, não uma matriz de compatibilidade nem um lockfile. Os comandos abaixo são para uso local em Windows/PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Antes de iniciar o dashboard, os cinco arquivos abaixo precisam estar disponíveis localmente em `data/raw/`:

- `results.csv`
- `races.csv`
- `drivers.csv`
- `constructors.csv`
- `status.csv`

**Eles não fazem parte desta distribuição e não há autorização de redistribuição confirmada.** A distribuição identificada é *Formula 1 Race Data*, por cjgdev no [Kaggle](https://www.kaggle.com/datasets/cjgdev/formula-1-race-data-19502017); a página declara CC BY-NC-SA 4.0. Na Fase 7C, os cinco CSVs locais coincidiram byte a byte com os membros correspondentes do ZIP comparado. A fonte primária anterior, o release exato, a cadeia de direitos e as condições de uso aplicáveis continuam pendentes; não obtenha nem redistribua arquivos sem confirmar que está autorizado. Sem esses arquivos, o carregamento principal interrompe o app com erro, e clonar o código não basta para abrir todas as áreas.

Para iniciar, depois de preparar os dados localmente:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

A área **Contexto Climático 2013–2014** necessita, adicionalmente, dos arquivos locais definidos em `src/dashboard/data.py`:

- `data/climate_phase3b_20261022/events.csv`
- `data/climate_phase3b_20261022/race_evidence.csv`
- `data/climate_phase3c_20261022/noaa_observations_quality.csv`
- `outputs/climate_phase3c_20261022/quality_summary_by_event.csv`
- `outputs/climate_phase3c_20261022/quality_summary_common_report_types.csv`

Sem o conjunto climático, essa área apresenta indisponibilidade local; isso não substitui os arquivos históricos necessários ao restante do app. O dashboard não deve buscar os dados na rede ao abrir.

## Validação local em ambiente independente

Em 28/10/2026, o projeto foi validado em uma instalação
independente no Windows, utilizando Python 3.12.

- Código clonado do repositório GitHub privado.
- Dependências instaladas em novo ambiente virtual.
- Arquivos de dados e gráficos copiados localmente, com integridade verificada por SHA-256.
- Suíte automatizada: **331 testes aprovados, nenhuma falha ou teste ignorado**.
- As seis áreas do Streamlit foram verificadas manualmente no navegador local.
- Nenhuma alteração pendente no Git ao término da validação.

Os datasets e gráficos necessários à validação não integram o repositório.

Essa validação demonstra reprodutibilidade local nas condições testadas.
Não comprova compatibilidade com hospedagem em nuvem nem autorização
para redistribuição pública dos dados e resultados derivados.

## Testes e auditoria

Com dependências instaladas, o comando da suíte é:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

As fixtures sintéticas permitem testar vários contratos sem expor os CSVs históricos, mas não assumem que uma cópia pública sem `data/` e `outputs/` consiga executar toda a suíte ou o aplicativo. Consulte [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md) para o último estado de validação conhecido e seu contexto.

A auditoria local pode ser executada com:

```powershell
.\.venv\Scripts\python.exe -m src.audit_data
```

Ela grava relatórios em `outputs/`. `--strict` trata warnings como bloqueantes, mas ainda grava o relatório antes de retornar erro. `python -m src.run_analysis` gera métricas, tabelas, figuras e auditoria; use-o somente quando for aceitável regenerar os artefatos locais. Não foi executado nesta etapa documental.

## Instalação limpa, demonstração e limites de publicação

Uma cópia de código sem dados pode servir para revisão do código e, após conferir os requisitos de cada teste, para executar testes baseados em fixtures sintéticas. Não é uma demonstração funcional do dashboard. Uma futura versão demonstrativa sintética poderia apresentar apenas dados inventados e identificados como sintéticos; **essa modalidade não está implementada** e não deve ser inferida a partir dos dados locais atuais.

O `.gitignore` protege `data/`, `outputs/`, notebooks históricos, ambientes e configurações locais para reduzir inclusões acidentais. Ele não remove arquivos previamente rastreados e não é uma licença. Antes de publicar, confirme o estado de rastreamento e remova do pacote de publicação, com autorização apropriada, qualquer conteúdo que não esteja liberado.

Veja [DATA_CONTRACT.md](DATA_CONTRACT.md), [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md) e [docs/PROVENIENCIA_E_LICENCAS.md](docs/PROVENIENCIA_E_LICENCAS.md). Este projeto não declara afiliação ou endosso da Formula 1, da FIA, de equipes ou de pilotos. Não foi criada uma licença para o código; não presuma permissões de reutilização do software ou dos dados.
