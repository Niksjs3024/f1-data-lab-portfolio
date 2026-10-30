# Relatório técnico — F1 Data Lab

**Estado documental: 2026-10-23.** Este relatório diferencia o estudo original feito em Google Colab, a aplicação e os artefatos locais atualmente inspecionados e as validações históricas registradas. Nenhuma análise, aplicação ou suíte de testes foi executada para esta atualização documental.

## 1. Escopo e evolução do projeto

O estudo começou como exploração no Google Colab. O notebook e os scripts originais em `notebooks/` são registros históricos; a implementação atual está organizada como projeto Python local, com módulos de ingestão, auditoria, análise e modelagem, além de uma aplicação Streamlit. A interface contém seis áreas e modos educativos **Explicação simples** e **Visão técnica**.

O objetivo analítico é descrever frequências associadas à posição de largada registrada, comparar voltas rápidas de 2013 e 2014 e avaliar modelos com divisão temporal. O projeto não estima efeito causal de regulamentos ou clima e não se destina a previsões de apostas.

## 2. Arquitetura atual

- `app.py`: configuração da página, tema, carregamento da base principal e navegação/despacho das seis áreas.
- `src/pipeline.py`: leitura, integração e seleção dos registros históricos.
- `src/data_quality.py` e `src/audit_data.py`: auditoria não destrutiva, validação e CLI.
- `src/analysis.py` e `src/visualization.py`: frequências e comparação histórica; gráficos estáticos gerais.
- `src/model.py` e `src/model_figures.py`: regressão, classificação, baselines, avaliação temporal e figuras de diagnóstico.
- `src/historical_reconciliation.py`: preparação/avaliação de registros de reconciliação histórica.
- `src/climate_data.py`: processamento das observações climáticas locais e política QC.
- `src/dashboard/`: loaders, gráficos interativos, renderização das views e explicações/glossário.
- `tests/`: cobertura unitária e de interface; as fixtures compartilhadas identificam-se como sintéticas.

A separação de módulos não elimina a dependência de dados locais: a base histórica é carregada ao iniciar o app, e a área climática lê um conjunto adicional.

## 3. Dados, unidade de análise e qualidade

Os cinco CSVs históricos locais são `results.csv`, `races.csv`, `drivers.csv`, `constructors.csv` e `status.csv`, em `data/raw/`. No recorte padrão 2004–2017 há 5.671 registros antes da exclusão explicitamente contabilizada de 14 linhas com `grid = 0`, resultando em 5.657 registros.

A unidade das frequências é o registro de resultado. Cada grid usa seu próprio denominador; grid repetido não é deduplicado silenciosamente. Abandonos, resultados com `laps = 0` e ausências de volta rápida não são excluídos por essas razões. `grid` é a posição efetiva de largada registrada, não a posição de qualificação, e não comprova isoladamente que houve largada. `finish_position` deriva de `positionOrder`, não é uma substituição da classificação oficial `position` nem necessariamente uma chegada física.

A auditoria preserva e classifica achados sem corrigir os CSVs. No modo padrão, erros bloqueantes impedem validação; warnings permanecem visíveis e podem ser tolerados. `--strict` também bloqueia warnings. Em P17, a versão local apresenta 265 registros em 263 eventos distintos por grids positivos repetidos em duas corridas; a política atual mantém os registros e usa N=265.

## 4. Frequências observadas

No recorte 2004–2017, P1 tem 263 resultados; as frequências registradas são vitória 53,23%, pódio 79,85%, Top 10 88,97% e pontuação (`points > 0`) 87,83%. P2 e P3 têm, respectivamente, 262 e 263 registros; suas frequências de pódio são 63,36% e 48,29%.

São estatísticas descritivas dos arquivos fornecidos, não probabilidades universais, previsão ou causalidade. Como as regras de pontuação variaram, Top 10 e pontuação são indicadores diferentes. Pontuação ausente não é imputada como zero.

## 5. Voltas rápidas 2013–2014 e reconciliação

A análise atual pareia piloto, construtora e circuito, usando a volta mais rápida registrada em cada ano e calculando `2014 − 2013`. A versão local tem 165 pares: média **+3,256 s**, mediana **+2,100 s**. Ao excluir o Japão como análise de sensibilidade, restam 156 pares: média **+2,475641 s** (aproximadamente +2,476 s) e mediana **+2,000 s**. O Japão contribui com nove pares; sua exclusão não valida a identidade dos demais tempos nem transforma a comparação em estimativa oficial.

A auditoria histórica registra inconsistências de identidade de tempos de volta rápida na Austrália 2014. A análise principal preserva o pareamento observado nos CSVs fornecidos; não corrige os valores automaticamente.

O artefato de reconciliação mais recente contém 825 linhas-fonte, nenhuma linha de evento/piloto associada em correspondência externa, 825 registros ainda classificados como `nao_verificado` por campo, nenhuma correção e zero pares com precisão confirmada por fonte oficial. Assim, **não existe nesta revisão uma comparação reconciliada externa**. Os 165 pares e as estatísticas acima descrevem a fonte local; não são homologados. Não se atribui diferença ao regulamento de 2014, ao clima, à pista ou à estratégia.

## 6. Modelos e baselines

A entrada é somente `grid`. O treino cobre 2004–2014 (4.424 registros) e o teste temporal reservado cobre 2015–2017 (1.233 registros). A seleção de limiar da classificação ocorre no treino; a divisão de teste não é usada para escolher modelo ou threshold.

| Avaliação | Métrica | Resultado local |
|---|---|---:|
| Regressão linear | MAE | 3,7748 posições |
| Regressão linear | RMSE / R² | 4,7096 / 0,3830 |
| Baseline `posição prevista = grid` | MAE | 3,6926 posições |
| Baseline `posição prevista = grid` | RMSE / R² | 5,2139 / 0,2438 |
| Baseline constante pela mediana do treino | MAE | 5,1736 posições |
| Classificador logístico de pódio | Precisão / recall / F1 | 0,4624 / 0,9222 / 0,6160 |
| Classificador logístico de pódio | ROC-AUC / PR-AUC | 0,9399 / 0,7208 |
| Classificador logístico de pódio | Brier / log-loss | 0,0665 / 0,2220 |

A regressão linear não vence o baseline de grid em MAE. O classificador usa threshold 0,2 escolhido no treino e, no teste, tem 193 falsos positivos e 14 falsos negativos. Sua ROC-AUC é igual à AUC do ranking isolado por `-grid`; portanto, o resultado não demonstra ganho de ordenação sobre grid sozinho. Métricas de classificação não provam utilidade futura nem substituem avaliação de calibração e generalização.

## 7. Camada climática

A área **Contexto Climático 2013–2014** consome arquivos locais; não faz uma consulta NOAA ao navegar. Os registros selecionados abrangem dez eventos (cinco GPs em cada temporada), com 90 observações dentro das janelas UTC definidas em relação ao horário programado. A política documentada interpreta flags NOAA TMP/DEW separadamente: TMP tem 90/90 valores QC-aceitos; DEW tem 89/90 aceitos e um valor suspeito excluído somente das estatísticas QC de DEW.

Observações são de estações regionais, não da superfície da pista. O horário programado não é a largada real. A cobertura não é contínua; timestamps não são pareados às voltas rápidas; evidências narrativas de corrida/qualificação não se convertem em medição instrumental. Esses dados não permitem concluir efeito do clima sobre tempos ou resultados.

## 8. Artefatos e estado de validação

Na inspeção documental atual havia 17 arquivos `test_*.py` e 10 imagens PNG em `outputs/figures/`. Esses são inventários de arquivos, **não** prova de execução recente nem garantia de que estejam autorizados para redistribuição. A documentação antiga mencionava três testes, três PNGs e ausência do Streamlit; essas afirmações descrevem uma validação histórica e não o inventário atual.

A última execução completa efetivamente conhecida no histórico desta colaboração foi `python -m pytest -q`, com **330 testes aprovados**, na Fase 5 (refinamento das explicações educativas, formatação pt-BR e revisões de dashboard). **A data exata dessa execução não está registrada nas evidências disponíveis.** Esse resultado é histórico: a suíte não foi reexecutada nesta Fase 6. Também não houve nesta etapa execução do app, teste de navegador ou `python -m src.run_analysis`.

`outputs/metrics.json`, tabelas, figuras, arquivos climáticos e relatórios de reconciliação são artefatos locais já existentes. `src.run_analysis` gera e pode sobrescrever outputs; não foi executado. O manifesto `outputs/climate_phase3c_20261022/MANIFEST_SHA256_20261022.json` é um inventário criado como referência do estado após a Fase 3B para a Fase 3C, com `historical_integrity_claim=false`; não prova preservação desde fases anteriores.

## 9. Proveniência, publicação e limitações

A distribuição dos cinco CSVs históricos locais foi identificada como *Formula 1 Race Data*, publicada por cjgdev no [Kaggle](https://www.kaggle.com/datasets/cjgdev/formula-1-race-data-19502017); a página declara CC BY-NC-SA 4.0. Na Fase 7C, os cinco arquivos locais coincidiram byte a byte com os membros correspondentes do ZIP comparado, e os SHA-256 coincidiram. Essa evidência identifica a distribuição comparada, mas não a fonte primária anterior, o release exato, a cadeia de direitos nem a autorização de redistribuição pública. As condições aplicáveis aos dados NOAA/Jolpica/F1 e às respectivas cópias ou derivados são registradas separadamente em [docs/PROVENIENCIA_E_LICENCAS.md](docs/PROVENIENCIA_E_LICENCAS.md). Atribuição, citação, hash ou transformação não equivalem a autorização de redistribuição.

O código do projeto também não tem licença de software escolhida nesta etapa. `requirements.txt` declara intervalos de versões das dependências, não um lockfile nem a licença de cada pacote. Antes de uma publicação, revisar os direitos do código e dos materiais de terceiros e confirmar o estado de rastreamento no Git. A CLI `git` não estava disponível no ambiente desta revisão, logo não foi possível verificar quais dados ou outputs já estão rastreados. Nenhum arquivo rastreado foi removido ou alterado por Git.

Uma cópia somente de código não inicializa o app sem os cinco CSVs de `data/raw/`; a view climática também requer arquivos próprios. Uma demonstração sintética seria uma opção futura, mas não está implementada. O `.gitignore` reduz inclusões acidentais e não remove conteúdo já rastreado nem concede direitos.

## Referências do projeto

- [README](README.md) — visão geral e execução local.
- [Contrato de dados](DATA_CONTRACT.md) — esquema, semântica, validação e proveniência pendente.
- [Registro de proveniência e licenças](docs/PROVENIENCIA_E_LICENCAS.md) — fontes, condições e pendências por categoria.
