# Contrato de dados — F1 Data Lab

Este documento define a interpretação dos CSVs locais, as regras de auditoria e validação e os limites de uso da base. O [README](README.md) contém o fluxo de execução. A implementação reprodutível está em `src/`; `notebooks/` é apenas histórico.

**Recorte padrão:** 2004–2017. **Ambiente de referência testado:** Python 3.12, sem declarar aprovação antecipada de novos testes ou validação de navegador. A presente documentação não altera os arquivos originais nem resolve sua licença.

## 1. Inventário e cobertura

| Tabela / arquivo em `data/raw/` | Registros locais | Chave primária | Papel |
|---|---:|---|---|
| `results.csv` | 23.777 | `resultId` | Registros de resultados de 1950–2017 |
| `races.csv` | 997 | `raceId` | Calendário de 1950–2018 |
| `drivers.csv` | 842 | `driverId` | Dimensão de pilotos |
| `constructors.csv` | 208 | `constructorId` | Dimensão de equipes/construtores |
| `status.csv` | 134 | `statusId` | Dimensão de situações do resultado |

**Calendário não equivale a resultados:** os **21 eventos de 2018** presentes em `races.csv` não têm linhas em `results.csv`. Não apresentar esta base como resultados até 2018. As dimensões também não implicam que todos os pilotos, equipes ou status apareçam no recorte analítico.

Para **2004–2017**:

| Etapa | Registros |
|---|---:|
| Resultados no período, antes do filtro de grid | 5.671 |
| Exclusão contabilizada: `grid = 0` | 14 |
| Resultados mantidos | 5.657 |

Estes são valores de referência da versão local recebida, não constantes a inserir nas métricas ou gráficos. A auditoria de cada execução deve registrar a cobertura e as exclusões efetivamente observadas.

## 2. Unidade de registro, chaves e relacionamentos

- A unidade original de `results` é o **registro de resultado identificado por `resultId`**. Não assumir que isso torna globalmente única a participação `(raceId, driverId)` em toda a história.
- Todas as PKs devem ser presentes, válidas e únicas. IDs não devem ser inventados, arredondados ou preenchidos para contornar falhas.
- FKs obrigatórias de `results`: `raceId → races.raceId`, `driverId → drivers.driverId`, `constructorId → constructors.constructorId` e `statusId → status.statusId`.
- As junções são **muitos-para-um** em cada dimensão e não podem multiplicar resultados. PK/FK inválida é bloqueante; filtro temporal não deve ocultar uma falha estrutural das tabelas fornecidas.
- `circuitId` é preservado como identificador do circuito. Não há `circuits.csv` entre os cinco arquivos contratados; portanto, não se declara verificada uma FK contra essa dimensão ausente.
- Participação `(raceId, driverId)` repetida no período moderno selecionado (**anos a partir de 2004**, conforme a política implementada) é bloqueante, mesmo com `resultId` distinto. As repetições históricas de **1950–1978** são auditadas separadamente e não devem ser apagadas por uma regra global de deduplicação.
- Um grid positivo repetido entre pilotos diferentes **não é a mesma coisa que participação duplicada**; ele gera warning, conforme a política abaixo.

## 3. Esquema de entrada e semântica de saída

### Núcleo utilizado na integração

| Tabela | Colunas de integração e análise |
|---|---|
| `results` | `resultId`, `raceId`, `driverId`, `constructorId`, `statusId`, `grid`, `positionOrder`, `points` |
| `races` | `raceId`, `year`, `circuitId`, `name` |
| `drivers` | `driverId`, `forename`, `surname` |
| `constructors` | `constructorId`, `name` |
| `status` | `statusId`, `status` |

Ausência de tabela ou coluna obrigatória é achado de esquema bloqueante, com diagnóstico de auditoria em vez de exclusão silenciosa. As colunas `laps`, `position`, `positionText`, `fastestLapTime` e `fastestLapSpeed` são opcionais na API; sua ausência é auditada como informação e os campos permanecem desconhecidos. Quando uma coluna de volta rápida existe, **suas células podem estar ausentes** sem invalidar o resultado. Colunas extras não autorizam descarte ou reescrita dos CSVs originais.

### Campos preservados e derivados

| Entrada | Saída integrada | Significado / tratamento |
|---|---|---|
| `resultId` | `result_id` | Identificador do registro; rastreabilidade até o original |
| `raceId` | `race_id` | Evento do calendário |
| `driverId` | `driver_id` | Piloto |
| `constructorId` | `constructor_id` | Equipe/construtor |
| `statusId` | `status_id` | Identificador do status |
| `circuitId` | `circuit_id` | Circuito, sem dimensão adicional no pacote |
| `year` | `year` | Temporada usada para filtrar |
| `grid` | `grid` | Posição efetiva de largada registrada; não é colocação na qualificação |
| `positionOrder` | `finish_position` | **Ordenação do resultado**; continua sendo a referência das taxas de vitória, pódio e Top 10 |
| `position` | `position` | **Posição/classificação oficial registrada**, potencialmente ausente; não é substituída por `positionOrder` |
| `positionText` | `position_text` | Representação textual original da classificação/situação; preservada sem reinterpretar códigos arbitrariamente |
| `laps` | `laps` | Número de voltas registrado; zero é valor informativo válido |
| `points` | `points` | Pontos registrados; ausente continua `NaN` |
| — | `points_known` | Indicador de disponibilidade da pontuação; `False` quando desconhecida |
| `fastestLapTime` | `fastest_lap_time` | Tempo da volta mais rápida; ausências não invalidam o resultado |
| `fastestLapSpeed` | `fastest_lap_speed` | Velocidade da volta mais rápida; valor inválido não essencial é auditado |
| `status.status` | `status` | Situação do resultado; preserva abandono, retirada e demais categorias |
| `races.name` | `grand_prix` | Nome do GP |
| `constructors.name` | `constructor` | Nome da equipe |
| `forename` + `surname` | `driver` | Rótulo do piloto; IDs continuam sendo a referência para junções |

**Compatibilidade:** fixtures antigos podem não conter `laps`, `position` e `positionText`. Esses campos são opcionais para compatibilidade, e sua ausência é auditada. Compatibilidade não significa que existam zero voltas ou classificação oficial conhecida: não preencher informação inexistente com valores factuais.

### Grid, classificação e status não são equivalentes

`grid` se refere à posição efetiva de largada registrada, que pode diferir da qualificação por penalizações e regras. **Um número positivo não comprova, isoladamente, que houve largada efetiva.** Não inferir isso sem examinar status, voltas e o contexto da fonte.

`finish_position` é a ordenação `positionOrder`; um abandono pode ter uma ordenação mesmo sem `position` oficial preenchida. Assim, `finish_position <= 10` descreve a ordenação registrada, não necessariamente chegada física, classificação oficial válida ou pontuação. O status e os campos originais permanecem disponíveis para análises mais específicas.

## 4. Validação, auditoria e modo estrito

### Separação de responsabilidades

- **`audit_tables`** inspeciona tabelas sem modificar os DataFrames de entrada, produzindo um relatório de achados; relatar erro de dados não equivale a lançar erro de validação imediatamente. Não corrige nem remove registros.
- **`validate_tables`** utiliza os achados para aplicar a política. Por padrão, bloqueia erros impeditivos e aceita warnings; com **`strict=True`**, warnings também impedem a aprovação. Falhas de validação usam **`DataValidationError`**, subclasse de `ValueError`.
- **`build_dataset`** mantém os defaults **2004–2017**, seleção de grid positivo e política não estrita. Não se muda o comportamento padrão do dashboard para modo estrito.

| Achado | Política padrão | Modo estrito |
|---|---|---|
| Tabela/coluna obrigatória ausente; esquema inválido | Bloqueia | Bloqueia |
| PK, FK ou ID obrigatório inválido, nulo ou inconsistente | Bloqueia | Bloqueia |
| Participação moderna duplicada no período selecionado | Bloqueia | Bloqueia |
| Numérico essencial inválido | Bloqueia | Bloqueia |
| Mesmo grid positivo em mais de um resultado da corrida | **WARNING; mantém registros** | Bloqueia por warning |
| Pontuação desconhecida | Mantém `NaN`, sinaliza disponibilidade | Warning impede aprovação |
| Valor de `laps` ausente no período | WARNING; preserva como desconhecido | Bloqueia por warning |
| Coluna opcional inteira ausente | INFO; mantém compatibilidade | Informação isolada não bloqueia |
| `grid = 0` no período selecionado | Exclusão explicitamente contabilizada | Não transforma zero em grid positivo |

O modo estrito não modifica os dados nem converte warnings em correções automáticas. A contagem de uma exclusão informativa, sozinha, não é sinônimo de erro. A severidade de cada achado deve ser lida no relatório.

### Domínios numéricos e ausências

- IDs devem ser inteiros válidos; `year` precisa permitir seleção temporal sem coerções silenciosas que escondam inconsistências.
- No recorte selecionado, `grid` deve ser inteiro finito não negativo; `grid = 0` é o caso especial de exclusão contado. Grid ausente, texto inválido, negativo, fracionário ou infinito não deve simplesmente desaparecer no filtro.
- `positionOrder` precisa ser válido para a ordenação analítica. Ausência ou numérico essencial inválido bloqueiam, em vez de remover a linha sem rastreabilidade.
- `points`, **quando presente**, deve ser numérico, finito e não negativo. Pontuação ausente é diferente de texto numérico inválido e é diferente de zero.
- `laps`, quando presente, deve ser inteiro finito não negativo. Ausência é auditada; **zero não invalida a linha**. Não aplicar `laps > 0` como filtro de largadas ou resultados.
- Volta rápida ausente ou inválida afeta sua análise específica, não autoriza excluir o registro da análise de resultados. Não aplicar `dropna()` indiscriminadamente.
- Ausências conhecidas na leitura devem permanecer reconhecíveis como ausências; não confundir tokens de origem com valores conhecidos nem preencher campos por conveniência.

### Formato dos achados

O relatório JSON contém uma lista `issues`. Cada achado possui ao menos:

| Campo | Conteúdo |
|---|---|
| `code` | Código identificador do achado |
| `table` | Tabela afetada |
| `severity` | `error`, `warning` ou `info` |
| `count` | Contagem inteira não negativa |
| `examples` | Lista de exemplos para investigação |
| `blocking` | Se bloqueia na política padrão; strict também considera warnings |

A representação JSON deve ser serializável sem valores JSON inválidos como `NaN` ou `Infinity` literais. Isso não muda a semântica dos DataFrames: **`points` e `pontos_pct` desconhecidos continuam `NaN` na análise**, com representação apropriada no relatório.

## 5. Seleção e denominadores

1. Auditar e validar esquema, identificadores e relacionamentos, sem usar o recorte para esconder PK/FK inválida.
2. Selecionar o período solicitado, **2004–2017 por padrão**.
3. Validar os valores essenciais do período e contabilizar registros excluídos por `grid = 0`.
4. Manter os registros com grid positivo, sem deduplicar arbitrariamente grid, sem exigir `laps > 0` e sem excluir abandonos ou volta rápida ausente.
5. Agrupar por `grid` e usar a contagem de **todos os registros desse grid no filtro** como denominador.

Na tabela de frequências, `corridas` é o nome histórico da coluna de contagem de registros do grid, **não garantia de número de corridas distintas quando há grid repetido**. Para cada grupo com `N` registros:

- `vitoria_pct`: `100 × contagem(finish_position == 1) / N`.
- `podio_pct`: `100 × contagem(1 <= finish_position <= 3) / N`.
- `top10_pct`: `100 × contagem(1 <= finish_position <= 10) / N`.
- `pontos_pct`: `100 × contagem(points > 0) / N` **somente se todos os pontos do grupo forem conhecidos**.

Se qualquer registro do grupo tiver pontuação desconhecida, `points` permanece `NaN`, `points_known=False` nesse registro e **`pontos_pct=NaN` para o grupo inteiro**. Não interpretar o desconhecido como zero e não passar a dividir só pelos casos conhecidos. **Vitória, pódio, Top 10 e `corridas` são preservados**, desde que seus próprios campos necessários sejam válidos.

As regras históricas de pontuação variaram. `points > 0` não pode ser substituído por Top 10. Frequências observadas não são garantias de eventos futuros.

## Anomalias conhecidas e tratamento

| Achado na versão local | Tratamento / pendência |
|---|---|
| `raceId=938`, grid P17 repetido: `resultId=22766` / `driverId=832` e `resultId=22768` / `driverId=831` | Warning não bloqueante no padrão. Manter ambos e investigar na fonte; sem correção arbitrária. |
| `raceId=941`, grid P17 repetido: `resultId=22829` / `driverId=829` e `resultId=22835` / `driverId=828` | Mesma política: manter, auditar e deixar pendente. |
| P17 no recorte: **265 registros em 263 corridas distintas** | Denominador atual **N = 265**. Não trocar apenas para 263: isso mudaria a unidade de análise sem corrigir/documentar os dados e o numerador. |
| `fastestLapSpeed` inválida em `resultId=23769` | Auditar como valor não essencial inválido; não remover o resultado da análise de chegada nem inventar velocidade. |
| Nomes com problemas de encoding | Preservar originais e IDs; eventual normalização deve ser explícita e rastreável, sem usar rótulos corrompidos como chaves. |
| Caractere NUL em `races.time` | Registrar anomalia de origem; não tratá-la como horário válido nem reescrever o CSV original. |
| Coluna extra final no cabeçalho de `constructors.csv` | Auditar coluna vazia/sem nome; não atribuir significado analítico nem alterar `raw`. |
| Participações `(raceId, driverId)` repetidas entre **1950–1978** | Separar do bloqueio de participação moderna; contexto histórico precisa ser investigado antes de qualquer deduplicação. |
| Retiradas/resultados com `laps = 0` | Manter quando os demais critérios forem satisfeitos; não usar filtro `laps > 0`. Grid positivo isolado não comprova largada. |
| Calendário com 21 eventos em 2018 e nenhum resultado correspondente | Declarar cobertura separada; não inferir resultados ausentes como zero nem apresentar 2018 como temporada analisável. |

Grids repetidos entre pilotos distintos não constituem, por si, PK duplicada nem participação `(raceId, driverId)` duplicada. As pendências de P17 podem impedir sucesso de `--strict` sem impedir a execução padrão. Qualquer correção futura deve registrar evidência da fonte, IDs afetados, regra adotada e impacto nas contagens.

## 7. CLI e artefatos de auditoria

Na raiz do projeto, com Python 3.12 e dependências disponíveis no ambiente local:

```powershell
python -m src.audit_data
python -m src.audit_data --start-year 2004 --end-year 2017 --output-dir outputs --raw-dir data/raw
python -m src.audit_data --strict --start-year 2004 --end-year 2017 --output-dir outputs --raw-dir data/raw
```

Interface: `python -m src.audit_data [--strict] [--start-year 2004 --end-year 2017] [--output-dir outputs] [--raw-dir data/raw]`.

Artefatos no diretório de saída:

- **`data_quality.json`**: representação estruturada para inspeção e automação.
- **`data_quality.md`**: relatório legível dos mesmos achados e do contexto da execução.

Os relatórios registram **hashes dos arquivos de entrada**, ambiente de execução, cobertura de calendário versus resultados, período escolhido, ausências, duplicações, integridade de FKs, valores inválidos, exclusões e proveniência. Hashes permitem identificar o conteúdo auditado; não provam autoria, licença ou correção histórica.

### Política de saída

- A CLI padrão grava relatórios e **não falha apenas por warnings**; falha quando há achados bloqueantes.
- A CLI `--strict` **grava os relatórios antes de retornar erro por warnings ou errors**. Um retorno não zero não deve fazer perder o diagnóstico que motivou a recusa.
- **`python -m src.run_analysis` também gera os relatórios de auditoria padrão** em `outputs/`, além dos artefatos analíticos usuais. A auditoria independente não requer executar modelos ou abrir o dashboard.
- Auditoria não autoriza reparos, remoção de dados ou publicação. Os CSVs de `data/raw/` permanecem intactos.

Os comandos documentados são instruções de uso, não alegação de execução nesta etapa. A validação posterior deve executar `python -m pytest -q` e, após mudanças de pipeline/métricas/modelos, `python -m src.run_analysis`; nenhuma aprovação futura ou teste de navegador é presumido aqui.

## Proveniência e direitos de uso

A distribuição dos cinco CSVs locais foi identificada como [Formula 1 Race Data, publicada por cjgdev no Kaggle](https://www.kaggle.com/datasets/cjgdev/formula-1-race-data-19502017). A página do dataset declara CC BY-NC-SA 4.0. Na auditoria da Fase 7C, os cinco CSVs locais coincidiram byte a byte com os membros correspondentes do ZIP conferido, com SHA-256 coincidente. Essa verificação estabelece correspondência com a cópia do ZIP comparada; não identifica a fonte primária anterior, não comprova a titularidade dos direitos incorporados e não autoriza redistribuição pública.

| Campo de proveniência | Estado |
|---|---|
| Distribuição / publicador | Identificada: Formula 1 Race Data / cjgdev (Kaggle) |
| URL da distribuição | Registrada na página do dataset, acima |
| Fonte primária anterior à distribuição Kaggle | **PENDENTE** |
| Versão / release / snapshot exato | **PENDENTE** |
| Data e método de extração | **PENDENTE** |
| Licença declarada na página do dataset | CC BY-NC-SA 4.0; declaração registrada, sem confirmação independente da cadeia de direitos |
| Licença aplicável a todas as fontes/componentes | **PENDENTE** |
| Permissão e condições de redistribuição pública | **PENDENTE** |

A correspondência dos bytes e dos hashes foi registrada como evidência da auditoria da Fase 7C; o dashboard não a verifica automaticamente. Data de execução da auditoria ou timestamp de arquivo não substituem a data de extração da fonte. A cláusula NC e as obrigações SA, a proveniência anterior e os direitos de terceiros precisam ser avaliados para o uso pretendido.

**Não publicar os CSVs enquanto essa pendência existir. A derivação não libera automaticamente `outputs/`: tabelas, relatórios, métricas e gráficos também exigem avaliação dos direitos e condições aplicáveis antes de publicação.** O `.gitignore` não constitui autorização. Não remover arquivos de `raw` como substituto dessa verificação, não atribuir licença por inferência e não realizar deploy ou publicação sem autorização explícita.
