# Proveniência e licenças — F1 Data Lab

**Revisão documental:** 2026-10-23. **Estado:** inventário de evidências e pendências; não é parecer jurídico nem autorização de redistribuição.

## Regra de publicação

Atribuição, URL, hash, transformação, estatística agregada ou presença em um repositório público não demonstram, por si só, autorização para redistribuir dados. Cada camada deve ser avaliada com seus termos e sua origem documentada. Enquanto houver pendência, mantenha os arquivos locais e não os inclua em uma versão pública.

A aplicação é local e lê arquivos do projeto. O carregamento não deve ser interpretado como permissão para compartilhar o conteúdo, nem como autorização para baixar ou republicar dados de terceiros.

## Inventário por origem e categoria

| Origem / categoria | Arquivos e uso pretendido | Fonte e termos conhecidos | Permissões confirmadas | Restrições e pendências |
|---|---|---|---|---|
| **CSV históricos locais** | `data/raw/results.csv`, `races.csv`, `drivers.csv`, `constructors.csv`, `status.csv`; entrada do pipeline, análise e dashboard. | Distribuição identificada: [Formula 1 Race Data, por cjgdev no Kaggle](https://www.kaggle.com/datasets/cjgdev/formula-1-race-data-19502017). A página declara CC BY-NC-SA 4.0. Na auditoria da Fase 7C, os cinco arquivos locais coincidiram byte a byte com os membros correspondentes do ZIP conferido e os SHA-256 coincidiram. Isso identifica a distribuição comparada, não a fonte primária anterior nem a titularidade dos direitos nela incorporados. | A licença CC BY-NC-SA 4.0 é registrada como declarada na página do dataset; não foi confirmada autorização de redistribuição pública para este projeto. | Fonte primária anterior ao Kaggle, versão/release exato, método/data de extração e direitos de componentes anteriores permanecem **PENDENTES**. Avaliar as condições NC/SA e os direitos aplicáveis ao uso pretendido; não tratar a correspondência de bytes ou hashes como autorização. Não publicar os CSVs enquanto essa avaliação estiver pendente. |
| **NOAA/NCEI — respostas e observações locais** | `data/climate_phase3b_20261022/noaa_raw/`, `noaa_observations.csv`, `noaa_query_log.csv`, `noaa_stations.csv`; observações de estações selecionadas para eventos de 2013–2014. | [NOAA ISD — página do produto](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database). A documentação de Fase 3A do projeto registrou que a página do produto não especifica uma licença individual. O [AWS Open Data Registry para NOAA ISD](https://registry.opendata.aws/noaa-isd/) informa que os dados NOAA são abertos para uso público, solicita atribuição quando dados inalterados são usados/compartilhados e orienta que dados modificados não sejam apresentados como dados NOAA originais e inalterados. | O registro AWS documenta condições de uso aberto e pedido de crédito para dado inalterado. Isso não confirma, para este projeto, os direitos de todo componente incorporado, snapshot, agregado ou derivado. | Manter atribuição e identificar transformações; verificar se há componentes de terceiros e condições específicas. A licença aplicável a cada observação e aos arquivos locais agregados não está determinada por esta revisão. Não tratar o registro AWS como uma licença específica que regularize todas as camadas. |
| **NOAA — derivados e indicadores QC** | `data/climate_phase3c_20261022/noaa_observations_quality.csv`, `outputs/climate_phase3c_20261022/quality_summary_by_event.csv`, `quality_summary_common_report_types.csv` e textos de encerramento; apresentação de valores, flags e resumos no dashboard. | São derivados das observações locais e da interpretação documentada das flags NOAA ISD. A política técnica consultada descreve QC; semântica de QC não é uma licença de redistribuição. | Nenhuma autorização específica para redistribuir estes arquivos derivados foi confirmada. | A transformação não resolve automaticamente os direitos da fonte. Revalidar atribuição, identificação de transformações e escopo de uso antes de publicar CSVs, imagens, tabelas, páginas ou outputs. A qualidade da estação não garante representatividade do circuito. |
| **Jolpica — dados/metadata de F1** | Referências Jolpica registradas em `data/climate_phase3b_20261022/events.csv` para metadata de circuito; referência usada na investigação de reconciliação histórica. Os CSVs históricos não têm proveniência que prove terem sido adquiridos de Jolpica. | [Termos de dados da Jolpica](https://github.com/jolpica/jolpica-f1/blob/main/TERMS.md) indicam uso não comercial e CC BY-NC-SA 4.0 para os dados; o [repositório Jolpica](https://github.com/jolpica/jolpica-f1) identifica Apache-2.0 para o software do projeto. São escopos distintos. | A licença/termos do software podem reger o software Jolpica conforme o repositório. Isso **não** prova que os CSVs locais do F1 Data Lab sejam dados Jolpica nem libera a sua redistribuição. | A cláusula NC pode ser incompatível com um portfólio público/promocional; não assumir elegibilidade. Confirmar atribuição, share-alike, uso pretendido e cadeia de origem antes de reutilizar qualquer dado Jolpica. Apache-2.0 do software não substitui CC BY-NC-SA dos dados. |
| **Formula1.com / F1 e FIA — conteúdo consultado** | URLs e descrições narrativas registradas em `data/climate_phase3b_20261022/race_evidence.csv`; páginas públicas usadas como fonte documental de evidência de evento. A política de referência de reconciliação também registra consulta a páginas de resultados F1. | [Avisos legais da Formula 1](https://www.formula1.com/en/information/legal-notices.7egvZU48hzrypubGBNcQKt) e [termos de uso da Formula One](https://corp.formula1.com/terms-of-use/). As páginas afirmam direitos autorais/propriedade intelectual sobre conteúdos do site e não fornecem uma licença geral para redistribuição de bases de dados. | A consulta e a citação de URLs como referência não equivalem a licença de reprodução/redistribuição. Nenhuma autorização específica foi confirmada para este projeto. | Não transcrever ou republicar tabelas, imagens ou bases completas sem permissão. A evidência narrativa local identifica fonte, classe e limitações, mas não se converte em dado instrumental nem comprova direito de republicação do texto ou da fonte. |
| **Código de autoria do projeto** | `app.py`, `src/`, `tests/` e documentação produzida para este projeto. Uso pretendido: demonstrar método e organização do código. | Não foi encontrado arquivo `LICENSE`/`COPYING`/`NOTICE` do projeto na árvore inspecionada. A presença de `CLAUDE.md` e de documentação não define licença de software. | Nenhuma licença aberta foi escolhida ou concedida nesta etapa. A autoria e eventual contribuição de terceiros devem ser revistas antes de declarar o código integralmente próprio. | **Não criar ou inferir uma licença.** Sem escolha expressa e revisão de autoria, a publicação de código não deve prometer direitos de reutilização. Um futuro repositório pode excluir dados, mas ainda precisa decidir explicitamente se e sob quais termos o código será publicado. |
| **Dependências e materiais de terceiros** | Pacotes declarados em `requirements.txt` (`pandas`, `numpy`, `matplotlib`, `scikit-learn`, `streamlit`, `plotly`, `pytest`) e quaisquer materiais que venham a ser incluídos em pacote. | Cada projeto de dependência mantém seus próprios termos e licenças; os intervalos de versão no `requirements.txt` não são inventário de licenças nem lockfile. | O projeto declara dependências; esta revisão não confirma permissão para redistribuir cópias empacotadas das dependências. | Se empacotar ou redistribuir um aplicativo, revisar a licença/avisos da versão efetivamente instalada de cada dependência e materiais incluídos. Não incluir ambientes virtuais ou caches no repositório. |
| **Outputs, imagens e resultados derivados** | `outputs/`: métricas, CSVs, relatórios, figuras PNG, sumários climáticos e artefatos de reconciliação. | Combinam ou transformam os dados descritos acima; alguns são cópias/derivações, outros agregações. O manifesto lista integridade e caminhos, não direitos. | Nenhuma autorização geral para redistribuir outputs foi confirmada. | Não assumir que estatísticas agregadas, tabelas, gráficos, relatórios, capturas de tela, hashes ou dumps são publicáveis. Verificar cada artefato e os direitos da fonte antes de incluí-lo em GitHub, LinkedIn ou outro local público. |
| **Notebook histórico e documentação local** | `notebooks/`, `GUIA_CLAUDE_CODE.md`, configurações `.claude/` e outros documentos locais. | Notebook histórico inclui caminhos de Colab/Google Drive e cálculos antigos; arquivos de configuração descrevem o ambiente de desenvolvimento. | Não há necessidade de redistribuí-los para explicar a implementação atual. | Excluir por padrão de uma futura versão pública ou revisar manualmente caminhos, saídas embutidas, termos, tokens de exemplo e conteúdo histórico. `.gitignore` não remove arquivos já rastreados. |
| **Fixtures sintéticas dos testes** | `tests/conftest.py` e fixtures associadas, com dados sintéticos para validações e testes de UI. | A fixture compartilhada identifica-se como sintética e usa rótulos inventados; outras fixtures devem ser revisadas antes de publicar. | São candidatas a permitir testes em uma cópia sem dados históricos, após confirmar que nenhuma fixture contém excerto real. | Não tratar o rótulo “sintético” de uma fixture como validação de todo o diretório de testes. A aplicação principal ainda requer arquivos locais. |

## Verificações posteriores — 28/10/2026

Foi realizada uma validação local independente no Windows, utilizando Python 3.12.

- Repositório privado clonado do GitHub.
- Dependências instaladas em um ambiente virtual independente.
- Integridade de 12 arquivos CSV e 10 imagens PNG conferida por SHA-256.
- Suíte automatizada: 331 testes aprovados, sem falhas ou testes ignorados.
- Seis áreas do dashboard Streamlit verificadas manualmente no navegador.
- Nenhum arquivo de data/ ou outputs/ identificado entre os arquivos rastreados.
- Histórico Git acessível com um único commit inicial: a2f56e9.
- Buscas iniciais por credenciais, caminhos pessoais e endereços internos sem correspondências.

Essas verificações demonstram reprodutibilidade local nas condições testadas,
mas não constituem auditoria exaustiva de segurança nem autorização para
redistribuir datasets, gráficos ou resultados derivados.

As pendências de proveniência e licenciamento permanecem aplicáveis.

## Política para um futuro repositório

Até resolver as pendências acima, a configuração do projeto ignora `data/`, `outputs/`, `notebooks/`, ambientes virtuais, caches, segredos e configurações pessoais para reduzir inclusões acidentais. O `.gitignore` não altera o índice do Git e não retira arquivos que já tenham sido rastreados. O estado de rastreamento deve ser verificado antes de qualquer publicação; nesta revisão a CLI Git não estava disponível, então não foi possível confirmar esse estado.

Uma publicação futura apenas do código exigiria, no mínimo:

1. confirmação de autoria e escolha explícita de licença para o código (não decidida aqui);
2. exclusão dos CSVs históricos, snapshots, derivados, outputs, notebooks e arquivos de configuração local enquanto não liberados;
3. revisão do histórico Git e da árvore já rastreada; qualquer limpeza de itens rastreados requer aprovação própria e uso de comandos apropriados, não apenas `.gitignore`;
4. documentação de como preparar dados somente a partir de fonte e sob condições de uso autorizadas, sem redistribuí-los;
5. revisão de dependências e de todas as capturas, tabelas e números que possam reproduzir conteúdo de terceiros.

Se a ausência dos datasets impedir a demonstração, um modo demonstrativo com dados inventados e claramente rotulados pode ser considerado em uma fase futura. Não está implementado neste projeto.

## Fontes oficiais consultadas

Consulta documental em **2026-10-23**:

- NOAA, [AWS Open Data Registry — NOAA Integrated Surface Database](https://registry.opendata.aws/noaa-isd/) — condições de uso/atribuição descritas pelo registro.
- NOAA/NCEI, [Integrated Surface Database](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) — página do produto; a documentação local de Fase 3A registra ausência de licença específica indicada na página.
- Jolpica, [TERMS.md](https://github.com/jolpica/jolpica-f1/blob/main/TERMS.md) e [repositório oficial](https://github.com/jolpica/jolpica-f1) — termos de dados e licença do software, tratados separadamente.
- Formula 1, [Legal Notices](https://www.formula1.com/en/information/legal-notices.7egvZU48hzrypubGBNcQKt) e [Terms of Use](https://corp.formula1.com/terms-of-use/) — avisos legais e restrições de uso do site.

As páginas podem mudar. A data registra quando foram consultadas para esta revisão; não transforma a consulta em autorização, parecer jurídico ou prova da proveniência dos arquivos locais.
