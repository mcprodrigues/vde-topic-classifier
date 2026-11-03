# vde-topic-classifier

Sistema de classificação de questões por tópico para o case "Classificação de tópicos" do Método VDE.
Para cada uma das 659 questões, decide `publicar`, `revisar` ou `fora` e devolve uma confiança calibrada.
Nenhum modelo de linguagem participa da decisão: o comando de publicação lê só os anexos e os artefatos versionados.

| Entregável | Onde |
|---|---|
| `publicacao.csv` (659 linhas) | [`outputs/publicacao.csv`](outputs/publicacao.csv) |
| `previsao.txt` | [`outputs/previsao.txt`](outputs/previsao.txt) |
| Relatório (até 4 páginas) | [`relatorio/relatorio.md`](relatorio/relatorio.md) |
| Memo para o professor | [`relatorio/memo_professor.md`](relatorio/memo_professor.md) |
| Figuras e métricas | [`outputs/confiabilidade.png`](outputs/confiabilidade.png), [`outputs/evaluation.json`](outputs/evaluation.json), [`outputs/ablations.csv`](outputs/ablations.csv) |

## Resultado em uma tabela

| Medida | Valor | De onde vem |
|---|---|---|
| Acerto do nó publicado (CV fora da amostra) | 0,655 (desvio entre repetições 0,009) | 80 rótulos de dev, 10 folds x 20 repetições |
| Acerto no cofre (20 rótulos nunca vistos) | 0,80 | `vde evaluate --open-lockbox` |
| AUC da confiança | 0,73 | OOF |
| Precisão nos 10% / 20% mais confiantes | 1,00 / 0,94 | OOF |
| Limiar certificável (P(precisão >= 0,95) >= 0,80) | nenhum | seriam necessários 32 acertos seguidos sem erro |
| Decisão nas 559 avaliadas | 0 publicar, 548 revisar, 11 fora | `outputs/publicacao.csv` |

Com 100 rótulos, o sistema não consegue **certificar** a régua de 95% em nenhum ponto da curva. Por isso, e por
decisão explícita (ADR-6), nada é publicado: as 548 questões vão para a fila do professor ordenada por confiança
(115 delas com confiança >= 0,90). O relatório mostra a curva, por que o teto existe e como destravar a publicação.

## Como rodar

Requer Python 3.11 ou superior. Não precisa de GPU nem de chave de API.

```bash
pip install -e .            # ou: pip install -e ".[dev]" para rodar os testes
vde publish                 # regenera outputs/publicacao.csv   (também: python -m vde publish)
vde evaluate --open-lockbox # métricas, figuras, relatório de mudança de distribuição, previsao.txt
pytest                      # inclui o teste "sem LLM, sem rede, CSV idêntico"
```

Tempos medidos num contêiner com 2 vCPUs:

| Comando | O que faz | Tempo | Usa LLM? |
|---|---|---|---|
| `vde publish` | lê artefatos e escreve `publicacao.csv` | 5 s | Não |
| `vde evaluate` | métricas a partir do OOF salvo, cofre, mudança de distribuição, previsão | 13 s | Não |
| `vde evaluate --ablations` | as 17 ablações (3 repetições cada) | 7 a 8 min | Não |
| `vde build model` | CV repetida 20x no dev e ajuste final nos 100 rótulos | 3,5 min | Não |
| `vde build canonical_tree`, `subject_map` | regras determinísticas | < 1 s cada | Não |
| `vde build embeddings` | LSA local sobre 3.891 textos | 4 s | Não |
| `vde build law_index`, `code_atlas`, `topic_cards` | valida o CSV revisado e grava a proveniência | < 1 s cada | Não |
| `vde build <artefato> --source llm` | pede uma **proposta** ao LLM em `artifacts/proposals/` para revisão humana | 5 a 25 min (estimado) | Sim, offline |

## A garantia "nenhum LLM na decisão"

* O caminho de `vde publish` chama apenas `load()` dos artefatos. O cliente de LLM é importado dentro de
  `vde.knowledge.base.propose_with_llm`, que só é chamado por `build(source="llm")`.
* `tests/test_no_llm.py` remove as chaves de API, bloqueia `socket.connect`, faz `import anthropic` falhar,
  roda `publish` e compara o SHA-256 do CSV gerado com o versionado. Outro teste confere, por análise
  estática, que nenhum módulo importa um cliente de LLM no topo.
* `vde publish` confere `artifacts/MANIFEST.json` antes de rodar e para se algum artefato mudou sem rebuild.

## Arquitetura

```
anexos/ -> data/ (limpeza HTML, segmentos) -> knowledge/ (artefatos) -> channels/ (evidências)
        -> aggregate.py (pesos por canal + modelo de acerto) -> tree.py (escolha do nó)
        -> calibrate.py (limiar por posterior Beta) -> decide.py (publicar / revisar / fora + motivo)
```

| Peça | Arquivo | Artefato | LLM offline? |
|---|---|---|---|
| Árvore canônica: 20 nós-sombra mapeados, lixeiras marcadas, regra de ramos gêmeos | `vde/knowledge/canonical_tree.py` | `canonical_tree.csv`, `branch_twins.csv` | Não |
| Índice de leis (60 leis e códigos, com apelidos) | `vde/knowledge/law_index.py` | `law_index.csv` | Proposto por LLM, revisão humana pendente |
| Atlas dos códigos: faixas de artigos do CP e do CPP para temas | `vde/knowledge/code_atlas.py` | `code_atlas.csv` | Proposto por LLM, revisão humana pendente |
| Cartões de tema para os 351 temas publicáveis | `vde/knowledge/topic_cards.py` | `topic_cards.csv` | Proposto por LLM, revisão humana pendente |
| Mapa do `assunto_origem` (evidência fraca) | `vde/knowledge/subject_map.py` | `subject_map.csv` | Não |
| Embeddings (LSA local, cache) | `vde/knowledge/embeddings.py` | `embeddings.npz` | Não |
| Canais: lei especial, artigo de código, BM25, protótipos, kNN, assunto, duplicatas | `vde/channels/` | (nenhum) | Não |
| Agregação e modelo de acerto | `vde/aggregate.py`, `vde/engine.py` | `model.json` (números legíveis, sem pickle) | Não |
| Cofre de 20 rótulos | `vde/splits.py` | `splits.json` | Não |

Mudanças em relação à estrutura proposta: `engine.py` concentra o ajuste por fold (para a CV não vazar rótulos);
`resources.py` carrega tudo uma vez; `splits.py`, `shift.py`, `forecast.py`, `report.py` e `pipeline.py`
separam cofre, mudança de distribuição, previsão, relatório e treino/publicação. Os comandos ficaram em
inglês (`publish`, `evaluate`, `build`), como todo o código.

## Origem de cada artefato e decisões manuais

Cada artefato tem um `*.prov.json` ao lado (hash, data, commit, como foi produzido, se usou LLM, se viu questões).
As decisões manuais estão em [`artifacts/manual_decisions.csv`](artifacts/manual_decisions.csv). As principais:

* **MD01 a MD03:** índice de leis, atlas e cartões foram propostos por um LLM nesta sessão de autoria (não havia
  chave de API no ambiente). O LLM recebeu só a árvore de temas e o conhecimento das leis, nunca as questões avaliadas.
  **A revisão humana desses três arquivos ainda está pendente** e é o primeiro passo antes de qualquer publicação.
* **MD04:** os 20 nós sem descrição e sem filhos ("Teoria da Pena (Parte 1)", "Penal > Lei Maria da Penha"...) são
  mapeados para o gêmeo canônico, detectado automaticamente por similaridade de título.
* **MD05:** Lei 9.099 vai para Legislação Especial > Juizado Especial Criminal, e a massa de Processo Penal >
  Procedimento Sumaríssimo é somada a esse ramo (`branch_twins.csv`): no dev, a equipe usa 4701 como central
  3 vezes e o sumaríssimo só como extra.
* **MD07 a MD09:** são `fora` por regra: o procedimento dos crimes funcionais (CPP 513 a 518), os crimes contra a
  ordem econômica (tema inativo) e o direito penal e processual militar.
* **MD10 a MD13:** ρ = 0,85 fixado a priori; nível de 80% no critério do limiar; cofre de 20; nunca publicar no nível 0.

## Fluxo de desenvolvimento

`feature/* -> develop -> main`, sempre com merge `--no-ff`. Branches: `feature/data-layer`,
`feature/knowledge-artifacts`, `feature/evidence-channels`, `feature/aggregation-decision`,
`feature/evaluation-outputs`, `feature/docs`. O release `v0.1.0` está em `main`.
