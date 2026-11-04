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
| Figuras e métricas | [`outputs/confiabilidade.png`](outputs/confiabilidade.png), [`outputs/evaluation.json`](outputs/evaluation.json), [`outputs/ablations.csv`](outputs/ablations.csv), [`outputs/ablations_20rep.csv`](outputs/ablations_20rep.csv) |

## Resultado em uma tabela

Todas as medidas fora da amostra vêm de CV de 10 folds x 20 repetições sobre os 100 rótulos, calculadas **uma
repetição por vez** (cada repetição é um modelo implantável), nunca sobre a confiança média entre repetições.

| Medida | Valor | De onde vem |
|---|---|---|
| Decisão nas 559 avaliadas | <!-- claim:published -->76<!-- /claim --> publicar, 472 revisar, 11 fora | `outputs/publicacao.csv` |
| Cobertura | <!-- claim:coverage -->0,14<!-- /claim --> (27 de 259 no banco, 49 de 300 na importação nova) | `outputs/publicacao.csv` |
| Precisão prevista nas publicadas | <!-- claim:precision -->0,93<!-- /claim --> (intervalo de 90%: 0,78 a 1,00) | `outputs/previsao.txt` |
| Chance de a precisão ficar abaixo de 0,95 | <!-- claim:below -->49%<!-- /claim --> | `evaluation.json`, `forecast` |
| Precisão aninhada da regra de publicação | <!-- claim:nested -->1,00<!-- /claim --> (7,8 publicadas por repetição; limite inferior de 90%: 0,75) | `evaluation.json`, `cv.threshold_cv` |
| Acerto do nó publicado (todas as questões) | 0,676 (desvio entre repetições 0,010) | `evaluation.json`, `cv` |
| AUC da confiança / ECE | 0,79 / 0,049 | `evaluation.json`, `cv` |
| Precisão nos 10% / 20% mais confiantes | <!-- claim:p10 -->0,98<!-- /claim --> / 0,96 (10% das repetições abaixo de 0,90 / 0,90) | `evaluation.json`, `cv` |

**Regra de publicação:** publicar no tema de nível 1 quando a confiança calibrada for >= 0,95. A regra foi fixada
antes de olhar a curva. Escolher o limiar olhando a curva daria precisão aninhada de 0,94, por causa do otimismo
da seleção. A previsão (0,93) fica abaixo da precisão aninhada (1,00) porque conta cada questão rotulada uma vez só,
usa prior de Jeffreys e alarga o intervalo da importação nova, que não tem nenhum rótulo.

## Como rodar

Requer Python 3.11 ou superior. Não precisa de GPU nem de chave de API.

```bash
pip install -e .            # ou: pip install -e ".[dev]" para rodar os testes
vde publish                 # regenera outputs/publicacao.csv   (também: python -m vde publish)
vde evaluate                # métricas, figuras, relatório de mudança de distribuição, previsao.txt
pytest                      # inclui o teste "sem LLM, sem rede, CSV idêntico"
```

Tempos medidos numa máquina com 16 núcleos:

| Comando | O que faz | Tempo | Usa LLM? |
|---|---|---|---|
| `vde publish` | lê artefatos e escreve `publicacao.csv` | 5 s | Não |
| `vde evaluate` | métricas a partir do OOF salvo, mudança de distribuição, previsão | 10 s | Não |
| `vde evaluate --ablations` | as 19 ablações (3 repetições cada) | ~15 min | Não |
| `vde build model` | CV repetida 20x nos 100 rótulos e ajuste final | 4,5 min | Não |
| `vde build all` | todos os artefatos do zero, na ordem de dependência | ~5 min | Não |
| `vde build canonical_tree`, `branch_twins`, `subject_map` | regras determinísticas | < 1 s cada | Não |
| `vde build embeddings` | LSA local sobre 3.891 textos (guarda vetores e o modelo LSA) | 8 s | Não |
| `vde build law_index`, `code_atlas`, `topic_cards` | valida o CSV revisado e grava a proveniência | < 1 s cada | Não |
| `vde build <artefato> --source llm` | pede uma **proposta** ao LLM em `artifacts/proposals/` para revisão humana | não medido | Sim, offline |

Questões novas: `vde publish` para se algum anexo mudou desde o último build (o `MANIFEST.json` guarda o hash dos
anexos). `vde publish --skip-manifest` classifica as questões que não estão no cache projetando-as no espaço LSA
salvo, sem reajustar nada.

## A garantia "nenhum LLM na decisão"

* O caminho de `vde publish` chama apenas `load()` dos artefatos. O cliente de LLM é importado dentro de
  `vde.knowledge.base.propose_with_llm`, que só é chamado por `build(source="llm")`.
* `tests/test_no_llm.py` remove as chaves de API, bloqueia `socket.connect`, faz `import anthropic` falhar,
  roda `publish` e compara o SHA-256 do CSV gerado com o versionado. Outro teste confere, por análise
  estática, que nenhum módulo importa um cliente de LLM no topo.
* `vde publish` confere `artifacts/MANIFEST.json` (artefatos e anexos) antes de rodar.
* Um clone limpo com `vde build all && vde publish` reproduz `publicacao.csv` byte a byte.

## Arquitetura

```
anexos/ -> data/ (limpeza HTML, segmentos) -> knowledge/ (artefatos) -> channels/ (evidências)
        -> aggregate.py (pesos por canal + modelo de acerto) -> tree.py (escolha do nó)
        -> calibrate.py (regra de limiar) -> decide.py (publicar / revisar / fora + motivo)
```

| Peça | Arquivo | Artefato | LLM offline? |
|---|---|---|---|
| Árvore canônica: 20 nós-sombra mapeados, lixeiras marcadas | `vde/knowledge/canonical_tree.py` | `canonical_tree.csv` | Não |
| Regra de ramos gêmeos, aprendida dos rótulos e reajustada em cada fold | `vde/knowledge/canonical_tree.py` (`learn_twins`) | `branch_twins.csv` (registro) | Não |
| Índice de leis (60 leis e códigos, com apelidos) | `vde/knowledge/law_index.py` | `law_index.csv` | Proposto por LLM, revisão humana pendente |
| Atlas dos códigos: faixas de artigos do CP e do CPP para temas | `vde/knowledge/code_atlas.py` | `code_atlas.csv` | Proposto por LLM, revisão humana pendente |
| Cartões de tema para os 351 temas publicáveis | `vde/knowledge/topic_cards.py` | `topic_cards.csv` | Proposto por LLM, revisão humana pendente |
| Mapa do `assunto_origem` (evidência fraca) | `vde/knowledge/subject_map.py` | `subject_map.csv` | Não |
| Embeddings (LSA local, vetores e modelo em cache) | `vde/knowledge/embeddings.py` | `embeddings.npz` | Não |
| Canais: lei especial, artigo de código, BM25, protótipos, kNN, assunto, duplicatas | `vde/channels/` | (nenhum) | Não |
| Agregação e modelo de acerto | `vde/aggregate.py`, `vde/engine.py` | `model.json` (números legíveis, sem pickle) | Não |
| Divisão dos rótulos | `vde/splits.py` | `splits.json` | Não |

## Origem de cada artefato e decisões manuais

Cada artefato tem um `*.prov.json` ao lado (hash, data, commit, como foi produzido, se usou LLM). As decisões
manuais estão em [`artifacts/manual_decisions.csv`](artifacts/manual_decisions.csv). As principais:

* **MD01 a MD03:** índice de leis, atlas e cartões foram propostos por um LLM numa sessão de autoria. O pedido
  incluía só a árvore de temas e o conhecimento das leis, mas a sessão tinha acesso ao repositório, questões
  incluídas: **não dá para verificar que o LLM nunca viu as questões**, e a proveniência diz isso.
  **A revisão humana desses três arquivos ainda está pendente.**
* **MD04:** os 20 nós sem descrição e sem filhos ("Teoria da Pena (Parte 1)", "Penal > Lei Maria da Penha"...) são
  mapeados para o gêmeo canônico, detectado automaticamente por similaridade de título.
* **MD05:** Lei 9.099 vai para Legislação Especial > Juizado Especial Criminal. O Procedimento Sumaríssimo de
  Processo Penal é somado a esse ramo pela regra de gêmeos, que agora é aprendida dos rótulos: o sumaríssimo
  aparece 4 vezes como extra ao lado de um central do Juizado e nunca como central.
* **MD07 a MD09:** são `fora` por regra: o procedimento dos crimes funcionais (CPP 513 a 518), os crimes contra a
  ordem econômica (tema inativo) e o direito penal e processual militar.
* **MD10, MD13:** ρ = 0,85 (só vale com `node_mode = "branch"`); nunca publicar no nível 0.
* **MD14:** publicar no tema de nível 1. Venceu a comparação de 20 repetições (precisão nos 10%: 0,985 contra
  0,970). O custo é que o aluno vê a questão no tema, não no subtema, e 30 das 76 publicadas caem em temas
  marcados como lixeira (sempre com confiança >= 0,95).
* **MD15:** regra de limiar calibrada (publicar quando P(acerto) >= 0,95) em vez de buscar na curva.
* **MD16:** o cofre de 20 questões da v0.1 foi inspecionado na análise de erro (Q0252, Q0303) e foi juntado aos
  rótulos. Não há mais teste cego: a garantia vem da CV aninhada.

## Fluxo de desenvolvimento

`feature/* -> develop -> main`, sempre com merge `--no-ff`. A v0.2 está em `develop`: `feature/forecast-format`,
`feature/nested-cv`, `feature/publish-slice`, `feature/shift-forecast`, `feature/production-hardening`,
`feature/docs-v0.2`. O release `v0.1.0` continua em `main`.
