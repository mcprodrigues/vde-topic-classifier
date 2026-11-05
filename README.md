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
| Figuras e métricas | [`outputs/confiabilidade.png`](outputs/confiabilidade.png), [`outputs/evaluation.json`](outputs/evaluation.json), [`outputs/ablations.csv`](outputs/ablations.csv), [`outputs/ablations_20rep.csv`](outputs/ablations_20rep.csv), [`outputs/embeddings_neural.csv`](outputs/embeddings_neural.csv) |

## Resultado em uma tabela

Todas as medidas fora da amostra vêm de CV de 10 folds x 20 repetições sobre os 100 rótulos, calculadas **uma
repetição por vez** (cada repetição é um modelo implantável), nunca sobre a confiança média entre repetições.

| Medida | Valor | De onde vem |
|---|---|---|
| Decisão nas 559 avaliadas | <!-- claim:published -->75<!-- /claim --> publicar, 473 revisar, 11 fora | `outputs/publicacao.csv` |
| Cobertura | <!-- claim:coverage -->0,13<!-- /claim --> (27 de 259 no banco, 48 de 300 na importação nova) | `outputs/publicacao.csv` |
| Precisão prevista nas publicadas | <!-- claim:precision -->0,95<!-- /claim --> (intervalo de 90%: 0,87 a 1,00) | `outputs/previsao.txt` |
| Chance de a precisão ficar abaixo de 0,95 | <!-- claim:below -->40%<!-- /claim --> | `evaluation.json`, `forecast` |
| Precisão aninhada da regra de publicação | <!-- claim:nested -->1,00<!-- /claim --> (7,9 publicadas por repetição; limite inferior de 90%: 0,75) | `evaluation.json`, `cv.threshold_cv` |
| Acerto do nó publicado (todas as questões) | 0,676 (desvio entre repetições 0,010) | `evaluation.json`, `cv` |
| AUC da confiança / ECE | 0,79 / 0,046 | `evaluation.json`, `cv` |
| Precisão nos 10% / 20% mais confiantes | <!-- claim:p10 -->0,98<!-- /claim --> / 0,96 (10% das repetições abaixo de 0,90 / 0,90) | `evaluation.json`, `cv` |

**Regra de publicação:** publicar no tema de nível 1 quando a confiança calibrada for >= 0,95, e só se houver
evidência de texto (termo dos cartões, lei, artigo ou assunto reconhecido) e o ramo escolhido tiver massa >= 0,30.
A regra foi fixada antes de olhar a curva. Escolher o limiar olhando a curva daria precisão aninhada de 0,94, por
causa do otimismo da seleção. A previsão (0,95) parte da confiança calibrada média das publicadas (0,97), com um
desvio de calibração igual ao maior entre o ECE da CV (0,046) e a incerteza dos rótulos dentro da faixa publicada
(0,056), maior na importação nova, que não tem nenhum rótulo, e ruído binomial. A checagem com os rótulos
(18 questões distintas, todas certas, tamanho efetivo 10,9) dá 0,96 (0,84 a 1,00) no banco e confirma o número,
mas é larga demais para ser a previsão.

## Como rodar

### 1. Requisitos

Python 3.11 ou superior, `git` e cerca de 500 MB de disco para o ambiente virtual. Não precisa de GPU, de rede
(depois da instalação) nem de chave de API.

### 2. Instalar

```bash
git clone <url-do-repositório> vde-topic-classifier
cd vde-topic-classifier
python3 -m venv .venv
source .venv/bin/activate                          # Windows: .venv\Scripts\activate
pip install -r requirements.lock                   # versões exatas que geraram os artefatos
pip install -e . --no-deps                         # instala o comando `vde`
```

`requirements.lock` fixa as versões usadas no treino (as mesmas que o `model.json` registra em `versions`).
`pip install -e ".[dev]"` também funciona, mas resolve versões novas das bibliotecas.

### 3. Regenerar a entrega

```bash
vde publish                    # escreve outputs/publicacao.csv   (também: python -m vde publish)
git status outputs/            # nada muda: o CSV gerado é idêntico ao versionado
```

`vde publish` lê só os anexos (`anexos/`) e os artefatos versionados (`artifacts/`), não chama nenhum modelo de
linguagem e termina com um resumo das decisões:

```
publicacao.csv: 659 linhas {'publicar': 167, 'revisar': 473, 'fora': 19}
```

As 100 questões da folha-verdade saem com o rótulo da equipe (92 `publicar`, 8 `fora`) e não entram na
avaliação. Nas 559 restantes ficam 75 `publicar`, 473 `revisar` e 11 `fora`. Para escrever em outro lugar:
`vde publish --out caminho/publicacao.csv`.

### 4. Avaliar e testar

```bash
vde evaluate                   # métricas, figura, mudança de distribuição e outputs/previsao.txt
pytest                         # 37 testes, inclusive "sem LLM, sem rede, CSV idêntico"
```

`vde evaluate` parte da OOF salva em `outputs/cv_oof.csv` (não refaz a CV) e reescreve `evaluation.json`,
`shift_report.json`, `confiabilidade.png` e `previsao.txt`.

### 5. Classificar uma importação nova

Substitua `anexos/questoes.jsonl` pelo arquivo novo, no mesmo formato, e rode:

```bash
vde publish --allow-annex-change
```

Sem a opção, `vde publish` para, porque o hash do anexo não bate com o `MANIFEST.json`. Com ela, as questões que
não estão no cache são projetadas no espaço LSA salvo, sem reajustar nada. Os artefatos e a configuração
continuam conferidos. `--skip-manifest` é um apelido antigo da mesma opção.

### 6. Retreinar ou refazer os artefatos

```bash
vde build model                # depois de mudar vde/config.toml ou a folha-verdade
vde build all                  # todos os artefatos do zero, na ordem de dependência, e depois o modelo
vde evaluate                   # atualiza métricas e previsão com o modelo novo
```

`vde publish` se recusa a rodar se `vde/config.toml` mudou desde o último `vde build model`: o `model.json`
guarda o hash da configuração com que foi treinado. Refazer os artefatos do zero só reproduz o CSV byte a byte
com as versões de `requirements.lock`, porque o SVD aleatorizado do LSA depende da pilha numérica.

### 7. Erros comuns

| Mensagem | Causa | O que fazer |
|---|---|---|
| `artifacts differ from MANIFEST.json` | um arquivo de `artifacts/` foi editado ou falta | `git checkout artifacts/`, ou refazer com `vde build <artefato>` |
| `annexes differ from MANIFEST.json` | um anexo mudou (importação nova) | `vde publish --allow-annex-change`, ou `vde build all` |
| `vde/config.toml differs from the configuration model.json was trained with` | a configuração mudou depois do treino | desfazer a mudança ou rodar `vde build model` |
| `vde: command not found` | o ambiente virtual não está ativo | `source .venv/bin/activate` ou `python -m vde ...` |

### 8. Comandos e tempos

Medidos num Intel Core i5-14400 (16 núcleos), a partir de um clone limpo:

| Comando | O que faz | Tempo | Usa LLM? |
|---|---|---|---|
| instalação (passo 2) | ambiente virtual e dependências | ~30 s | Não |
| `vde publish` | lê artefatos e escreve `publicacao.csv` | 5 a 11 s | Não |
| `vde evaluate` | métricas a partir da OOF salva, mudança de distribuição, previsão | 10 a 20 s | Não |
| `pytest` | 37 testes | ~40 s | Não |
| `vde build model` | CV de 10 folds x 20 repetições nos 100 rótulos e ajuste final | ~10 min | Não |
| `vde build all` | todos os artefatos do zero e o modelo | ~10 min | Não |
| `vde evaluate --ablations` | as ablações de `ABLATIONS` (3 repetições cada), em `outputs/ablations.csv` | ~25 min | Não |
| `vde ablate --repeats 20 --only E3_always_level1,E3_branch_rho,level1_descend_in_bins,level1_minus_prototypes` | a comparação de 20 repetições da escolha do nó, em `outputs/ablations_20rep.csv` | ~40 min | Não |
| `python scripts/neural_embeddings.py` | LSA contra E5 small, base e large, em `outputs/embeddings_neural.csv` (requer `pip install -e ".[neural]"`) | ~2 h | Não |
| `vde build canonical_tree`, `branch_twins`, `subject_map` | regras determinísticas | < 1 s cada | Não |
| `vde build embeddings` | LSA local sobre 3.891 textos (guarda vetores e o modelo LSA) | 8 s | Não |
| `vde build law_index`, `code_atlas`, `topic_cards` | valida o CSV revisado e grava a proveniência | < 1 s cada | Não |
| `vde diagnose` | compara o banco com a importação nova | ~1 s | Não |
| `vde build <artefato> --source llm` | pede uma **proposta** ao LLM em `artifacts/proposals/` para revisão humana | não medido | Sim, offline |

## Fluxo do sistema

O sistema tem três momentos: construir o conhecimento (offline, uma vez), treinar o modelo (offline, a cada
rodada de rótulos) e publicar (a cada importação, sem LLM). Só o último roda na entrega.

```
 OFFLINE: CONHECIMENTO (vde build <artefato>)            OFFLINE: TREINO (vde build model)
 temas.csv ──> árvore canônica, nós-sombra, lixeiras     folha_verdade.csv (100 rótulos)
           ──> índice de leis, atlas de artigos,               │  CV 10 folds x 20 repetições
               cartões de tema (LLM propõe, vira dado)          │  pesos dos canais + modelo de acerto
 questoes.jsonl + cartões ──> embeddings LSA                    ▼
                                    │                      model.json (pesos, limiar 0,95,
                                    ▼                      hash da config, versões)
                               artifacts/ + MANIFEST.json ◄─────┘
                                    │
 PUBLICAÇÃO (vde publish, sem LLM)  ▼
 questão ──> 1 leitura ──> 2 canais ──> 3 agregação ──> 4 nó ──> 5 confiança ──> 6 decisão ──> publicacao.csv
```

Para cada questão, a publicação faz:

1. **Leitura** (`vde/data/`): limpa o HTML, separa enunciado, comando, assertivas (I, II, III...) e
   alternativas, e marca o tipo da questão (certo/errado, assertivas, múltipla escolha) e a negação.
2. **Canais de evidência** (`vde/channels/`): sete canais dão um score por tema, todos no tema canônico. São a lei
   especial citada, o artigo do CP ou do CPP citado (pelo atlas), o BM25 sobre os cartões de tema, a
   similaridade com os cartões (protótipos), os vizinhos rotulados (kNN), o `assunto_origem` e as quase
   duplicatas de questões rotuladas.
3. **Agregação** (`vde/aggregate.py`): um peso aprendido por canal combina os scores numa probabilidade sobre os
   temas candidatos.
4. **Escolha do nó** (`vde/tree.py`): soma a probabilidade por tema de nível 1 (com a regra de ramos gêmeos) e
   escolhe o maior. Outros ramos com massa >= 0,20 viram extras. O nível 0 nunca é publicado.
5. **Confiança** (`vde/engine.py`): uma logística sobre 17 features da questão (concordância entre canais, massa
   do ramo, suporte dos segmentos, vizinhos `fora`, tipo de questão...) dá P(acerto) do nó escolhido. As
   features ficam limitadas à faixa vista no treino.
6. **Decisão** (`vde/decide.py`), nesta ordem:
   * questão da folha-verdade: sai com o rótulo da equipe;
   * regra explícita de `fora` (quase duplicata de um `fora` rotulado, norma sem tema ativo, disciplina militar
     sem âncora): `fora`, com o motivo;
   * barreira de evidência (nenhum canal de texto com score, ou massa do ramo < 0,30): `revisar`, com a sugestão;
   * confiança >= 0,95: `publicar`;
   * caso contrário: `revisar`, com a sugestão e o motivo (dúvida entre ramos, alternativas espalhadas, lei que
     aponta para outro ramo...).
7. **Escrita** (`vde/pipeline.py`): grava o CSV no formato do LEIA-ME e se recusa a gravar uma linha que o quebre
   (`publicar` ou `revisar` sem tema, `revisar` ou `fora` sem motivo).

**Avaliação** (`vde evaluate`): parte das previsões fora da amostra da CV e calcula precisão, cobertura,
calibração, a validação aninhada da regra de publicação e a previsão (`previsao.txt`), separada por fonte
(banco e importação nova).

**Operação semanal**, como descrita no relatório (seção d):

```
importação nova ──> vde publish --allow-annex-change ──> publicar (aluno) / revisar (fila do professor) / fora
                                                               │
       vde build model + vde evaluate  ◄── correções do professor viram rótulos na folha-verdade
```

## A garantia "nenhum LLM na decisão"

* O caminho de `vde publish` chama apenas `load()` dos artefatos. O cliente de LLM é importado dentro de
  `vde.knowledge.base.propose_with_llm`, que só é chamado por `build(source="llm")`.
* `tests/test_no_llm.py` remove as chaves de API, bloqueia `socket.connect`, faz `import anthropic` falhar,
  roda `publish` e compara o SHA-256 do CSV gerado com o versionado. Outro teste confere, por análise
  estática, que nenhum módulo importa um cliente de LLM no topo.
* `vde publish` confere `artifacts/MANIFEST.json` (artefatos e anexos) e o hash da configuração antes de rodar.
* `vde publish` com os artefatos versionados reproduz `publicacao.csv` byte a byte. Refazer os artefatos do zero
  (`vde build all`) só reproduz o CSV com as versões de `requirements.lock`: o SVD aleatorizado do LSA depende da
  pilha numérica. O `model.json` registra as versões usadas.

## Peças e artefatos

| Peça | Arquivo | Artefato | LLM offline? |
|---|---|---|---|
| Árvore canônica: 20 nós-sombra mapeados, lixeiras marcadas | `vde/knowledge/canonical_tree.py` | `canonical_tree.csv` | Não |
| Regra de ramos gêmeos, aprendida dos rótulos e reajustada em cada fold | `vde/knowledge/canonical_tree.py` (`learn_twins`) | `branch_twins.csv` (registro) | Não |
| Índice de leis (60 leis e códigos, com apelidos) | `vde/knowledge/law_index.py` | `law_index.csv` | Proposto por LLM, revisado pela autora |
| Atlas dos códigos: faixas de artigos do CP e do CPP para temas | `vde/knowledge/code_atlas.py` | `code_atlas.csv` | Proposto por LLM, revisado pela autora |
| Cartões de tema para os 351 temas publicáveis | `vde/knowledge/topic_cards.py` | `topic_cards.csv` | Proposto por LLM, revisado pela autora |
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
  Os três arquivos foram revisados pela autora em 04/11/2026, sem nenhuma linha alterada.
* **MD04:** os 20 nós sem descrição e sem filhos ("Teoria da Pena (Parte 1)", "Penal > Lei Maria da Penha"...) são
  mapeados para o gêmeo canônico, detectado automaticamente por similaridade de título.
* **MD05:** Lei 9.099 vai para Legislação Especial > Juizado Especial Criminal. O Procedimento Sumaríssimo de
  Processo Penal é somado a esse ramo pela regra de gêmeos, que agora é aprendida dos rótulos: o sumaríssimo
  aparece 4 vezes como extra ao lado de um central do Juizado e nunca como central.
* **MD07 a MD09:** são `fora` por regra: o procedimento dos crimes funcionais (CPP 513 a 518), os crimes contra a
  ordem econômica (tema inativo) e o direito penal e processual militar.
* **MD10, MD13:** ρ = 0,85 (só vale com `node_mode = "branch"`); nunca publicar no nível 0.
* **MD14:** publicar no tema de nível 1. Venceu a comparação de 20 repetições (precisão nos 10%: 0,985 contra
  0,970 ao descer ao subtema e 0,975 ao descer só nas lixeiras; `vde ablate --repeats 20`). O custo é que o aluno vê a questão no tema, não no subtema, e 30 das 75 publicadas caem em temas
  marcados como lixeira (sempre com confiança >= 0,95).
* **MD15:** regra de limiar calibrada (publicar quando P(acerto) >= 0,95) em vez de buscar na curva.
* **MD17:** barreira de evidência: nunca publicar sem nenhum canal de texto com score ou com massa do ramo < 0,30.
  Uma questão vazia chegava a 0,91 de confiança; o corte fica abaixo da menor massa publicada (0,41).
* **MD18:** recalibração de Platt testada e descartada: ECE igual (0,049 contra 0,046) e cobertura pela metade.
* **MD16:** o cofre de 20 questões da v0.1 foi inspecionado na análise de erro (Q0252, Q0303) e foi juntado aos
  rótulos. Não há mais teste cego: a garantia vem da CV aninhada.

## Fluxo de desenvolvimento (git)

`feature/* -> develop -> main`, sempre com merge `--no-ff`. O release `v0.2.0` está em `main`: `feature/forecast-format`,
`feature/nested-cv`, `feature/publish-slice`, `feature/shift-forecast`, `feature/production-hardening`,
`feature/docs-v0.2`, `feature/forecast-calibrated`, `feature/neural-embeddings`, `feature/review-fixes` e
`feature/ablations-refresh`. O release `v0.1.0` continua marcado pela tag de mesmo nome.
