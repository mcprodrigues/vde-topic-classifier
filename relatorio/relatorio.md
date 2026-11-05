# Relatório: classificação de tópicos sem LLM na decisão

**Resumo.** O sistema organiza o conhecimento em quatro artefatos revisáveis (árvore canônica, índice de leis,
atlas de artigos do CP e do CPP, cartões de tema) e transforma a evidência de sete canais determinísticos numa
probabilidade de acerto calibrada. A regra, fixada antes de olhar a curva: publicar no tema de nível 1 quando
P(acerto) >= 0,95 e houver evidência de texto. Nas 559 questões avaliadas, ela publica **75 (13,4%)**, manda 473
para revisão e 11 para `fora`. Na validação aninhada sobre os 100 rótulos, não errou nenhuma (7,9 publicadas por
repetição; limite inferior de 90%: 0,75). A previsão é **0,95 (0,87 a 1,00)**, com 40% de chance de ficar abaixo de
0,95: o risco está na importação nova, que publica mais e não tem nenhum rótulo.

## a) O sistema, peça por peça

**1. Leitura.** O HTML é limpo de forma determinística e a questão é separada em enunciado, comando, assertivas e
alternativas, cada uma com seu vetor. O tipo da questão e a negação ("INCORRETA", "exceto") viram features.

**2. Árvore canônica.** Dos 375 temas ativos, 20 são "nós-sombra" de nível 1, sem descrição e sem filhos: versões
"(Parte 1/2)" das lixeiras e cópias em Penal ou Processo Penal de temas de Legislação Especial. A detecção é
automática (similaridade de título) e todo o score vai para o gêmeo canônico: sobram 351 temas publicáveis. A
**regra de ramos gêmeos** é aprendida dos rótulos e reajustada em cada fold: um ramo que só aparece como extra ao
lado do central de outra disciplina (Procedimento Sumaríssimo ao lado do Juizado Especial Criminal) é somado a ela.

**3. Conhecimento proposto por LLM e versionado como dado.** Um LLM propôs, offline, o **índice de leis** (60
entradas, com apelidos e escopo), o **atlas dos códigos** (261 faixas de artigos: CP 155 a 156 = Furto) e os
**cartões de tema** (termos-chave dos 351 temas, o caminho zero-shot para os temas sem exemplo). O pedido não
incluía as questões, mas a sessão tinha acesso ao repositório, e a proveniência registra isso. Os três arquivos
foram revisados pela autora, sem nenhuma linha alterada.

**4. Canais de evidência** (`score(qid) -> {tema: score}`): lei especial citada, artigo de código citado, BM25
sobre os cartões, similaridade com os cartões, kNN nos rotulados, `assunto_origem` como evidência fraca e quase
duplicatas. kNN e duplicatas são reajustados em cada fold e nunca veem a própria questão.

**5. Agregação.** Uma logit condicional aprende **um peso por canal**, maximizando a massa nos nós que contam como
acerto: os rótulos ensinam em que evidência confiar, não quais temas existem.

**6. Escolha do nó.** Publicar no tema de nível 1 acerta sempre que o rótulo é o tema ou um subtema dele. Em 20
repetições (`outputs/ablations_20rep.csv`), o nível 1 venceu descer ao subtema (precisão nos 10% mais confiantes
0,985 contra 0,970; AURC 0,143 contra 0,163) e descer só nas lixeiras (0,975; 0,164). O custo: **30 das 75
publicadas caem em temas marcados como lixeira**, sempre com confiança >= 0,95, e o professor deve saber disso.

**7. Confiança.** Uma logística com 17 features agnósticas à classe (concordância entre canais, massa do ramo,
suporte dos segmentos, vizinhos `fora`, tipo de questão...) é treinada em features fora da amostra de folds
internos, com as features limitadas à faixa vista no treino. ECE 0,046 nas linhas da CV. Uma recalibração de Platt
foi testada e descartada: não melhora o ECE e corta a cobertura de 75 para 37, porque puxa para baixo o topo da
fila, onde os rótulos acertam 18 de 18. Escolher o limiar olhando a curva daria precisão aninhada 0,94 (otimismo da
seleção); por isso a regra é fixa em 0,95.

**8. Decisão.** `fora` só sai de regra explícita: quase duplicata de um `fora` rotulado, norma sem tema ativo (CPP
513 a 518, crimes contra a ordem econômica) ou disciplina militar sem âncora. O que só "parece fora" vai para
revisão. Uma barreira de evidência manda para revisão a questão sem nenhum canal de texto com score ou com massa
do ramo < 0,30: hoje não barra nenhuma publicação, mas uma questão vazia chegava a 0,91 de confiança.

**O que testei e descartei** (`outputs/ablations.csv`, 3 repetições; desvio de ~1 p.p. entre repetições):

| Experimento | Acerto | AUC | AURC | Veredito |
|---|---|---|---|---|
| BM25 só com título e caminho | 0,487 | 0,760 | 0,302 | Linha de base |
| BM25 com cartões + kNN | 0,647 | 0,745 | 0,199 | Cartões ~+14 p.p. |
| Sistema completo (nível 1) | 0,673 | 0,791 | 0,143 | Referência |
| Sem lei especial / sem artigo | 0,673 / 0,680 | 0,781 / 0,764 | 0,146 / 0,149 | Ficam: melhoram a ordenação |
| + segmentos como canal | 0,653 | 0,809 | 0,144 | Viraram feature, não canal |
| + matriz de confusão do `assunto_origem` | 0,707 | 0,735 | 0,143 | Descartado: AUC pior e não transfere |
| Folha top-1, sem raciocínio de ramo | 0,643 | 0,781 | 0,178 | Descartado |
| Embeddings E5 small / base / large (20 rep.) | 0,68 / 0,69 / 0,68 | 0,72 / 0,73 / 0,74 | 0,18 / 0,18 / 0,17 | Descartados: aninhada 0,88 / 0,88 / 0,95 |

Também descartados: prior de lixeira (peso aprendido 0), escala linear dos canais (AUC 0,737) e as fatias de
publicação `clean` e `anchored` (mesma precisão, metade da cobertura).

## b) Os 100 rótulos

**Uso.** CV de 10 folds repetida 20 vezes. Em cada fold, kNN, duplicatas, regra de gêmeos e pesos são ajustados só
no treino, e o modelo de acerto usa features fora da amostra de 5 folds internos. Cada repetição é um modelo
implantável, então as métricas são calculadas uma repetição por vez; a regra passa ainda por uma validação
aninhada (aplicada em 1/5 das questões, 10 vezes). Não há cofre: as 20 questões separadas na v0.1 foram lidas na
análise de erro (Q0252, Q0303) e voltaram para a CV.

**O que mostraram sobre o sistema.** O acerto do nó publicado é 0,676 (desvio de 0,010 entre repetições), a AUC é
0,79 e os 10% / 20% mais confiantes acertam 0,985 / 0,965, mas caem para 0,90 em 10% das repetições. A regra
publica 7,9 rotuladas por repetição, todas certas: são 18 questões distintas, tamanho efetivo 10,9, não 156.

**O que mostraram sobre a convenção da equipe.** (1) Leis especiais vão para o ramo de Legislação Especial; as
sombras quase nunca são usadas. (2) O Procedimento Sumaríssimo só aparece como extra. (3) Os 8 `fora` têm três
origens: tema inativo (Q0039), lacuna da árvore (Q0017, Q0002) e questões que parecem classificáveis (Q0043, Q0087,
Q0252, Q0291, Q0303). Nenhuma das 8 chega a 0,95; a mais alta é Q0043, com 0,89.

## c) Diagnóstico

**A importação nova é o maior risco.** Nenhum dos 100 rótulos vem de Q0360 a Q0659, que são 300 das 559 avaliadas.
O classificador de domínio quase não separa as fontes (AUC 0,55), mas a importação nova tem mais questões acima de
0,95 (16% contra 10%) e publica 48 das 75. A previsão parte da confiança de cada publicada e soma um desvio de
calibração comum por fonte: no banco, o maior entre o ECE (0,046) e a incerteza dos rótulos dentro da faixa
publicada (0,056); na importação nova, √2 vezes maior. Dá 0,96 (0,85 a 1,00) no banco e 0,95 (0,81 a 1,00) na
importação nova. A checagem pelos rótulos (10,9 questões efetivas, todas certas) dá 0,96: coerente, mas larga.

**`fora` disfarçados.** Questões que a equipe marca como `fora` mas que o texto põe num tema (Q0043, fixação da
pena; Q0291, prazo do inquérito na Lei de Drogas) eram os erros do topo na v0.1. A solução é a regra de `fora`
da equipe (memo), não um modelo melhor.

**O que o sistema ainda erra.** (1) Questões de caso em que a narrativa esconde o tema doutrinário: resolveria um
segundo nível de cartões com termos de teoria geral (embeddings neurais não resolvem, ver tabela). (2) Colchas de
assertivas que cobrem vários temas: vão para revisão por dispersão. (3) Gêmeos entre disciplinas (Ação Penal em
Penal e em Processo Penal): faltam rótulos.

## d) Produção: 50 mil questões por semana

**Custo e tempo.** Sem GPU e sem LLM na decisão. Uma questão nova é projetada no espaço LSA salvo, sem reajuste
(`vde publish --allow-annex-change`, que continua conferindo artefatos e configuração). A decisão leva ~3 ms por
questão: 50 mil questões, ~3 min de CPU. kNN e duplicatas comparam cada questão só com as rotuladas, então a
memória cresce com os rótulos, não com a importação. O LLM só entra quando um tema novo pede um cartão, revisado
antes de entrar.

**Como a correção do professor vira dado.** Cada `revisar` confirmado ou corrigido entra num CSV de rótulos com
data e versão do modelo. Toda semana: reajustar pesos e modelo de acerto (`vde build model`, ~10 min), refazer a
CV aninhada e a previsão, e monitorar a fração publicada por fonte, a distribuição de confiança e a AUC do
classificador de domínio (alerta acima de 0,7). O primeiro uso do tempo do professor: conferir uma amostra das
publicadas da importação nova.

## O que eu faria com mais tempo

1. **Rotular a importação nova.** Uns 50 rótulos de Q0360 a Q0659, a começar pelas publicadas, trocariam o desvio
   de calibração suposto (√2 vezes o do banco) por um medido.
2. **Mais rótulos no topo da fila.** A faixa >= 0,95 se apoia em 18 questões; cerca de 31 confirmações seguidas do
   professor certificam 95% com 80% de confiança e permitiriam testar limiares abaixo de 0,95.
3. **Um teste cego novo:** 20 rótulos separados sem olhar, só para conferir a previsão no fim.
4. **Acabamento restante.** Separar as interfaces dos canais (o motor usa métodos que não estão na interface,
   como `out_share` e `top_cosine`) e dar à confiança das linhas `fora` um sentido próprio: hoje é a do tema
   que não vai para a saída.
