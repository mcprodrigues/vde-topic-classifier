# Relatório: classificação de tópicos sem LLM na decisão

**Resumo.** O sistema organiza o conhecimento em quatro artefatos revisáveis (árvore canônica, índice de leis,
atlas de artigos do CP e do CPP, cartões de tema) e transforma a evidência de sete canais determinísticos numa
probabilidade de acerto calibrada. A regra de publicação, fixada antes de olhar a curva, é: publicar no tema de
nível 1 quando P(acerto) >= 0,95. Nas 559 questões avaliadas, ela publica **76 (13,6%)**, manda 472 para revisão
e 11 para `fora`. Na validação aninhada sobre os 100 rótulos, a regra não errou nenhuma (7,8 publicadas por
repetição; limite inferior de 90%: 0,75). A previsão para a entrega é **0,93 (0,78 a 1,00)**, com 49% de chance
de ficar abaixo de 0,95: o risco está na importação nova, que publica mais e não tem nenhum rótulo.

**O que mudou da v0.1 para a v0.2.** A v0.1 não publicava nada: exigia uma certificação impossível com 80 rótulos e
media o topo da fila sobre a confiança média de 20 repetições, que nenhum modelo implantado produz. Além disso, o
cofre de 20 questões tinha sido lido na análise de erro. A v0.2 junta o cofre aos rótulos, mede tudo uma repetição
por vez, troca a busca do limiar por uma regra fixa e prevê a precisão separadamente por fonte.

## a) O sistema, peça por peça

**1. Leitura da questão.** O HTML é limpo de forma determinística e a questão é separada em enunciado, comando,
assertivas (I, II, III...) e alternativas, cada uma com seu próprio vetor. O tipo da questão (certo/errado,
assertivas, múltipla escolha) e a negação ("INCORRETA", "exceto") viram features.

**2. Árvore canônica.** Dos 375 temas ativos, 20 são "nós-sombra" de nível 1, sem descrição e sem filhos: as
versões "(Parte 1/2)" das lixeiras e as cópias em Penal ou Processo Penal de temas de Legislação Especial. A
detecção é automática (similaridade de título) e todo o score é somado no gêmeo canônico. Sobram 351 temas
publicáveis. A **regra de ramos gêmeos** também é aprendida dos rótulos: um ramo que só aparece como extra ao lado
do central de outra disciplina (o Procedimento Sumaríssimo ao lado do Juizado Especial Criminal, 4 vezes) é somado
a ela. A regra é reajustada dentro de cada fold, então não vaza rótulo na CV.

**3. Conhecimento proposto por LLM e versionado como dado.** Um LLM propôs, offline: (i) o **índice de leis**, com
60 entradas, apelidos e escopo `in`/`out`; (ii) o **atlas dos códigos**, com 261 faixas de artigos (CP 155 a 156 =
Furto); (iii) os **cartões de tema**, com termos-chave para os 351 temas, que são o caminho zero-shot para os temas
sem exemplo rotulado. O pedido não incluía as questões, mas a sessão de autoria tinha acesso ao repositório, então
isso não é verificável e a proveniência registra assim. A revisão humana dos três arquivos está pendente.

**4. Canais de evidência**, todos com a interface `score(qid) -> {tema: score}`: lei especial citada, artigo de
código citado, BM25 sobre os cartões, similaridade com os cartões (protótipos), kNN nos rotulados (reajustado por
fold), `assunto_origem` como evidência fraca e quase duplicatas.

**5. Agregação.** Uma logit condicional aprende **um peso por canal**, maximizando a massa de probabilidade nos
nós que contam como acerto. Os rótulos ensinam em que evidência confiar, não quais temas existem. Pesos finais:
BM25 0,57, protótipos 0,44, assunto 0,38, kNN 0,36, lei especial 0,28, artigo 0,17, duplicatas 0.

**6. Escolha do nó.** Pela regra de acerto, publicar no tema de nível 1 acerta sempre que o rótulo é o tema ou um
dos seus subtemas. Na comparação de 20 repetições (`outputs/ablations_20rep.csv`), publicar sempre no nível 1
venceu descer ao subtema com ρ = 0,85: precisão nos 10% mais confiantes 0,985 contra 0,970, AURC 0,142 contra
0,162. O custo: o aluno vê a questão no tema, não no subtema, e **30 das 76 publicadas caem em temas marcados
como lixeira** (Crimes contra a Administração Pública, 16; Teoria do Crime: Fato Típico, 6). São questões que o
sistema põe no tema certo com confiança >= 0,95, não questões alheias ao tema, mas o professor deve saber disso.
O nível 0 continua proibido.

**7. Confiança e regra de publicação.** Uma logística com 17 features agnósticas à classe (concordância entre
canais, massa do ramo, suporte dos segmentos, vizinhos `fora`, tipo de questão...) é treinada em features fora da
amostra de folds internos. A calibração é boa: ECE 0,049 nas linhas da CV (Platt 0,033 e isotônica 0,090 não
compensam). A regra publica quando a confiança >= 0,95. Testei também escolher o limiar olhando a curva fora da
amostra: na validação aninhada, isso dá precisão 0,94, porque a busca acha pontos de sorte com poucos rótulos.
Uma regra fixa não tem esse otimismo.

**8. Decisão.** `fora` só sai de regra explícita: quase duplicata de um `fora` rotulado, norma sem tema ativo
(CPP 513 a 518, crimes contra a ordem econômica) ou disciplina militar sem âncora no escopo. Tudo que apenas
"parece fora" vai para revisão, com a sugestão e o motivo.

**O que testei e descartei** (`outputs/ablations.csv`, 3 repetições cada, desvio de ~1 p.p. entre repetições;
as decisões de configuração usaram 20 repetições):

| Experimento | Acerto | AUC | AURC | Veredito |
|---|---|---|---|---|
| E0: BM25 só com título e caminho | 0,487 | 0,757 | 0,304 | Linha de base |
| E1: BM25 com cartões + kNN | 0,647 | 0,747 | 0,197 | Cartões ~+14 p.p., kNN ~+2 p.p. (sem kNN: 0,653) |
| E1 + protótipos | 0,670 | 0,748 | 0,167 | Entra; sem protótipos o acerto cai para 0,647 |
| Sistema completo (nível 1) | 0,673 | 0,792 | 0,142 | Referência |
| Sem lei especial / sem artigo | 0,673 / 0,680 | 0,782 / 0,764 | 0,146 / 0,149 | Ficam: não mudam o acerto, mas melhoram a ordenação |
| + prior de lixeira | 0,673 | 0,792 | 0,142 | **Descartado**: peso aprendido = 0 |
| + canal de segmentos | 0,653 | 0,810 | 0,144 | Ordena melhor, acerta menos; virou feature, não canal |
| + matriz de confusão do `assunto_origem` | 0,707 | 0,732 | 0,143 | **Descartado**: AUC pior, e o formato muda na importação nova |
| Escala linear dos canais | 0,690 | 0,738 | 0,155 | **Descartado**: AUC pior |
| Folha top-1, sem raciocínio de ramo | 0,643 | 0,778 | 0,179 | **Descartado** |
| Descer ao subtema (ρ = 0,85) | 0,663 | 0,754 | 0,165 | **Descartado** na v0.2 (era o padrão da v0.1) |
| Fatias de publicação `clean` / `anchored` | | | | **Descartadas**: precisão aninhada igual (1,00), cobertura cai pela metade |

## b) Os 100 rótulos

**Uso.** Os 100 rótulos passam por CV de 10 folds repetida 20 vezes. Dentro de cada fold, o kNN, as duplicatas, a
regra de gêmeos e os pesos são ajustados só no treino, e o modelo de acerto é treinado em features fora da amostra
de 5 folds internos. Cada repetição é um modelo implantável, então as métricas são calculadas uma repetição por
vez. A regra de publicação passa ainda por uma validação aninhada (aplicada em 1/5 das questões, 10 vezes).

**Por que não há mais cofre.** A v0.1 separou 20 questões, mas duas delas (Q0252 e Q0303) foram lidas na análise
de erro e citadas no memo. Um cofre lido não é cego, então as 20 voltaram para a CV. Dois sinais pedem cautela:
juntar o cofre subiu a AUC de 0,73 para 0,76, e as questões do cofre casam mais termos raros dos cartões de tema do
que as do banco (2,3 contra 1,6 por questão). Não há prova de vazamento, mas os cartões foram escritos numa sessão
com acesso às questões.

**O que mostraram sobre o sistema.** O acerto do nó publicado é 0,676 (desvio de 0,010 entre repetições), a AUC
da confiança é 0,79, e os 10% / 20% mais confiantes acertam 0,985 / 0,965. Em 10% das repetições, porém, esses
números caem para 0,90. A regra de publicação escolhe em média 7,8 rotuladas por repetição, todas certas. São 18
questões distintas; como as mesmas questões se repetem entre repetições, o tamanho efetivo é 10,9, não 156.

**O que mostraram sobre a convenção da equipe.** (1) Para leis especiais, a equipe usa o ramo de Legislação
Especial, e as sombras quase nunca. (2) O Procedimento Sumaríssimo só aparece como extra. (3) Os 8 `fora` têm três
origens: tema inativo (Q0039), lacuna da árvore (Q0017, Q0002) e questões que parecem classificáveis (Q0043, Q0087,
Q0252, Q0291, Q0303). Nenhuma das 8 chega a 0,95 de confiança na v0.2; a mais alta é Q0043, com 0,89.

## c) Diagnóstico

**A importação nova é o maior risco.** Os 100 rótulos vêm todos do banco (Q0002 a Q0356); nenhum vem de Q0360 a
Q0659, que são 300 das 559 avaliadas. O classificador de domínio quase não separa as fontes pelas features do
modelo (AUC 0,55), mas a importação nova tem confiança média maior (0,70 contra 0,65) e mais questões acima de 0,95
(16% contra 10%). Por isso ela publica 49 das 76 questões. A previsão trata essa fonte à parte: mesmos rótulos,
pesos de importância e tamanho efetivo cortado pela metade, o que dá 0,91 (0,67 a 1,00) contra 0,96 (0,81 a 1,00)
no banco.

**`fora` disfarçados.** Questões que a equipe marca como `fora` mas que o texto põe num tema (Q0043, fixação da
pena; Q0291, prazo do inquérito na Lei de Drogas) eram os erros do topo na v0.1. A solução não é um modelo melhor,
é a regra que a equipe usa para `fora` (memo).

**O que o sistema ainda erra e o que resolveria.** (1) Questões de caso em que a narrativa esconde o tema
doutrinário: resolveria um segundo nível de cartões com termos de teoria geral, ou embeddings neurais
(`vde build embeddings --backend st`). (2) Colchas de assertivas que cobrem vários temas: vão para revisão por
dispersão. (3) Gêmeos entre disciplinas (Ação Penal em Penal e em Processo Penal): faltam rótulos.

## d) Produção: 50 mil questões por semana

**Custo e tempo.** Sem GPU. O modelo LSA (vocabulário, idf e componentes) fica salvo junto dos vetores, e uma
questão nova é projetada no mesmo espaço sem reajuste. Os canais e a decisão rodam a ~8 ms por questão: 50 mil
questões levam ~7 min de CPU por semana. Custo de LLM na decisão: zero. O LLM só entra quando um tema novo pede um
cartão, e o cartão é revisado antes de entrar.

**Como a correção do professor vira dado.** Cada `revisar` confirmado ou corrigido entra num CSV de rótulos com
data e versão do modelo. Toda semana: (1) reajustar pesos e o modelo de acerto (`vde build model`, minutos);
(2) refazer a CV aninhada e a previsão; (3) monitorar a AUC do classificador de domínio, a fração publicada por
fonte e a distribuição de confiança, com alerta se a AUC passar de 0,7 ou se a fração publicada de uma fonte
dobrar. O primeiro uso do tempo do professor deve ser rotular uma amostra das publicadas da importação nova: é a
fonte sem rótulo e a que mais pesa no intervalo.
