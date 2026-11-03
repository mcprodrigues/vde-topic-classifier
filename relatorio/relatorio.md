# Relatório: classificação de tópicos sem LLM na decisão

**Resumo.** O sistema organiza o conhecimento em quatro artefatos revisáveis (árvore canônica, índice de leis,
atlas de artigos do CP e do CPP, cartões de tema) e transforma a evidência de sete canais determinísticos numa
probabilidade de acerto calibrada. Fora da amostra, acerta o nó publicado em 65,5% das questões de desenvolvimento
e em 80% do cofre. Os 10% mais confiantes acertam 100%. Ainda assim, **com 100 rótulos nenhum limiar certifica
95% de precisão**, e a entrega publica zero questões, mandando 548 para revisão ordenadas por confiança. A seção
(c) mostra que o teto vem sobretudo da convenção de `fora` da equipe, e não só do classificador.

## a) O sistema, peça por peça

**1. Leitura da questão.** O HTML é limpo de forma determinística e a questão é separada em enunciado, comando,
assertivas (I, II, III...) e alternativas. São 2.881 segmentos, e cada um recebe seu próprio vetor. O tipo da
questão (certo/errado, assertivas, múltipla escolha) e a negação ("INCORRETA", "exceto") viram features.

**2. Árvore canônica.** Dos 375 temas ativos, 20 são "nós-sombra" de nível 1, sem descrição e sem filhos: as
versões "(Parte 1/2)" das lixeiras e cópias em Penal ou Processo Penal de temas de Legislação Especial. São 8
títulos duplicados, não 3. A detecção é automática (similaridade de título) e todo o score é somado no gêmeo
canônico, que é o que a folha usa. Sobram 351 temas publicáveis, 89 deles de nível 1. Uma regra de ramos gêmeos,
aprendida da folha, manda o Procedimento Sumaríssimo de Processo Penal para Legislação Especial > Juizado Especial
Criminal.

**3. Conhecimento proposto por LLM e versionado como dado.** Um LLM propôs, offline e vendo só a árvore:
(i) o **índice de leis**, com 60 entradas, apelidos ("Lei Maria da Penha", "LEP", "ECA") e escopo `in`/`out`;
(ii) o **atlas dos códigos**, com 261 faixas de artigos, por exemplo CP 155 a 156 = Furto e CPP 28-A = ANPP.
O CP e o CPP já são uma árvore alinhada à da plataforma, e faixas generalizam para artigos que a folha nunca
viu; (iii) os **cartões de tema**, com termos-chave, artigos e súmulas para cada um dos 351 temas, que formam o
caminho zero-shot para os 290 temas sem exemplo rotulado. Não gerei pseudo-enunciados: os termos já levaram o
acerto de 47,5% para 65%, e enunciados sintéticos aumentariam o risco de vazamento de estilo. A revisão humana
dos três arquivos está pendente e registrada.

**4. Canais de evidência**, todos com a mesma interface (`score(qid) -> {tema: score}`), o que permite ligar e
desligar cada um: lei especial citada, artigo de código citado, BM25 sobre os cartões, similaridade de
embedding com os cartões (protótipos), kNN nos rotulados (reajustado por fold, sem votar em si mesmo),
`assunto_origem` como evidência fraca e quase duplicatas pelo texto completo.

**5. Agregação.** Uma logit condicional aprende **um peso por canal** maximizando a massa de probabilidade nos
nós que contam como acerto pela regra do ramo. Os rótulos ensinam em que evidência confiar, não quais temas
existem, e por isso o modelo generaliza para temas sem exemplo. Pesos finais: BM25 0,57, protótipos 0,44,
assunto 0,38, kNN 0,36, lei especial 0,28, artigo 0,17, duplicatas 0 (não há duplicatas no dev).

**6. Escolha do nó.** Pela regra de acerto, P(acerto | tema T) = p(T) + Σ p(filhos de T), que nunca é menor
que P(acerto | subtema). Publico no melhor ramo de nível 1 e desço ao subtema só se ele tiver ρ = 0,85 da massa
do ramo (valor fixado a priori, pela utilidade para o aluno). O nível 0 é proibido: contaria como acerto para
qualquer rótulo de nível 1 e recriaria a maior lixeira possível.

**7. Confiança e limiar.** Uma logística com 17 features agnósticas à classe (concordância entre canais, massa do
ramo, suporte dos segmentos, vizinhos `fora`, tipo de questão...) é treinada em features fora da amostra de
folds internos. O limiar é o menor valor com P(precisão >= 0,95) >= 0,80 sob posterior Beta e precisão pontual
>= 0,97. Também calculo Learn-then-Test, que dá o mesmo resultado: nenhum limiar certificável.

**8. Decisão.** `fora` só sai de regra explícita: quase duplicata de um `fora` rotulado, norma sem tema ativo
(CPP 513 a 518, crimes contra a ordem econômica) ou disciplina militar sem âncora no escopo. Tudo que apenas
"parece fora" vai para revisão. Cada `revisar` traz a sugestão e o motivo (dúvida entre dois ramos, alternativas
espalhadas, lei que aponta para outro ramo, vizinhos `fora`).

**O que testei e descartei** (`outputs/ablations.csv`, 3 repetições cada; o desvio entre repetições é de 0,9 p.p.,
então diferenças abaixo de 2 p.p. são ruído):

| Experimento | Acerto | AURC | Veredito |
|---|---|---|---|
| E0: BM25 só com título e caminho | 0,475 | 0,328 | Linha de base |
| E1: BM25 com cartões + kNN | 0,612 | 0,232 | Os cartões são o maior ganho (+14 p.p.) |
| E1 + protótipos | 0,646 | 0,201 | Entra |
| Sistema completo | 0,654 | 0,196 | Referência |
| Sem lei especial / sem artigo de código | 0,650 / 0,650 | 0,196 / 0,190 | Empate; ficam como features legíveis |
| + prior de lixeira | 0,654 | 0,196 | **Descartado**: peso aprendido = 0 |
| + canal de segmentos | 0,650 | 0,174 | Melhora o ranking, mas não o acerto; virou feature, não canal |
| + matriz de confusão do `assunto_origem` | 0,667 | 0,201 | **Descartado**: AUC pior, e o formato muda na importação nova |
| E3: folha top-1, sem raciocínio de ramo | 0,621 | 0,222 | **Descartado**: -3,3 p.p. |
| E3: sempre nível 1 | 0,667 | 0,169 | Melhor na métrica; mantive ρ = 0,85 pelo aluno (custa ~1,3 p.p.) |
| Calibração isotônica | ECE 0,171 | | **Descartada** (bruta 0,072; Platt 0,049 com Brier pior) |
| Quase duplicatas | sem efeito | | Mantida só como regra de `fora` (Q0553 herda o `fora` de Q0017) |

## b) Os 100 rótulos

**Uso.** 20 questões estratificadas formam um **cofre**, aberto uma única vez no fim. As 80 restantes passam por
CV de 10 folds repetida 20 vezes. Dentro de cada fold, o kNN, as duplicatas e os pesos são ajustados só no treino,
e o modelo de acerto é treinado em features fora da amostra de 5 folds internos. O limiar é escolhido nas
confianças fora da amostra, e sua precisão honesta vem de uma segunda CV sobre as questões (escolho em 4/5, meço
em 1/5). Toda a análise de erro foi feita nas 80; o cofre deu 80% de acerto contra 65,5% na CV, sem sinal de
overfitting das minhas decisões.

**O que mostraram sobre o sistema.** A confiança é razoavelmente calibrada (ECE 0,07; figura em
`outputs/confiabilidade.png`), com AUC de 0,73. A concordância entre canais é a feature mais informativa
(AUC 0,77 sozinha). Âncoras de lei ajudam pouco: artigos citados na narrativa de um caso descrevem os fatos, não
o tema. A Q0208 cita o art. 121 do CP e é sobre concurso de pessoas.

**O que mostraram sobre a convenção da equipe.** (1) O nível 1 é suficiente: publicar no ancestral de nível 1 do
rótulo acerta 92 de 92. (2) Para leis especiais, a equipe usa o ramo de Legislação Especial (LEP 4 de 5 casos,
Juizado 3 de 3), e as sombras quase nunca. (3) Os 8 `fora` têm três origens: tema inativo (Q0039, cartel),
lacuna da árvore (Q0017, procedimento dos crimes funcionais; Q0002, criminologia crítica) e questões que parecem
classificáveis (Q0043, Q0087, Q0252, Q0291, Q0303).

## c) Diagnóstico

**O teto vem dos `fora` disfarçados.** Entre as 20 questões mais confiantes do dev, os dois únicos erros são Q0043
(fixação da pena) e Q0291 (prazo do inquérito na Lei de Drogas), ambas `fora` na folha. O sistema as coloca com
alta confiança em Teoria da Pena e Tráfico de Drogas, porque é isso que o texto diz. Se cerca de 5% da população
for desse tipo e indistinguível pelo texto, a precisão máxima de qualquer publicação gira em torno de 95%: o
orçamento de erro inteiro. A solução não é um modelo melhor, é a regra que a equipe usa para `fora` (memo).

**Por que zero publicadas.** Com 80 rótulos, certificar 95% com 80% de confiança exige 32 acertos seguidos sem
erro no topo da fila; com 90% de confiança, 45. A curva fora da amostra cai abaixo de 95% depois de 8 questões.
Operar no ponto (topo de 10%) daria cobertura de ~10% com intervalo de precisão de ~70% a 100%. Preferimos não
publicar a publicar sem garantia.

**Mudança de distribuição.** A importação nova vem de outra fonte de metadados: zero questões rotuladas como
Legislação Especial (contra 32 no banco), 10 militares, assunto em formato de edital e enunciado mediano mais
curto (662 contra 830 caracteres). Mas o classificador de domínio, usando as features que o modelo vê, separa as
duas fontes com AUC de 0,56, quase o acaso. A confiança média é parecida (0,63 nas novas, 0,58 no restante do
banco). Por isso `disciplina_origem` só entra nas regras de `fora`.

**O que o sistema ainda erra e o que resolveria.** (1) Questões de caso em que a narrativa esconde o tema
doutrinário (erro de tipo num caso de crime sexual): resolveria um segundo nível de cartões com termos de teoria
geral, ou embeddings neurais. O ambiente de construção não alcançou o hub de modelos, então o LSA local é o
backend atual, e `vde build embeddings --backend st` troca o modelo sem mudar mais nada. (2) Colchas de assertivas
que cobrem quatro temas: hoje vão para revisão por dispersão, e a equipe precisa dizer se são `fora` ou se ganham
extras. (3) Gêmeos entre disciplinas (Ação Penal em Penal e em Processo Penal): faltam rótulos para decidir.

## d) Produção: 50 mil questões por semana

**Custo e tempo.** Sem GPU: o LSA vetoriza 3.891 textos em 4 s, e 50 mil questões (~250 mil segmentos) levam
~5 min de CPU. Os canais e a decisão rodam a ~8 ms por questão, cerca de 7 min por semana. Custo de LLM na
decisão: zero. Com um embedding neural pequeno em CPU, seriam ~1,5 h por semana; via API em cache, alguns dólares.
O LLM só entra quando um tema novo pede um cartão (centavos por tema).

**Como a correção do professor vira dado.** Cada `revisar` confirmado ou corrigido entra num CSV de rótulos com
data e versão do modelo. Toda semana: (1) reajustar pesos e o modelo de acerto (`vde build model`, minutos);
(2) recertificar o limiar com os rótulos acumulados; (3) monitorar a AUC do classificador de domínio, a fração
de `fora` por regra e a distribuição de confiança, com alerta se a AUC passar de 0,7. Os cartões são revistos
quando um tema acumula erros. A publicação destrava quando o topo da fila acumular confirmações suficientes:
com ~30 acertos seguidos, o limiar certifica no nível de 80%.

**Fila do professor.** As 548 questões em revisão saem ordenadas por confiança; 115 têm confiança >= 0,90 e são
confirmações rápidas. Confirmá-las primeiro é ao mesmo tempo o trabalho mais barato e o que mais rápido libera a
publicação automática.
