# Toranja Tabular - Tabular GrapeFruit
## Explicabilidade para modelos que possuem muitas variáveis explicativas com valores ausentes (Missings).

O [Lime](https://github.com/marcotcr/lime) em sua essência é um excelente método para explicar modelos não lineares de forma simples e rápida, porém, é muito comum encontrar em bases produtivas, variáveis explicativas que contenham valores ausentes, chamados de **Missings**. Nestes valores ausentes, muitas vezes, pode-se encontrar informações discriminantes valiosas, quando utilizadas em nossos modelos, e portanto, quando modelados estes valores normalmente não são tratados de alguma maneira especial, principalmente utilizando as mais novas ferramentas de boosting, como LightGBM ou CatBoost.
Infelizmente, ao utilizar o Lime, é obrigatória a substituição de todos os valores das variáveis **Missings** por algum número real, fazendo com que parte da explicabilidade do modelo seja perdida, principalmente caso o valor **Missing** tenha algum valor altamente discriminativo na base. Desta forma, o **Toranja Tabular** é uma revisão do Lime, onde pode-se analisar modelos que tenham como entrada essas variáveis (explicativas) sem que haja perda da explicabilidade do modelo. Nele, a explicabilidade dos **Missings** é mostrada em uma coluna apartada, o que permite a devida análise individual dos casos.

Para utilizar uma simples análise de explicabilidade, como no Lime, deve-se seguir os seguintes passos:

```sh
from toranja import toranja
```
A seguir, deve-se colocar a base utilizada no desenvolvimento do modelo (com apenas as colunas que serão utilizadas na escoragem), o modelo(apenas funciona para modelos que possuam a função **predict_proba**), e as colunas categóricas, como índices, utilizadas no modelo (caso o modelo não tenha sido desenvolvido com Dummies/One Hot).

```sh
tor = toranja(df_desenvolvimento,modelo,cat_cols=[0,1,4,5])
```

Para a simples explicabilidade de uma amostra, devemos fazer o seguinte:

```sh
_  = tor.explain_alone(df_desenvolvimento.iloc[20])
```
Como resposta teremos algo como:

![explicabilidade_simples](imagens/explicabilidade_simples.PNG)

Nota-se, no exemplo acima, que o **Toranja Tabular** cria um novo valor terminado em **'_nan'** caso os **Missings** das variáveis explicativas tenham alta discriminação.

## Criação de Grupos de Explicabilidade

É comum que uma análise completa de um modelo demore em geral muitas horas, e caso análises precisem ser feita recorrentemente, é comum que não se tenha uma resposta de prontidão. Deste modo, **Toranja** fornece, com técnicas de agrupamentos de explicabilidade, uma forma simples de se analisar um a possível explicação da resposta do modelo sem que se tenha toda a análise terminada. O agrupamento fará parte da classe criada para explicabilidade do modelo e um relatório em Excel será emitido:

```sh
tor.tabular_analysis(df_desenvolvimento,sample_base=1)
```
Nas figuras abaixos podemos observar como ficariam a análise de explicabilidade:

![grupogeral](imagens/groups_general.jpg)

![grupo1](imagens/group1.JPG)
