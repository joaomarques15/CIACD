1. Importações e Dependências
O código utiliza a biblioteca nltk (Natural Language Toolkit), a mais popular para NLP em Python.

word_tokenize: Divide uma frase em unidades menores (palavras/tokens).

stopwords: Carrega palavras irrelevantes (artigos, preposições) que serão descartadas.

NaiveBayesClassifier: O "cérebro" do modelo. É um classificador probabilístico baseado no Teorema de Bayes, que calcula a chance de uma frase ser positiva ou negativa com base nas palavras que ela contém.

2. O Fluxo de Funcionamento
O script segue o pipeline padrão de ciência de dados:

A. Base de Treino
Define uma lista de tuplas (texto, rótulo). O modelo "aprende" que palavras como "maravilhoso" estão associadas a pos (positivo) e "péssimo" a neg (negativo).

B. Função de Pré-processamento (preprocessar)
Esta é a parte mais crítica. Para que o computador entenda o texto, ele precisa ser limpo:

Lower case: Transforma tudo em minúsculo para que "Filme" e "filme" sejam lidos como a mesma palavra.

Tokenização: Separa a string em uma lista de palavras.

Filtragem (Alpha & Stopwords):

p.isalpha(): Remove números e pontuações (como "!", ",").

p not in stop_words: Remove palavras que não trazem sentimento (como "de", "o", "a").

Dicionário de Features: Transforma a lista em um formato que o NLTK aceita: {palavra: True}.

C. Treinamento
O modelo percorre a base_treino, conta a frequência das palavras em cada categoria e cria uma tabela de probabilidades.

D. Testes e Experimentos
O script submete frases novas (que não estavam no treino) ao classificador para ver como ele se comporta. Aqui são testados desafios reais:

Sarcasmo: "Maravilhoso, dormi todo" (O modelo provavelmente errará aqui, pois lerá "maravilhoso" como positivo).

Negação: "Não é... ruim" (O modelo pode se confundir com a palavra "ruim").
