import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Downloads necessários para português
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # <--- linha adicionada p/ correção erro de compatibilidade


# 1. Base de Dados de Treino (Exemplo simplificado para reprodução)
# Em um cenário real, carregaríamos um CSV com milhares de linhas.
base_treino = [
("O elenco era demais, mas a história não me convenceu.", "neg"),
("Não é que o filme seja ruim, mas achei bem cansativo.", "neg"),
("A fotografia é linda, porém o roteiro é péssimo.", "neg"),
("Gostei da atuação, mas o final deixou a desejar.", "neg"),
("É um filme ok, nem bom, nem ruim.", "neg")
]

# 2. Pré-processamento
stop_words = set(stopwords.words('portuguese'))

def preprocessar(texto):
    # Tokenização e conversão para minúsculas
    palavras = word_tokenize(texto.lower())
    # Remoção de pontuação e stopwords (palavras sem valor semântico: de, o, a, em...)
    filtradas = [p for p in palavras if p.isalpha() and p not in stop_words]
    # Retorna um dicionário: {'filme': True, 'maravilhoso': True}
    return {palavra: True for palavra in filtradas}

# 3. Preparação dos dados para o NLTK
featuresets = [(preprocessar(texto), sentimento) for (texto, sentimento) in base_treino]
random.shuffle(featuresets)

# Treinando o modelo com a base disponível
classifier = NaiveBayesClassifier.train(featuresets)

# 4. EXPERIMENTOS (Testes formais para o relatório)
testes = [
    "O roteiro é muito bom",                     # Óbvio Positivo
    "O filme é terrível e chato",                # Óbvio Negativo
    "Não é que o filme seja ruim, mas cansa",    # Negação (Desafio)
    "Nossa, que filme maravilhoso, dormi todo",  # Sarcasmo (Desafio)
    "O som estava muito alto"                    # Ambiguidade (Desafio)
]

print("--- RESULTADOS DOS EXPERIMENTOS ---")
for frase in testes:
    caracteristicas = preprocessar(frase)
    resultado = classifier.classify(caracteristicas)
    print(f"Frase: '{frase}' -> Sentimento: {resultado}")

# Exibir as palavras que mais influenciam o modelo
print("\n--- PALAVRAS MAIS INFORMATIVAS ---")
classifier.show_most_informative_features(5)