# FeelAnalysis
Performs an extraction of information via Twitter, with the police theme in order to perform the analysis of feelings.

Tecnicas utilizadas

stopwords:
Método consiste em remover palavras muito frequentes.
“a”, “de”, “o”, “da”, “que”,”e”,”do”

Stemming: 
Essa técnica consiste em reduzir o termo ao seu radical.
palavra “frequentemente” após esse processo se torna “frequent”, a palavra “copiar” após esse processo se torna “copi”

BagWords: 
É representado como um multiconjunto de suas palavras (o "saco"), desconsiderando a estrutura gramatical e até mesmo a ordenação delas, mas mantendo sua multiplicidade.

Cross Validation: 
Esta consiste em dividir todo o dado em K partes, essas partes que se chamam folds. Dessas partes uma será separada para teste e as outras restantes serão usadas para treinar o modelo. Isso é feito repetidamente até que o modelo seja treinado e testado com todas as partes (folds).

Métricas de Classificação

Precisão (precision) é calculada da seguinte forma:
precision = true positive / (true positive + false positive)

Isso significa o número de vezes que uma classe foi predita corretamente dividida pelo número de vezes que a classe foi predita.

Por exemplo, o número de vezes que a classe Positivo foi classificada corretamente dividido pelo número de classes classificadas como Positivo.

Revocação (recall) é calculada da seguinte forma:

recall = true positive / (true positive + false negative)

Isso significa o número de vezes que uma classe foi predita corretamente (TP) dividido pelo número de vezes que a classe aparece no dado de teste (FN).

Por exemplo, o número de vezes que a classe Positivo foi predita corretamente dividido pelo número de classes Positivo que contém no dado de teste.

Bigrams
Essa modelagem consiste em passar duas palavras como features para o classificador ao invés de apenas uma.
Dessa forma, estamos dizendo que uma palavra tem uma relação com outra palavra, veja um exemplo:

Na frase: “Eu não gosto desse governo” na modelagem inicial passamos para o modelo cada palavra sendo uma feature, ficaria = {eu, não, gosto,desse, governo}

Usando Bigrams, passaríamos para o modelo 2 palavras, veja:
{eu não, não gosto, gosto desse, desse governo}
