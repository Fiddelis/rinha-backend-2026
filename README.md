# rinha-backend-2026

Backend para o desafio da Rinha 2026 que expoe `POST /fraud-score` e toma decisao de aprovação a partir de busca vetorial aproximada sobre transacoes historicas rotuladas.

A ideia central deste projeto foi fugir de uma abordagem pesada em tempo de requisição. Em vez de depender de um modelo grande ou de uma pipeline complexa de inferencia online, a aplicação transforma cada transação em um vetor numerico compacto, consulta uma base previamente organizada e devolve um score a partir dos vizinhos mais proximos. O foco desde o inicio foi manter latencia baixa, consumo de memoria controlado e comportamento previsivel mesmo em um cenario apertado de CPU.

## Visao geral

Quando uma requisição chega em `POST /fraud-score`, o JSON e convertido com `msgspec` para uma estrutura tipada, sem passar por uma camada mais custosa de validação generica. A partir dali, o `Vectorizer` monta um vetor `float32` com 14 features normalizadas que representam o contexto daquela compra. Esse vetor e enviado para o `VectorSearch`, que tenta primeiro um caminho rapido com uma arvore de alta confianca. Quando esse atalho nao consegue tomar uma decisao suficientemente segura, a consulta segue para o indice IVF, onde a busca aproximada recupera os vizinhos mais relevantes. O `fraud_score` final nasce da media dos rotulos desses vizinhos, e a resposta considera a transação aprovada quando `fraud_score < 0.6`.

Exemplo de resposta:

```json
{
  "approved": true,
  "fraud_score": 0.2
}
```

## Arquitetura

Arquivos principais:

- [src/main.py](/Users/fiddelis/Projects/rinha-backend-2026/src/main.py): API Starlette, schema de entrada, warmup e endpoints.
- [src/vectorizer.py](/Users/fiddelis/Projects/rinha-backend-2026/src/vectorizer.py): engenharia de features e normalização.
- [src/search.py](/Users/fiddelis/Projects/rinha-backend-2026/src/search.py): busca vetorial, fast paths e estrategia de scoring.
- [src/index_utils.py](/Users/fiddelis/Projects/rinha-backend-2026/src/index_utils.py): quantização e treino da arvore de alta confianca.
- [src/loader.py](/Users/fiddelis/Projects/rinha-backend-2026/src/loader.py): carga dos artefatos com `numpy.load` e `mmap`.

## O que foi feito para performance

Grande parte do trabalho aqui foi concentrada em tirar custo do caminho quente da requisição. O primeiro passo foi tornar a entrada barata: o parse e feito com `msgspec`, o que reduz alocacoes e evita o custo de estruturas genericas demais para um contrato que ja e conhecido. Em seguida, a aplicação converte tudo para um vetor de tamanho fixo, o que torna a inferencia basicamente uma combinação de transformacoes simples e calculo vetorial com `numpy`.

Outro ponto importante foi introduzir um fast path de decisao. Antes de consultar a base vetorial, a aplicação percorre uma arvore binaria treinada offline. Quando a folha encontrada representa um caso muito confiavel, a resposta sai dali mesmo. Isso ajuda a derrubar o custo medio por request, porque uma parte dos casos nem precisa passar pela busca ANN.

Quando a arvore nao fecha a conta, entra o IVF. Em vez de comparar a query com todos os vetores conhecidos, o sistema compara primeiro com os centroides e visita apenas as listas invertidas mais promissoras. O efeito pratico disso e simples: reduzir bastante a quantidade de comparacoes necessarias sem abrir mao de um criterio consistente para recuperar vizinhos proximos.

A busca tambem foi desenhada para ser economica em memoria. O caminho principal opera sobre vetores quantizados em `uint8`, o que diminui footprint e barateia a etapa inicial de distancia. Depois disso, quando faz sentido, um conjunto pequeno de candidatos e reranqueado em `float16`, recuperando precisao onde ela realmente importa. Essa separação entre uma fase barata de seleção e uma fase curta de refinamento foi uma das decisoes mais relevantes do projeto.

No build do indice, os vetores sao reorganizados fisicamente por cluster. Isso melhora a localidade de memoria e torna a leitura mais sequencial em runtime, o que combina melhor com a forma como a busca percorre as listas do IVF. Somado a isso, os artefatos sao carregados com `mmap` sempre que possivel, evitando inflar a heap logo no startup e permitindo um uso mais controlado dos arrays.

Tambem houve um cuidado com o boot e com o ambiente de execução. A aplicação faz warmup explicito no `lifespan`, tocando os metadados principais antes de entrar em produção, e a imagem final do container copia apenas o necessario para servir a inferencia. No deploy usado aqui, duas instancias da aplicação ficam atras de um `nginx` com keepalive, o que ajuda a distribuir carga de forma simples e barata.

## Tecnicas usadas

As principais tecnicas aplicadas neste backend sao:

- vetorização manual de features tabulares;
- normalização e clamp para estabilizar distancia euclidiana;
- busca ANN com IVF;
- quantização `float32 -> uint8`;
- reranking em `float16`;
- arvore de decisao binaria como fast path de alta confianca;
- memory mapping com `numpy`;
- processamento em lote no build do indice;
- reorganização dos dados por cluster para melhorar localidade.

## Pipeline offline

O backend depende de artefatos precomputados em `resources/`. A parte mais custosa do trabalho fica fora do caminho online e e resolvida antes do deploy.

### `scripts/build_index.py`

Le `resources/references.json` e gera a base vetorial principal, alem das versoes compactadas e da arvore usada no fast path:

- `vectors.npy`
- `vectors_f16.npy`
- `vectors_q8.npy`
- `labels.npy`
- `norms.npy`
- `tree.npz`

### `scripts/build_ivf.py`

Constroi o IVF a partir dos vetores base:

- treina centroides com `KMeans` sobre amostra;
- refina atribuicoes em batches;
- reordena vetores e labels por cluster;
- grava `centroids.npy`, `cluster_indices.npy` e `cluster_offsets.npy`.

### `scripts/build_quantized.py`

Regenera:

- `vectors_f16.npy`
- `vectors_q8.npy`
- `tree.npz`

### `scripts/rebuild_resources.sh`

Script utilitario para reconstruir apenas o necessario, com suporte a:

- `--force-full`
- `--force-quantized`
- `--force-ivf`
