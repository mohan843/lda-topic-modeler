# LDA Topic Modeler

An interpretable Natural Language Processing mini-project that discovers latent topics in a collection of documents using Latent Dirichlet Allocation with collapsed Gibbs sampling.

## Overview

Topic modeling is an unsupervised technique for organizing a document collection when labels are unavailable. This project implements the core LDA sampling process from scratch and exposes topic-word distributions, document-topic mixtures, and an approximate perplexity metric.

## Problem statement

Given unlabeled text documents, infer a configurable number of latent topics and summarize each document by the proportion of its words assigned to every topic.

## Features

- Collapsed Gibbs-sampling LDA implementation.
- Configurable topic count, iterations, Dirichlet priors, and random seed.
- Top-word summaries for each discovered topic.
- Document-topic probability distributions.
- Approximate corpus perplexity.
- One-document-per-line input format and CLI.
- Unit tests for tokenization, training, distributions, perplexity, loading, and validation.

## Technology stack

- Python 3.10+
- Natural language processing and unsupervised learning
- Latent Dirichlet Allocation
- Collapsed Gibbs sampling
- Python `random`, `re`, `dataclasses`, `argparse`, and `unittest`
- Standard library only

## Installation

```bash
git clone https://github.com/mohan843/lda-topic-modeler.git
cd lda-topic-modeler
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party dependencies are required.

## Input format

Provide a UTF-8 text file with one document per line. The included sample contains technology, sports, and cooking documents.

## Execution

```bash
python topic_model.py data/documents.txt --topics 3 --iterations 150 --top-words 5
```

Use the Python API:

```python
from topic_model import LDATopicModel

model = LDATopicModel(n_topics=3, iterations=150, seed=7).fit(documents)
print(model.topic_summaries(top_n=8))
print(model.document_topic_distribution(0))
```

## Sample output

```text
Topic 0: dinner (0.109), food (0.109), ingredient (0.109), kitchen (0.109), recipe (0.109)
Topic 1: code (0.109), data (0.109), model (0.109), python (0.109), algorithm (0.055)
Topic 2: player (0.173), team (0.173), league (0.116), match (0.116), basketball (0.058)
Document 1 topic distribution: [0.016, 0.968, 0.016]
```

Because LDA is stochastic and topic identities are exchangeable, topic numbering and exact probabilities can vary with the seed, corpus, and iteration count.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Interpretation

A topic is represented by words with high estimated probability within that topic. A document-topic distribution indicates how strongly each topic explains a document. Topic labels are not supplied by the algorithm; a human should inspect top words and assign meaningful names.

## Limitations and responsible use

LDA uses a bag-of-words representation and ignores word order, syntax, and context. Short or highly repetitive documents can produce unstable topics. Topic models can expose sensitive themes, so corpus access, preprocessing, and downstream use should respect privacy and fairness requirements.

## Future improvements

- Add stop-word filtering and configurable tokenization.
- Support sparse matrices and larger corpora.
- Add topic coherence scoring and held-out perplexity.
- Export topic-word and document-topic matrices as JSON.
- Compare LDA with NMF, BERTopic, and contextual embeddings.
