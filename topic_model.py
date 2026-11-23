"""Interpretable Latent Dirichlet Allocation with collapsed Gibbs sampling."""

from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

TOKEN_PATTERN = re.compile(r"[a-z]{2,}")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class TopicSummary:
    topic_id: int
    top_words: tuple[tuple[str, float], ...]


class LDATopicModel:
    """Small educational LDA model trained with collapsed Gibbs sampling."""

    def __init__(self, n_topics: int = 3, iterations: int = 100, alpha: float = 0.1, beta: float = 0.01, seed: int = 7) -> None:
        if n_topics < 1 or iterations < 1:
            raise ValueError("n_topics and iterations must be positive")
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.n_topics, self.iterations = n_topics, iterations
        self.alpha, self.beta, self.seed = alpha, beta, seed

    def fit(self, documents: Sequence[str]) -> "LDATopicModel":
        tokenized = [tokenize(document) for document in documents]
        if not tokenized or any(not tokens for tokens in tokenized):
            raise ValueError("documents must be non-empty and contain alphabetic tokens")
        vocabulary = sorted(set().union(*tokenized))
        self.vocabulary_ = vocabulary
        self.word_to_id_ = {word: index for index, word in enumerate(vocabulary)}
        encoded = [[self.word_to_id_[word] for word in tokens] for tokens in tokenized]
        self.n_documents_, self.n_words_ = len(encoded), len(vocabulary)
        self.doc_topic_counts_ = [[0] * self.n_topics for _ in encoded]
        self.topic_word_counts_ = [[0] * self.n_words_ for _ in range(self.n_topics)]
        self.topic_counts_ = [0] * self.n_topics
        rng = random.Random(self.seed)
        assignments = []
        for doc_id, words in enumerate(encoded):
            doc_assignments = []
            for word_id in words:
                topic = rng.randrange(self.n_topics)
                doc_assignments.append(topic)
                self._increment(doc_id, word_id, topic, 1)
            assignments.append(doc_assignments)

        for _ in range(self.iterations):
            for doc_id, words in enumerate(encoded):
                for position, word_id in enumerate(words):
                    old_topic = assignments[doc_id][position]
                    self._increment(doc_id, word_id, old_topic, -1)
                    weights = [
                        (self.doc_topic_counts_[doc_id][topic] + self.alpha)
                        * (self.topic_word_counts_[topic][word_id] + self.beta)
                        / (self.topic_counts_[topic] + self.n_words_ * self.beta)
                        for topic in range(self.n_topics)
                    ]
                    total = sum(weights)
                    draw = rng.random() * total
                    cumulative = 0.0
                    new_topic = self.n_topics - 1
                    for topic, weight in enumerate(weights):
                        cumulative += weight
                        if draw <= cumulative:
                            new_topic = topic
                            break
                    assignments[doc_id][position] = new_topic
                    self._increment(doc_id, word_id, new_topic, 1)
        return self

    def _increment(self, doc_id: int, word_id: int, topic: int, amount: int) -> None:
        self.doc_topic_counts_[doc_id][topic] += amount
        self.topic_word_counts_[topic][word_id] += amount
        self.topic_counts_[topic] += amount

    def document_topic_distribution(self, document_id: int) -> list[float]:
        if not hasattr(self, "doc_topic_counts_"):
            raise ValueError("fit must be called before inference")
        if not 0 <= document_id < self.n_documents_:
            raise IndexError("document_id out of range")
        total = sum(self.doc_topic_counts_[document_id]) + self.n_topics * self.alpha
        return [(count + self.alpha) / total for count in self.doc_topic_counts_[document_id]]

    def topic_summaries(self, top_n: int = 5) -> list[TopicSummary]:
        if not hasattr(self, "topic_word_counts_"):
            raise ValueError("fit must be called before summaries")
        if top_n < 1:
            raise ValueError("top_n must be positive")
        summaries = []
        for topic, counts in enumerate(self.topic_word_counts_):
            denominator = self.topic_counts_[topic] + self.n_words_ * self.beta
            ranked = sorted(
                ((self.vocabulary_[word_id], (count + self.beta) / denominator) for word_id, count in enumerate(counts)),
                key=lambda item: (-item[1], item[0]),
            )[:top_n]
            summaries.append(TopicSummary(topic, tuple(ranked)))
        return summaries

    def perplexity(self, documents: Sequence[str]) -> float:
        """Approximate corpus perplexity using fitted topic-word probabilities."""
        if not hasattr(self, "topic_word_counts_"):
            raise ValueError("fit must be called before perplexity")
        log_probability, token_count = 0.0, 0
        for document in documents:
            for word in tokenize(document):
                if word not in self.word_to_id_:
                    continue
                word_id = self.word_to_id_[word]
                probability = sum(
                    self.document_topic_distribution(0)[topic]
                    * (self.topic_word_counts_[topic][word_id] + self.beta)
                    / (self.topic_counts_[topic] + self.n_words_ * self.beta)
                    for topic in range(self.n_topics)
                )
                log_probability += math.log(max(probability, 1e-12))
                token_count += 1
        return math.exp(-log_probability / token_count) if token_count else float("inf")


def load_documents(path: str | Path) -> list[str]:
    documents = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not documents:
        raise ValueError("document file contains no documents")
    return documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one document per line")
    parser.add_argument("--topics", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--top-words", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    documents = load_documents(args.input)
    model = LDATopicModel(args.topics, args.iterations).fit(documents)
    for summary in model.topic_summaries(args.top_words):
        words = ", ".join(f"{word} ({probability:.3f})" for word, probability in summary.top_words)
        print(f"Topic {summary.topic_id}: {words}")
    for index in range(min(3, len(documents))):
        distribution = ", ".join(f"{value:.3f}" for value in model.document_topic_distribution(index))
        print(f"Document {index + 1} topic distribution: [{distribution}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
