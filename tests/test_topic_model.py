import unittest

from topic_model import LDATopicModel, load_documents, tokenize


class LDATests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            "python code model data",
            "software code server python",
            "football team player match",
            "soccer team player goal",
            "recipe kitchen food dinner",
            "baking recipe flour dessert",
        ]

    def test_tokenizer_is_case_insensitive(self):
        self.assertEqual(tokenize("Python, MODEL!"), ["python", "model"])

    def test_fit_creates_topic_summaries(self):
        model = LDATopicModel(n_topics=3, iterations=50, seed=3).fit(self.documents)
        summaries = model.topic_summaries(top_n=3)
        self.assertEqual(len(summaries), 3)
        self.assertTrue(all(len(summary.top_words) == 3 for summary in summaries))

    def test_document_distribution_sums_to_one(self):
        model = LDATopicModel(n_topics=3, iterations=50).fit(self.documents)
        distribution = model.document_topic_distribution(0)
        self.assertEqual(len(distribution), 3)
        self.assertAlmostEqual(sum(distribution), 1.0)

    def test_perplexity_is_finite(self):
        model = LDATopicModel(n_topics=3, iterations=30).fit(self.documents)
        self.assertGreater(model.perplexity(self.documents), 0)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            LDATopicModel(n_topics=0)
        with self.assertRaises(ValueError):
            LDATopicModel().fit(["", ""])
        with self.assertRaises(IndexError):
            LDATopicModel().fit(self.documents).document_topic_distribution(99)

    def test_document_loader(self):
        documents = load_documents("data/documents.txt")
        self.assertEqual(len(documents), 9)


if __name__ == "__main__":
    unittest.main()
