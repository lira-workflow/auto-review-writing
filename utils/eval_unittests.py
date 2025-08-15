# Unit Testing the metrics to ensure they work properly
import unittest
from eval_metrics.recall import (
    card,
    heading_soft_recall,
    heading_entity_recall,
    extract_entities_from_list,
    article_entity_recall,
)
from eval_metrics.rouge import rouge_eval


class TestFunctions(unittest.TestCase):

    def test_card(self):
        result = card(("text1", "text2"))
        self.assertAlmostEqual(
            result,
            1.095,
            places=3,
            msg="Cardinality computation failed for orthogonal vectors.",
        )

    def test_heading_soft_recall(self):
        gold_headings = ["Heading1", "Heading2"]
        pred_headings = ["Heading2"]
        result = heading_soft_recall(gold_headings, pred_headings)
        self.assertAlmostEqual(
            result,
            0.916,
            places=3,
            msg="Soft recall should be close to one with these overlapping headings.",
        )

    def test_extract_entities_from_list(self):
        result = extract_entities_from_list(
            [
                "I would like to mention two things: Entity1 and Flower.",
                "My pet Iguana is happy.",
            ]
        )
        result.sort()
        self.assertEqual(
            result,
            ["entity1", "flower", "happy", "my pet iguana"],
            "Entity extraction failed.",
        )

    def test_extract_entities_from_list_2(self):
        # This test is aimed at checking the NER capabilities for more obscure entities
        # (i.e., medical terms. The example here is adapted from medSpacy)
        result = extract_entities_from_list(
            [
                "Patient has hx of stroke. Mother diagnosed with diabetes. No evidence of pna."
            ]
        )
        result.sort()
        self.assertEqual(
            result,
            ["diabetes", "diagnosed", "mother", "patient", "pna", "stroke"],
            "Entity extraction failed.",
        )

    def test_heading_entity_recall(self):
        gold_headings = ["Heading with Entity1"]
        pred_headings = ["Another heading with Entity1"]
        result = heading_entity_recall(
            gold_headings=gold_headings, pred_headings=pred_headings
        )
        self.assertAlmostEqual(
            result, 1.0, "Entity recall computation failed when entities match."
        )

    def test_heading_entity_recall_2(self):
        gold_entities = ["PersonX", "Mamba", "Zinc"]
        pred_entities = ["PersonY", "Cat"]
        result = heading_entity_recall(gold_entities, pred_entities)
        self.assertAlmostEqual(
            result, 0.0, "Entity recall should be zero when no entities match."
        )

    def test_article_entity_recall(self):
        gold_article = "This article contains Entity1."
        pred_article = "Another article with Entity1."
        result = article_entity_recall(
            gold_article=gold_article, pred_article=pred_article
        )
        self.assertAlmostEqual(
            result,
            1.0,
            "Entity recall computation failed for articles with matching entities.",
        )

    def test_compute_rouge_scores(self):
        gold_answer = ["This is a sample answer."]
        pred_answer = ["This is a sample answer."]
        result = rouge_eval(gold_answer, pred_answer, n_jobs=1)["mean"]
        self.assertAlmostEqual(
            result["ROUGE1"],
            1.0,
            "ROUGE1 score should be 1.0 for identical text.",
        )
        self.assertAlmostEqual(
            result["ROUGE2"],
            1.0,
            "ROUGE2 score should be 1.0 for identical text.",
        )
        self.assertAlmostEqual(
            result["ROUGEL"],
            1.0,
            "ROUGEL score should be 1.0 for identical text.",
        )

    def test_compute_rouge_scores_2(self):
        gold_answer = ["Writing is fun and easy."]
        pred_answer = ["Writing is easy."]
        result = rouge_eval(gold_answer, pred_answer, n_jobs=1)["mean"]

        self.assertAlmostEqual(
            result["ROUGE1"],
            0.75,
            "ROUGE1 score should be around 0.75.",
        )
        self.assertAlmostEqual(
            result["ROUGE2"],
            0.333,
            "ROUGE2 score should be around 0.333.",
        )
        self.assertAlmostEqual(
            result["ROUGEL"],
            0.75,
            "ROUGEL score should be 0.75 for identical text.",
        )


if __name__ == "__main__":

    print("Running Unit Tests for ROUGE and Heading Recall..\n")
    unittest.main()
