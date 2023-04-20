from typing import TypeVar
import unittest
from src.huggingface import (
    HuggingFaceLocalModel,
    HuggingFaceHostedInferenceModel,
    HuggingFaceFreeInterenceModel,
)

PromptGenerator = TypeVar("PromptGenerator")


class Test(unittest.TestCase):
    def test_inference(self):
        hf = HuggingFaceHostedInferenceModel()
        completion = hf.get_completion(
            "stabilityai/stablelm-tuned-alpha-3b",
            "Can you please let us know more details about your ",
            "",
        )
        print(completion)
        assert unittest.TestCase.assertIsNotNone(self, completion, "Completion is None")

    def test_local(self):
        hf = HuggingFaceLocalModel()
        completion = hf.get_completion(
            "stabilityai/stablelm-tuned-alpha-3b",
            "Can you please let us know more details about your ",
            "",
        )
        print(completion)
        assert unittest.TestCase.assertIsNotNone(self, completion, "Completion is None")

    def test_free(self):
        hf = HuggingFaceFreeInterenceModel()
        completion = hf.get_completion(
            "stabilityai/stablelm-tuned-alpha-3b",
            "Can you please let us know more details about your ",
            "",
        )
        print(completion)
        assert unittest.TestCase.assertIsNotNone(self, completion, "Completion is None")
