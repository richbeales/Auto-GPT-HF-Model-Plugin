import pytest
from typing import TypeVar

from hf_model_chat_plugin.huggingface import (
    HuggingFaceFreeInterenceModel,
    HuggingFaceHostedInferenceModel,
    HuggingFaceLocalModel,
)

PromptGenerator = TypeVar("PromptGenerator")


class TestHuggingFace():
    def test_hosted_inference(self):
        hf = HuggingFaceHostedInferenceModel()
        completion = hf.get_completion(
            "stabilityai/stablelm-tuned-alpha-3b",
            "Can you please let us know more details about your ",
            "",
            temperature=0.5,
            max_tokens=200,
        )
        print(completion)
        assert len(completion) > 0

    # def test_local(self):
    #    hf = HuggingFaceLocalModel()
    #    completion = hf.get_completion(
    #        "stabilityai/stablelm-tuned-alpha-3b",
    #        "Can you please let us know more details about your ",
    #        "",
    #        temperature=0.5,
    #        max_tokens=200
    #    )
    #    print(completion)
    #    assert len(completion) > 0


    # def test_free(self):
    #    hf = HuggingFaceFreeInterenceModel()
    #    completion = hf.get_completion(
    #        "stabilityai/stablelm-tuned-alpha-3b",
    #        "Can you please let us know more details about your ",
    #        "",
    #        temperature=0.5,
    #        max_tokens=200
    #    )
    #    print(completion)
    #    assert len(completion) > 0

