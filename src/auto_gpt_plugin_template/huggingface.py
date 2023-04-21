import dotenv
import requests
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


class HuggingFaceModel:
    def make_conversation(self, prompt) -> str:
        hf_prompt = ""
        for message in prompt:
            if message["role"] == "user":
                hf_prompt += "<|SYSTEM|>" + message["content"]
            elif message["role"] == "assistant":
                hf_prompt += "<|USER|>" + message["content"]
            elif message["role"] == "system":
                hf_prompt += "<|USER|>" + message["content"]
            else:
                hf_prompt += "<|USER|>" + message["content"]
        return hf_prompt

    def get_completion(
        self,
        model: str,
        conv,
        last_message: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {"inputs": conv}
        response = self.query(model, payload, temperature, max_tokens)
        print(response)
        try:
            return response[0]["generated_text"]
        except:
            return ""


class HuggingFaceHostedInferenceModel(HuggingFaceModel):
    def query(self, model, payload: dict, temperature, max_tokens) -> dict:
        headers = {
            "Authorization": "Bearer " + dotenv.get_key(".env", "HUGGINGFACE_TOKEN")
        }
        host = dotenv.get_key(".env", "HUGGINGFACE_HOSTED_URL")
        if host:
            payload["options"] = {"use_cache": False, "wait_for_model": True}
            payload["parameters"] = {
                "top_p": 1.0,
                "temperature": temperature,
                "max_length": max_tokens,
                "return_full_text": True,
            }
            response = requests.post(host, headers=headers, json=payload)
            return response.json()
        else:
            return [{"generated_text": "Missing Hostname!"}]


class HuggingFaceFreeInterenceModel(HuggingFaceModel):
    API_URL = "https://api-inference.huggingface.co/models/"

    def query(self, model, payload: dict, temperature, max_tokens) -> dict:
        # return [{"generated_text": "Bypassing free API for now"}]
        headers = {
            "Authorization": "Bearer " + dotenv.get_key(".env", "HUGGINGFACE_TOKEN")
        }
        payload["options"] = {"use_cache": False, "wait_for_model": True}
        response = requests.post(self.API_URL + model, headers=headers, json=payload)
        return response.json()


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class HuggingFaceLocalModel(HuggingFaceModel):
    def query(self, model, prompt, temperature, max_tokens):
        return ""  # CUDA issue on my machine - skip for now

        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model)
        model.half().cuda()

        system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
        - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
        - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
        - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
        - StableLM will refuse to participate in anything that could harm a human.
        """

        prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
        )
        print(tokenizer.decode(tokens[0], skip_special_tokens=True))
        return tokenizer.decode(tokens[0], skip_special_tokens=True)
