import json
import requests


class LLMInterface:
    def generate_json(self, system_prompt, user_prompt):
        raise NotImplementedError


class vLLMInterface(LLMInterface):
    def __init__(self, base_url="http://localhost:8000/v1", model="Qwen/Qwen3.5-9B", enable_thinking=False):
        self.base_url = base_url
        self.model = model
        self.enable_thinking = enable_thinking

    def generate_json(self, system_prompt, user_prompt):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
        }

        response = requests.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()


class LLMwithAPI(LLMInterface):
    def __init__(self, client, model="openrouter/auto"):
        self.client = client
        self.model = model

    def generate_json(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model, temperature=0.0, response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return json.loads(response.choices[0].message.content)
