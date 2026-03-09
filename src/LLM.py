import json
import requests


class LLMInterface:
    def generate_json(self, system_prompt, user_prompt):
        raise NotImplementedError


class OllamaLLM(LLMInterface):
    def __init__(self, model="qwen2.5:7b", url="http://localhost:11434/api/chat"):
        self.model = model
        self.url = url

    def generate_json(self, system_prompt, user_prompt):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
        }

        r = requests.post(self.url, json=payload)
        content = r.json()["message"]["content"]

        return json.loads(content)


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
