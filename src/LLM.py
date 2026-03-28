import requests


class LLMInterface:
    def generate_response(self, user_prompt, system_prompt=""):
        raise NotImplementedError


class vLLMInterface(LLMInterface):
    def __init__(self, model="Qwen/Qwen3.5-9B", base_url="http://localhost:8000/v1", **kwargs):
        print(f"\tLLM: {model} at {base_url}")
        self.base_url = base_url
        self.model = model
        self.kwargs = kwargs

    def generate_response(self, user_prompt, system_prompt=""):
        messages = [{"role": "user", "content": user_prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        payload = {
            "model": self.model, "temperature": self.kwargs.get('temperature', 0.5),
            "messages": messages, "stream": self.kwargs.get('stream', False),
            "chat_template_kwargs": {
                "enable_thinking": self.kwargs.get('enable_thinking', False)
            },
        }

        response = requests.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']


class LLMwithAPIKEY(LLMInterface):
    def __init__(self, client, model="openrouter/auto"):
        self.client = client
        self.model = model

    def generate_response(self, user_prompt, system_prompt=""):
        response = self.client.chat.completions.create(
            model=self.model, temperature=0.0, response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.choices[0].message.content
