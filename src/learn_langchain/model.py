import os

from langchain.chat_models import init_chat_model
from pydantic import SecretStr


class Model:
    model = None

    def __init__(self):
        model_name = os.getenv("MODEL")
        model_provider = os.getenv("MODEL_PROVIDER")
        self.model = init_chat_model(model=model_name, model_provider=model_provider,
                                     api_key=SecretStr(os.getenv("GOOGLE_API_KEY")))

    def get_model(self):
        return self.model

    def invoke(self, prompt):
        return self.model.invoke(prompt)
