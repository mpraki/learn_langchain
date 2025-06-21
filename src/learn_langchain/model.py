import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


class Model:
    model = None

    def __init__(self):
        load_dotenv(dotenv_path=".env.config")
        model_name = os.getenv("MODEL")
        model_provider = os.getenv("MODEL_PROVIDER")
        self.model = init_chat_model(model=model_name, model_provider=model_provider)

    def getModel(self):
        return self.model

    def invoke(self, prompt):
        return self.model.invoke(prompt)
