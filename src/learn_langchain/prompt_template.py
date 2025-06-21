from logging import info

from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate


class PromptTemplate:

    def learn(self):
        model = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")
        prompt_template = ChatPromptTemplate.from_template(self.__prompt_template())
        prompt = "Hello, How are you?"
        response = model.invoke(prompt_template.format_prompt(text=prompt, language="Tamil"))
        info(response.content)
        response = model.invoke(prompt_template.format_prompt(text=prompt, language="Malayalam"))
        info(response.content)
        response = model.invoke(prompt_template.format_prompt(text=prompt, language="Marathi"))
        info(response.content)

    @staticmethod
    def __prompt_template() -> str:
        return "Translate the text that is delimited by triple backticks to {language}. I do not want the breakdown. text: ```{text}```"
