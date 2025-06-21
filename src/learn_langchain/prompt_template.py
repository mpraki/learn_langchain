from logging import info

from langchain.prompts import ChatPromptTemplate

from src.learn_langchain.model import Model


class PromptTemplate:

    def learn(self):
        prompt_template = ChatPromptTemplate.from_template(self.__prompt_template())
        prompt = "Hello, How are you?"
        response = Model().invoke(prompt_template.format_prompt(text=prompt, language="Tamil", style="sarcastic"))
        info(response.content)
        response = Model().invoke(prompt_template.format_prompt(text=prompt, language="Tamil", style="formal"))
        info(response.content)
        response = Model().invoke(prompt_template.format_prompt(text=prompt, language="Tamil", style="informal"))
        info(response.content)

    @staticmethod
    def __prompt_template() -> str:
        return "Translate the text that is delimited by triple backticks to {language} in {style} way. I do not want the breakdown. text: ```{text}```"
