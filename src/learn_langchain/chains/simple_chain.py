import os
from logging import info

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr


class SimpleChain:

    def learn(self):
        llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL"),
                                     api_key=SecretStr(os.getenv("GOOGLE_API_KEY")))

        prompt = PromptTemplate(
            input_variables=["place"],
            template="Best food to taste in {place}?",
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        info(chain.invoke("Chennai"))
