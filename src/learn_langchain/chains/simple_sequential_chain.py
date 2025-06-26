import os
from logging import info

from langchain.chains import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr


class SimpleSeqChain:

    def learn(self):
        llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL"),
                                     api_key=SecretStr(os.getenv("GOOGLE_API_KEY")))

        first_prompt = PromptTemplate(
            input_variables=["place"],
            template="What is the one best food to taste in {place}? Just give me the name of the food name.",
        )
        first_chain = LLMChain(llm=llm, prompt=first_prompt)

        second_prompt = PromptTemplate(
            input_variables=["food"],
            template="What is the one best restaurant to taste {food}? Just give me the name of the restaurant.",
        )
        second_chain = LLMChain(llm=llm, prompt=second_prompt)

        simple_sequential_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=True)
        response = simple_sequential_chain.invoke("Chennai")
        info(f"Response: {response}")
        assert response.get("input") == "Chennai" and response.get("output") is not None
