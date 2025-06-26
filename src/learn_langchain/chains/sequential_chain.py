import os
from logging import info

from langchain.chains import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr


class SeqChain:

    def learn(self):
        llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL"),
                                     api_key=SecretStr(os.getenv("GOOGLE_API_KEY")))

        first_prompt = ChatPromptTemplate.from_template(
            template="What is the one best food to taste in {place}? Just give me the name of the food name."
        )
        first_chain = LLMChain(llm=llm, prompt=first_prompt, output_key="food")

        second_prompt = ChatPromptTemplate.from_template(
            template="What is the one best restaurant to taste {food}? Just give me the name of the restaurant."
        )
        second_chain = LLMChain(llm=llm, prompt=second_prompt, output_key="restaurant")

        third_prompt = ChatPromptTemplate.from_template(
            template="Write a short review about {restaurant} in one line.")
        third_chain = LLMChain(llm=llm, prompt=third_prompt, output_key="review")

        fourth_prompt = ChatPromptTemplate.from_template(
            template="Translate the review to Tamil: {review}. Just translate without explanation."
        )
        fourth_chain = LLMChain(llm=llm, prompt=fourth_prompt, output_key="tamil_review")

        simple_sequential_chain = SequentialChain(chains=[first_chain, second_chain, third_chain, fourth_chain],
                                                  input_variables=["place"],
                                                  output_variables=["food", "restaurant", "review", "tamil_review"],
                                                  verbose=True)
        response = simple_sequential_chain.invoke("Chennai")

        info(f"Response: {response}")
