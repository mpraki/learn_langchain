import os
from logging import info

from langchain.chains.llm import LLMChain
from langchain.chains.router import MultiPromptChain, LLMRouterChain
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr


class RouterChain:

    def learn(self):
        llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL"),
                                     api_key=SecretStr(os.getenv("GOOGLE_API_KEY")))

        prompt_1 = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert on animals."),
                ("human", "{input}"),
            ]
        )
        chain_1 = LLMChain(llm=llm, prompt=prompt_1)

        prompt_2 = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert on vegetables."),
                ("human", "{input}"),
            ]
        )
        chain_2 = LLMChain(llm=llm, prompt=prompt_2)

        destination_chains = {
            "animals": chain_1,
            "vegetables": chain_2
        }

        destinations = ", ".join(destination_chains.keys())
        info(f"destinations - {destinations}")

        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=destinations
        )

        info(f"router_template - {router_template}")

        router_prompt = PromptTemplate(template=router_template, input_variables=["input"],
                                       output_parser=RouterOutputParser())
        router_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt)

        default_prompt = ChatPromptTemplate.from_template("{input}")
        default_chain = LLMChain(llm=llm, prompt=default_prompt)

        chain = MultiPromptChain(router_chain=router_chain,
                                 destination_chains=destination_chains,
                                 default_chain=default_chain,
                                 verbose=True)

        response = chain.invoke("What is the color of black cat? :)")
        info(f"Response: {response}")

        response2 = chain.invoke("What is the color of red chilli? :)")
        info(f"Response: {response2}")

        response3 = chain.invoke("How long is the Marina beach?")
        info(f"Response: {response3}")
