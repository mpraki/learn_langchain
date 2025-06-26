import os
from logging import basicConfig, error, info

from dotenv import load_dotenv, find_dotenv

from learn_langchain import SeqChain, RouterChain
from src.learn_langchain import PromptTemplate, OutputParser, Memory
from src.learn_langchain.chains.simple_chain import SimpleChain
from src.learn_langchain.chains.simple_sequential_chain import SimpleSeqChain


def main():
    load_dotenv(find_dotenv(".env.secrets"))
    load_dotenv(find_dotenv(".env.config"))

    info("******* Prompt Template *******")
    PromptTemplate().learn()

    info("******* Output Parser *******")
    OutputParser().learn()

    info("******* Conversation Chain / Memory *******")
    Memory().learn()

    info("******* Simple Chain *******")
    SimpleChain().learn()

    info("******* Simple Sequential Chain *******")
    SimpleSeqChain().learn()

    info("******* Sequential Chain *******")
    SeqChain().learn()

    info("******* Router Chain *******")
    RouterChain().learn()


def configure_logging():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file = os.path.join(project_root, "learn_langchain.log")
    basicConfig(level="INFO", format='%(asctime)s %(levelname)s %(message)s', filename=log_file, filemode='w')


if __name__ == "__main__":
    configure_logging()
    try:
        info("******* LangChain Learning Application *******")
        main()
    except Exception as e:
        error(f"Error occurred while running the application: {e}", exc_info=True)
