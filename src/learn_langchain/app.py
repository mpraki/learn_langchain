import os
from logging import basicConfig, error

from dotenv import load_dotenv

from src.learn_langchain import PromptTemplate, OutputParser


def main():
    load_dotenv(verbose=True)
    PromptTemplate().learn()
    OutputParser().learn()


def configure_logging():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file = os.path.join(project_root, "learn_langchain.log")
    basicConfig(level="INFO", format='%(asctime)s %(levelname)s %(message)s', filename=log_file, filemode='a')


if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except Exception as e:
        error(f"Error occurred while running the application: {e}", exc_info=True)
