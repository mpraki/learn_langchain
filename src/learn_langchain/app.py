from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


def main():
    load_dotenv(verbose=True)
    model = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")
    response = model.invoke("What is the capital of France?")
    print(response.content)


if __name__ == "__main__":
    main()
