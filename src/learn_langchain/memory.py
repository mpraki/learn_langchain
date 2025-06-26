from logging import info

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory

from src.learn_langchain.model import Model


class Memory:

    def learn(self):
        model = Model().get_model()

        info("\n ***** Buffer Memory *****")

        buffer_memory = ConversationBufferMemory()
        buffer_memory.save_context({"input": "My name is Prakash Mani!"}, {"output": "Hello Prakash Mani!"})
        conversation = ConversationChain(llm=model, memory=buffer_memory, verbose=False)
        info(conversation.predict(
            input="What are the months the rain can be expected in Attur, Salem? Just give me the months."))
        response = conversation.predict(input="Do you know me?")
        info(response)  # expected: LLM should know the name, as the memory is not limited
        assert 'Prakash Mani' in response, "'Prakash Mani' not found in response"

        info("\n ***** Window Memory *****")

        window_memory = ConversationBufferWindowMemory(k=1)  # k=1 means only the last conversation will be remembered
        window_memory.save_context({"input": "I live in Attur, Salem."}, {"output": "Attur, Salem is a nice place!"})
        window_memory.save_context({"input": "I visited India, UAE, Netherlands, Belgium and France."},
                                   {"output": "Wow! You have visited many countries!"})
        # info(window_memory.buffer)
        info(f"load_memory_variables - {window_memory.load_memory_variables({})}")

        conversation2 = ConversationChain(llm=model, memory=window_memory, verbose=False)
        response1 = conversation2.predict(
            input="Do you know where do I live? Just tell me whether you know it or not. Explanations are not needed :) I visited India, UAE, Netherlands, Belgium and France.")
        info(response1)  # expected: LLM should not know where I live, as the memory is limited to 1 conversation
        assert 'no' or 'not' in response1, "Location is not expected in the response"

        response2 = conversation2.predict(
            input="Do you know the countries I visited? Just tell me the country names, no other blah blah stuffs :)")
        info(
            response2)  # expected: LLM should know the countries I visited, as the memory is limited to 1(last) conversation
        assert 'Netherlands' and 'France' in response2, "'Netherlands' and 'France' not found in response"
