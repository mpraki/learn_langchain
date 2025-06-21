from logging import info

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.learn_langchain.model import Model


class OutputParser:

    def learn(self):
        prompt = "I grow yam and maize crops in my 4 acre farm field in Attur, Salem."
        template = "For the text: '{text}', extract the information: crops, acre and location. " \
                   "{format_instructions}"

        crops_schema = ResponseSchema(name="crops", description="The crops grown in the farm field",
                                      type="List[string]")
        acre_schema = ResponseSchema(name="acre", description="The size of the farm field in acres", type="int")
        location_schema = ResponseSchema(name="location", description="The location of the farm field", type="string")
        response_schemas = [crops_schema, acre_schema, location_schema]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions(only_json=True)

        prompt_template = ChatPromptTemplate.from_template(
            template=template
        )

        formatted_prompt = prompt_template.format_messages(
            text=prompt, format_instructions=format_instructions
        )

        info(formatted_prompt)

        response = Model().invoke(formatted_prompt)
        json_output = parser.parse(response.content)

        info(json_output)
        info(json_output.get("crops"))
        info(json_output.get("acre"))
        info(json_output.get("location"))
