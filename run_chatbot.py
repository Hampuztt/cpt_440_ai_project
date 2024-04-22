#!/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
from langchain import PromptTemplate
from langchain.utilities import GoogleSerperAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.tools.retriever import create_retriever_tool

from langchain_community.llms import Ollama
import os

# 1. Load Retriever


embeddings = OllamaEmbeddings()
db = FAISS.load_local("recipe_qa", embeddings, allow_dangerous_deserialization=True) # loading
retriever = db.as_retriever(search_kwargs={"k": 1})

# 2. Create Tools

retriever_tool = create_retriever_tool(
    retriever,
    "Database Search",
    "Search for information about Recipe. For any questions about Recipe, you must use this tool!",
)

os.environ["SERPER_API_KEY"] = 'YOUR_SERPER_API_KEY'
search = GoogleSerperAPIWrapper()
tool_google = Tool(
        name="Google Search",
        func=search.run,
        description="useful for when you need to ask with search"
    )

tool_names = [retriever_tool, tool_google]

# 3. Create Agent

# Template for Database

template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the following questions as best you can. You have access to the following tools:

Database Search: You can access a database that includes thousands of recipes in the world. This database is very useful to answer questions about how to cook a dish, how many steps need to cook a dish. This input includes the food name.
Google Search: Useful for when you need to answer questions about other events. The input is the question to search relavant information.

Strictly use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Database Search, Google Search]
Action Input: the input to the action, should be a question.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

For examples:
Question: How can we cook monkey-brains-cocktail?
Thought: First, I need to find the recipe of monkey-brains-cocktail
Action: Database Search
Action Input: How do we cook monkey-brains-cocktail?
Observation: The first step is: ingredients. For this cocktail, you'll need the following:Banana Rum CaviarCrème de bananesDark rumBrown sugarAgar AgarVegetable oilTall glassSauce panSyringe or eye dropperStrainerBaileys SlushIrish creamIceBlenderFor the final presentation I use these cool 2.5 ounce skull glasses from thinkgeek.com\n\nThe second step is: make banana rum caviar. The molecular gastronomy element of this drink is loosely based on Mix That Drink's rum caviar. With agar agar you can transform the dark rum and banana liqueur into little jelly spheres that resemble fish eggs, or in this case, bits of monkey brains. Fill a tall glass with cooking oil. Put the glass of oil in the freezer to get cold.In a small sauce pan, stir together ¼ cup dark rum, ¼ cup creme de bananes and 2 grams of agar agar. Bring to a boil.Add ¼ cup of brown sugar, stir to dissolve all the sugar and remove from heat.Allow the rum mixture to cool for 3 minutes.Remove the glass of oil from the freezer.Suck up the rum mixture with a syringe and slowly add drops to the glass of oil. As the drops fall through the cold oil they turn into little jelly balls. This works best when the oil is really cold.While you can use an eye dropper, a syringe makes the process much faster and you want to work quickly before the rum mixture in your sauce pan starts to solidify and form a skin.Strain the rum caviar from the oil and rinse it under cold water.You can reuse your oil for the next batch.The amounts listed are enough for 3-4 drinks. Increase your batch size as needed. You can make the caviar in advance and store it in the fridge until just before serving.\n\nThe third step is: blend the baileys slush. In a blender or food processor, mix ice and Irish cream.Blend until you have a nice slush, similar to a blended margarita.\n\nThe fourth step is: serve and enjoy!. Pour the Baileys slush into the glass.Top with banana rum caviar.Enjoy!
Thought: I now know the final answer.
Final Answer: You can cook monkey-brains-cocktail by 4 steps. Firsly, preparing the ingredients before making Banana Rum Caviar. Then, you blend the Baileys Slush. Finally, pour the Baileys slush into the glass.Top with banana rum caviar.Enjoy!

For examples:
Question: How old is CEO of Microsoft wife?
Thought: First, I need to find who is the CEO of Microsoft.
Action: Google Search
Action Input: Who is the CEO of Microsoft?
Observation: Satya Nadella is the CEO of Microsoft.
Thought: Now, I should find out Satya Nadella's wife.
Action: Google Search
Action Input: Who is Satya Nadella's wife?
Observation: Satya Nadella's wife's name is Anupama Nadella.
Thought: Then, I need to check Anupama Nadella's age.
Action: Google Search
Action Input: How old is Anupama Nadella?
Observation: Anupama Nadella's age is 50.
Thought: I now know the final answer.
Final Answer: Anupama Nadella is 50 years old.

### Input:
{input}

### Response:
{agent_scratchpad}"""

temp_Ins = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Question: {thought}
Query: {query}
Observation: {observation}

### Input:
Make a short summary of useful information from the result observation that is related to the question.

### Response:"""

prompt_Ins = PromptTemplate(
    input_variables=["thought", "query", "observation"],
    template=temp_Ins,
)

class CustomPromptTemplate(StringPromptTemplate):
    """Schema to represent a prompt for an LLM.

    Example:
        .. code-block:: python

            from langchain import PromptTemplate
            prompt = PromptTemplate(input_variables=["foo"], template="Say {foo}")
    """

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    template: str
    """The prompt template."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""

    validate_template: bool = False
    """Whether or not to try validating the template."""
    

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        if len(intermediate_steps) > 0:
            regex = r"Thought\s*\d*\s*:(.*?)\nAction\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)\nObservation"
            text_match = intermediate_steps[-1][0].log
            if len(intermediate_steps) > 1:
                text_match = 'Thought: ' + text_match
            match = re.search(regex, text_match, re.DOTALL)            
            my_list = list(intermediate_steps[-1])
            p_INS_temp = prompt_Ins.format(thought=match.group(1).strip(), query=match.group(3).strip(), observation=my_list[1])
            my_list[1] = llm(p_INS_temp)
            my_tuple = tuple(my_list)            
            intermediate_steps[-1] = my_tuple
            
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f" {observation}\nThought:"
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        return self.template.format(**kwargs)
    
prompt = CustomPromptTemplate(input_variables=["input", "intermediate_steps"],
                              template=template,validate_template=False)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)(\n)+(.*)Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        print('llm output ', llm_output)
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        
output_parser = CustomOutputParser()

llm = Ollama(model="gemma:7b")
llm_chain = LLMChain(llm=llm, prompt=prompt)

agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:", "\nObservations:"], 
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tool_names, verbose=True)

# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.

class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )


class Output(BaseModel):
    output: str

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
