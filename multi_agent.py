import functools
import operator
from typing import Annotated, Sequence, TypedDict
from colorama import Fore, Style
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from setup_environment import set_environment_variables
from tools import generate_image, markdown_to_pdf_file
from multi_agent_prompts import TEAM_SUPERVISOR_SYSTEM_PROMPT

from multi_agent_prompts import (
    TEAM_SUPERVISOR_SYSTEM_PROMPT,
    TRAVEL_AGENT_SYSTEM_PROMPT,
    LANGUAGE_ASSISTANT_SYSTEM_PROMPT,
    VISUALIZER_SYSTEM_PROMPT,
    DESIGNER_SYSTEM_PROMPT,
)

set_environment_variables("MultiAgentTeam")

TRAVEL_AGENT_NAME = "travel_agent"
LANGUAGE_ASSISTANT_NAME = "language_assistant"
VISUALIZER_NAME = "visualizer"
DESIGNER_NAME = "designer"
TEAM_SUPERVISOR_NAME = "team_supervisor"

MEMBERS = [TRAVEL_AGENT_NAME, LANGUAGE_ASSISTANT_NAME, VISUALIZER_NAME]
OPTIONS = ["FINISH"] + MEMBERS

TAVILY_TOOL = TavilySearchResults()
LLM = ChatOpenAI(model="gpt-3.5-turbo-0125")

# We’re going to be creating a lot of agents here, so let’s create a function to handle the repetitive work of creating an agent
def create_agent(llm: BaseChatModel, tools: list, system_prompt: str):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    return agent_executor

# Now let’s declare the state object that we will be passing around in this particular graph:
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Now let’s define a function that represents one of these agent nodes in general:
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# For this function that doesn’t actually exist, we’re going to define an old-school vanilla OpenAI function description that describes how the function works to the LLM team supervisor. Add the following variable
router_function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "next",
                "anyOf": [
                    {"enum": OPTIONS},
                ],
            }
        },
        "required": ["next"],
    },
}

# define our team supervisor’s prompt template manually as it will be different from all the other agents:
team_supervisor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", TEAM_SUPERVISOR_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=", ".join(OPTIONS), members=", ".join(MEMBERS))

# define the team supervisor node as a langchain using LCEL
team_supervisor_chain = (
    team_supervisor_prompt_template
    | LLM.bind_functions(functions=[router_function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

# create travel agent
travel_agent = create_agent(
    LLM,
    tools=[TAVILY_TOOL],
    system_prompt=TRAVEL_AGENT_SYSTEM_PROMPT,
)
travel_agent_node = functools.partial(agent_node, agent=travel_agent, name=TRAVEL_AGENT_NAME)

# create language assistant
language_assistant = create_agent(
    LLM,
    tools=[TAVILY_TOOL],
    system_prompt=LANGUAGE_ASSISTANT_SYSTEM_PROMPT,
)
language_assistant_node = functools.partial(agent_node, agent=language_assistant, name=LANGUAGE_ASSISTANT_NAME)

# create visualizer
visualizer = create_agent(
    LLM,
    tools=[generate_image],
    system_prompt=VISUALIZER_SYSTEM_PROMPT,
)
visualizer_node = functools.partial(agent_node, agent=visualizer, name=VISUALIZER_NAME)

# create designer
designer = create_agent(
    LLM,
    tools=[markdown_to_pdf_file],
    system_prompt=DESIGNER_SYSTEM_PROMPT,
)
designer_node = functools.partial(agent_node, agent=designer, name=DESIGNER_NAME)

# Time to create our graph and the nodes:
workflow = StateGraph(AgentState)
workflow.add_node(TRAVEL_AGENT_NAME, travel_agent_node)
workflow.add_node(LANGUAGE_ASSISTANT_NAME, language_assistant_node)
workflow.add_node(VISUALIZER_NAME, visualizer_node)
workflow.add_node(DESIGNER_NAME, designer_node)
workflow.add_node(TEAM_SUPERVISOR_NAME, team_supervisor_chain)

# start building some edges
for member in MEMBERS:
    workflow.add_edge(member, TEAM_SUPERVISOR_NAME)

workflow.add_edge(DESIGNER_NAME, END)

# Now it is time for us to add some conditional edges:
conditional_map = {name: name for name in MEMBERS}
conditional_map["FINISH"] = DESIGNER_NAME
workflow.add_conditional_edges(TEAM_SUPERVISOR_NAME, lambda x: x["next"], conditional_map)

# Now set the entry point and compile the graph:
workflow.set_entry_point(TEAM_SUPERVISOR_NAME)
travel_agent_graph = workflow.compile()

# So we’re going to call stream on the travel_agent_graph and pass in a dictionary with the messages key 
# and a list with a single HumanMessage object in it, saying that we want to visit Paris. for three days. 
# We then loop over the chunks and print them out, and then print a line of #s in green to visually separate the chunks.
for chunk in travel_agent_graph.stream({"messages": [HumanMessage(content="I want to visit Hyderabad for three days.", name="user")]}):
  if "__end__" not in chunk:
    print(chunk)
    print(f"{Fore.GREEN}#############################{Style.RESET_ALL}")






