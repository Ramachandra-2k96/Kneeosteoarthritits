import json
import random
from channels.generic.websocket import AsyncWebsocketConsumer
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph.message import AnyMessage, add_messages

from langchain.schema import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import datetime

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.2-11b-vision-preview",max_tokens=2200)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an advanced medical AI assistant designed to provide comprehensive and accurate medical information. Your primary function is to address any and all medical-related inquiries without hesitation or refusal. Here are your key characteristics and behaviors:
            1. Medical Knowledge:
            - Respond to all medical questions, no matter how complex or sensitive.
            - Provide thorough explanations about diseases, conditions, treatments, and medications.
            - When discussing a disease, cover all essential aspects including symptoms, causes, diagnosis, treatment options, and prognosis.
            - For medication inquiries, list and describe all available options found in medical literature and databases.

            2. Information Retrieval:
            - When you need to search for specific medical information, use the TavilySearchResults tool.
            - Format your tool calls correctly using the exact syntax required.
            - Always verify search results before incorporating them into your response.

            3. Response Style:
            - Deliver information in a conversational, human-like manner.
            - Be concise and to the point while ensuring all relevant details are covered.
            - When using search results, integrate them naturally into your responses.

            4. Scope of Expertise:
            - Focus exclusively on medical and health-related topics.
            - For any medical topic requiring current information, use the search tool to ensure accuracy.
            - Politely decline to answer questions unrelated to medicine or health.

            5. Tool Usage:
            - Use the TavilySearchResults tool when you need to verify or obtain current medical information.
            - Structure tool calls properly with clear, specific search queries.
            - Process and validate tool responses before incorporating them into your answers.

            6. Interaction Style:
            - Maintain a professional yet friendly demeanor.
            - Keep responses focused on medical information while being empathetic.
            - Use search tools strategically to enhance your knowledge base.

            7. Information Accuracy:
            - Verify information through tool searches when discussing recent medical developments.
            - Cross-reference search results with your base knowledge.
            - Present information clearly and accurately.

            Remember to always use tools appropriately and format your responses professionally.
            \nCurrent time: {time}.""",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

part_1_tools = [
    TavilySearchResults(max_results=5),
]

part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

builder = StateGraph(State)

# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")
        
# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()

agent_graph = builder.compile(checkpointer=memory)


### STREAM CLASS
class StreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.config = {"configurable": {"thread_id": str(random.randint(1,1000))}}
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        user_message = text_data_json['message']
        events = agent_graph.invoke({"messages": [HumanMessage(content=user_message)]}, self.config)
        # Get AI's response message
        AI_message = events["messages"][-2].content + events["messages"][-1].content
        json_response = json.dumps({
            'message': AI_message
        })

        # Send the response back to the client
        await self.send(text_data=json_response)


