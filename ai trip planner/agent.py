import os
from typing import TypedDict, Annotated, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
# --- Import ChatGroq instead of ChatOpenAI ---
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
# --- Import ToolNode and remove ToolExecutor ---
from langgraph.prebuilt import ToolNode
from serpapi import SerpApiClient
from dotenv import load_dotenv
load_dotenv()
from tools.flights_finder import flights_finder 
from tools.hotel_finder import hotels_finder
from langchain_core.messages import HumanMessage, AIMessage,AnyMessage, SystemMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import datetime
from langsmith import traceable
from langchain_ollama.llms import OllamaLLM

CURRENT_YEAR = datetime.datetime.now().year
CUREENT_DATE=datetime.datetime.now().date().strftime("%Y-%m-%d")


TOOLS_SYSTEM_PROMPT = TOOLS_SYSTEM_PROMPT = f"""You are a smart travel agency. Your goal is to help users book flights and hotels.

The current date is {CUREENT_DATE}.

**Tool Use Guide:**
1.  **First, understand the user's request.** If you have enough information, decide which tools to use to find the necessary information.
2.  **After the tools provide you with data (in a ToolMessage), your job is to synthesize that information.** You must create a clear, user-friendly summary of the flight and hotel options.
3.  **Once you have the tool data, DO NOT call the same tools again.** Your final answer should be a of the data you have, not another tool call.
4.  Include links, prices, and logos in your final  as requested. For example:
    - Rate: $581 per night
    - Total: $3,488
"""

tools = [flights_finder, hotels_finder]

# --- Define the LLM using Groq ---
model = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
# model = OllamaLLM(model="llama3.1")

model_with_tool= model.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

workflow = StateGraph(AgentState)
def call_model(state: AgentState):
    """Node that invokes the LLM to decide the next action."""
    messages = state['messages']
    messages = [SystemMessage(content=TOOLS_SYSTEM_PROMPT)] + messages

    response = model_with_tool.invoke(state['messages'])
    return {"messages": [response]}
# Add this function to your script
import json
from langchain_core.messages import ToolMessage, AIMessage
from typing import TypedDict, Annotated, Any
from langgraph.graph.message import add_messages



# def process_tool_results(state: AgentState):
#     """
#     Parses and summarizes tool outputs to create a concise summary for the LLM.
#     """
#     last_message = state["messages"][-1]
#     if not isinstance(last_message, ToolMessage):
#         return state

#     tool_name = last_message.name
#     tool_output = last_message.content

#     try:
#         data = json.loads(tool_output)
#         summary = f"## Summary from {tool_name}:\n"

#         if tool_name == "flights_finder":
#             if not data:
#                 summary += "No flights found."
#             else:
#                 for i, flight in enumerate(data[:3]): # 'flight' here is the whole itinerary
#                     # --- THIS IS THE FIX ---
#                     # Access the airline from the first leg of the journey
#                     airline_name = flight['flights'][0]['airline']
#                     summary += (
#                         f"{i+1}. **{airline_name}**: "
#                         f"Price: ${flight['price']}, "
#                         f"Duration: {flight['total_duration'] // 60}h {flight['total_duration'] % 60}m\n"
#                     )

#         elif tool_name == "hotels_finder":
#             if not data:
#                 summary += "No hotels found."
#             else:
#                 for i, hotel in enumerate(data[:3]):
#                     summary += (
#                         f"{i+1}. **{hotel['name']}** ({hotel.get('hotel_class', 'N/A')}): "
#                         f"Rating: {hotel.get('overall_rating', 'N/A')}/5, "
#                         f"Price: {hotel.get('rate_per_night', {}).get('lowest', 'N/A')}\n"
#                     )
        
#         # Replace the large ToolMessage with our new summary
#         new_messages = state["messages"][:-1] + [ToolMessage(content=summary, tool_call_id=last_message.tool_call_id)]
#         return {"messages": new_messages}

#     except (json.JSONDecodeError, KeyError) as e:
#         # If it's not valid JSON or a key is missing, return the original content
#         # This prevents crashing on error messages from tools
#         print(f"Could not process tool output, passing through raw: {e}")
#         return state

tool_node = ToolNode(tools)
def custom_tools_condition(state: AgentState):
    last_msg = state["messages"][-1]
   
    # If the last AI message has tool calls, go to "action"
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "action"
    return END

workflow = StateGraph(AgentState)

# --- Add the agent node ---
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
# workflow.add_node("process_results", process_tool_results)

# --- Set the entry point and edges ---
workflow.set_entry_point("agent")

# --- Use the prebuilt tools_condition for the conditional edge ---
# This checks if the last message in the state contains tool_calls.
# If it does, it routes to "action". Otherwise, it routes to END.
workflow.add_conditional_edges(
    "agent",
    custom_tools_condition,
    {
        "action": "action",
        END: END,
    },
)
workflow.add_edge("action", "agent")

# workflow.add_edge("action", "process_results")

# # 3. Add an edge from your new node back to the agent
# workflow.add_edge("process_results", "agent")

# --- Compile the graph ---
app = workflow.compile()

inputs = {"messages": [HumanMessage(content="Find me a good hotel and flight from New York (JFK) to Los Angeles (LAX) on October 15, 2025 for 2 people.")]}
for output in app.stream(inputs):
    for value,key in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
        print("\n---\n")

