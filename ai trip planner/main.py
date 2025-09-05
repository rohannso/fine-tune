import streamlit as st
import os
import time
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import datetime

# --- Import your existing tools ---
from tools.flights_finder import flights_finder
from tools.hotel_finder import hotels_finder

# Load environment variables from .env file
load_dotenv()

# --- 1. DEFINE THE AGENT'S BACKEND LOGIC (Your existing code) ---

# This section contains the LangGraph agent you've already built.
# No changes are needed here.

CUREENT_DATE = datetime.datetime.now().date().strftime("%Y-%m-%d")

TOOLS_SYSTEM_PROMPT = f"""You are a smart travel agency. Your goal is to help users book flights and hotels.

The current date is {CUREENT_DATE}.

**Tool Use Guide:**
1.  **First, understand the user's request.** If you have enough information, decide which tools to use to find the necessary information.
2.  **After the tools provide you with data (in a ToolMessage), your job is to synthesize that information.** You must create a clear, user-friendly summary of the flight and hotel options.
3.  **Once you have the tool data, DO NOT call the same tools again.** Your final answer should be a summary of the data you have, not another tool call.
4.  Include links, prices, and logos in your final summary as requested. For example:
    - Rate: $581 per night
    - Total: $3,488
"""

# We cache the compiled graph to avoid rebuilding it on every user interaction.
@st.cache_resource
def build_agent():
    """Builds and compiles the LangGraph agent."""
    tools = [flights_finder, hotels_finder]
    model = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    model_with_tool = model.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def call_model(state: AgentState):
        messages = [SystemMessage(content=TOOLS_SYSTEM_PROMPT)] + state['messages']
        response = model_with_tool.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def custom_tools_condition(state: AgentState):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "action"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        custom_tools_condition,
        {"action": "action", END: END},
    )
    workflow.add_edge("action", "agent")
    
    return workflow.compile()

# Build the agent
app = build_agent()

# --- 2. BUILD THE STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="✈️ AI Travel Assistant", layout="wide")
st.title("✈️ AI Travel Assistant")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! How can I help you plan your trip today?")
    ]

# Display past messages
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

# The streaming function to simulate token-by-token output
def stream_response(text):
    """Yields words from the text to simulate a streaming response."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)

# Handle user input
if prompt := st.chat_input("e.g., Find me a flight and hotel..."):
    # Add user message to session state and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare inputs for the LangGraph agent
    inputs = {"messages": st.session_state.messages}

    # Display an empty assistant message with a spinner while processing
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # The final response is the last event in the stream
            final_response = None
            for output in app.stream(inputs):
                # The stream yields a dictionary with the node name as the key
                for key, value in output.items():
                    if key == "agent" and value["messages"]:
                        # We've received a message from the agent, which could be the final one
                        final_response = value["messages"][-1]
            
            # Now, stream the content of the final response
            if final_response:
                st.write_stream(stream_response(final_response.content))
                # Add the final AI response to the session state
                st.session_state.messages.append(final_response)
            else:
                st.error("Sorry, I couldn't get a response. Please try again.")