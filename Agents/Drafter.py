from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

## global variables
document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """updates the doctring with the given content from llm given the user was not happy so it got reassigned to this function again"""

    global document_content
    document_content = content
    return f"Updated the document content! The current content is {document_content}"


@tool
def save(filename: str) -> str:
    """ "
    Saves the current document to a text file and finishes the process
    Args: document context that needs to be stored in the file"""

    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as f:
            f.write(document_content)
            print("Sucecssfully written the document content into the file")
            return f"Document saved successfully as {filename}"

    except Exception as e:
        return f"Error saving document : {str(e)}"


tools = [update, save]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    system_promot = SystemMessage(
        content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """
    )

    ## ask the question first
    if not state["messages"]:
        user_input = "I am ready to update a document! What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\n What changes you would like to make to this document?")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_promot] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    # Print the AI's response content
    print(f"\n🤖 AI: {response.content}")

    # Check if the response contains tool calls and print which tools are being used
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tool_call["name"] for tool_call in response.tool_calls]
        print(f"🔧 USING TOOLS: {tool_names}")
    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]

    if not messages:
        return "continue"

    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if isinstance(message, ToolMessage) and "saved" in message.content.lower():
            return "end"  # goes to the end edge which leads to the endpoint

    return "continue"


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {"continue": "agent", "end": END},
)


app = graph.compile()


def run_document_agent():
    print("\n ===== DRAFTER =====")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
