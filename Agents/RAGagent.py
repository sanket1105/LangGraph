import os
from operator import add as add_messages
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph

load_dotenv()

# Initialize models
llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load and process PDF
pdf_path = "/Users/shankii/Documents/LangGraph/Agents/Stock_Market_Performance_2024.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# Load and split documents
pages = PyPDFLoader(pdf_path).load()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(pages)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=os.getcwd(),
    collection_name="stock_market",
)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


@tool
def retriever_tool(query: str) -> str:
    """Search and return information from the Stock Market Performance 2024 document."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    return "\n\n".join(
        f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)
    )


# Setup agent state and tools
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [retriever_tool]
llm = llm.bind_tools(tools)
tools_dict = {tool.name: tool for tool in tools}


# Agent functions
def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    return (
        hasattr(state["messages"][-1], "tool_calls")
        and len(state["messages"][-1].tool_calls) > 0
    )


def call_llm(state: AgentState) -> AgentState:
    """Call the LLM with the current state."""
    messages = [
        SystemMessage(
            content="You are an AI assistant answering questions about Stock Market Performance in 2024. Use the retriever tool to find information and cite your sources."
        )
    ] + list(state["messages"])
    return {"messages": [llm.invoke(messages)]}


def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    return {
        "messages": [
            ToolMessage(
                tool_call_id=t["id"],
                name=t["name"],
                content=str(
                    tools_dict[t["name"]].invoke(t["args"].get("query", ""))
                    if t["name"] in tools_dict
                    else "Invalid tool name"
                ),
            )
            for t in state["messages"][-1].tool_calls
        ]
    }


# Build and compile graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges(
    "llm", should_continue, {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()


def run_agent():
    """Run the RAG agent in an interactive loop."""
    print("\n=== RAG AGENT ===")
    while True:
        user_input = input("\nWhat is your question (type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        result = rag_agent.invoke({"messages": [HumanMessage(content=user_input)]})
        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    run_agent()
