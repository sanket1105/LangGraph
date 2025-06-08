# LangGraph Examples and Agents

This repository contains a collection of examples and implementations using LangGraph, showcasing various patterns and architectures for building AI agents and workflows.

## Repository Structure

```
.
├── Agents/               # Agent implementations
│   ├── React.py         # ReAct pattern agent with mathematical tools
│   ├── MemoryAgent.py   # Agent with memory capabilities
│   └── Bot.py           # Basic bot implementation
├── Graphs/              # Jupyter notebooks with graph examples
│   ├── HelloWorld.ipynb # Basic LangGraph introduction
│   ├── Sequential.ipynb # Sequential workflow examples
│   ├── Conditional.ipynb # Conditional branching examples
│   ├── Looping.ipynb    # Loop-based workflow examples
│   └── MultipleInputs.ipynb # Handling multiple inputs
└── requirements.txt     # Project dependencies
```

## Features

### Agents

- **ReAct Agent**: Implements the ReAct (Reasoning and Acting) pattern with mathematical tools
- **Memory Agent**: Demonstrates agent memory and state management
- **Basic Bot**: Simple bot implementation for basic interactions

### Graph Examples

- Hello World introduction to LangGraph
- Sequential workflow patterns
- Conditional branching and decision making
- Loop-based workflows
- Multiple input handling

## Prerequisites

- Python 3.x
- OpenAI API key
- Jupyter Notebook (for running examples)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd LangGraph
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Running Agent Examples

```python
from Agents.React import app, HumanMessage

# Example input
inputs = {
    "messages": [
        HumanMessage(content="Add 40 + 12 and then multiply the result by 6.")
    ]
}

# Process the input
for s in app.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    message.pretty_print()
```

### Running Graph Examples

1. Start Jupyter Notebook:

```bash
jupyter notebook
```

2. Navigate to the `Graphs` directory and open any of the example notebooks:
   - `HelloWorld.ipynb` for basic concepts
   - `Sequential.ipynb` for sequential workflows
   - `Conditional.ipynb` for conditional logic
   - `Looping.ipynb` for loop-based workflows
   - `MultipleInputs.ipynb` for handling multiple inputs

## Available Tools

The ReAct agent comes with the following mathematical tools:

- `add(a: int, b: int)`: Adds two numbers
- `subtract(a: int, b: int)`: Subtracts two numbers
- `multiply(a: int, b: int)`: Multiplies two numbers

## Architecture

The project demonstrates various architectural patterns:

1. **ReAct Pattern**:

   - Agent receives input messages
   - Model processes input and decides whether to use tools
   - Tools are executed when needed
   - Process continues until final response

2. **Graph-based Workflows**:
   - Sequential processing
   - Conditional branching
   - Loop-based operations
   - Multiple input handling

## Dependencies

Key dependencies include:

- langgraph
- langchain
- langchain_openai
- langchain_community
- ipython
- dotenv
- typing
- chromadbx
- langchain_chroma

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
