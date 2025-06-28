# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph code generation tool that converts graph specifications written in a simple text notation into complete LangGraph applications. The system generates state definitions, node implementations, and main execution files from graph descriptions.

## Core Architecture

### Main Components

1. **Flask Web Application** (`app.py`): Web interface for graph visualization and code generation
2. **Generation Pipeline**: Series of Python modules that generate different components:
   - `gen_state_spec.py` → `gen_state_code.py` → `gen_node_spec.py` → `gen_node_code.py` → `gen_graph_code.py` → `gen_main.py`
3. **Utility Module** (`mk_utils.py`): Core utilities for graph parsing, validation, LLM interactions, and project setup
4. **Graph Specifications**: Text files in `graphs/` directory using custom notation language

### Graph Notation Language

The system uses a custom text-based notation for describing LangGraph workflows:
- Simple transitions: `node1 -> node2`
- Conditional routing: `router -> {condition1: target1, condition2: target2}`
- Parallel execution: `node1 -> [node2, node3] -> merger`
- Human-in-the-loop: `node -> human_input -> node2`

### Configuration System

- `default.yaml`: Main configuration file specifying LLM providers, models, and generation settings
- Supports multiple LLM providers: OpenAI, Anthropic, OpenRouter, Google
- Separate configurations for spec generation vs code generation
- Configurable caching (file or SQLite)

### Generation Flow

1. Parse graph specification from text notation
2. Validate graph structure and detect patterns
3. Generate state specification (markdown)
4. Generate state code (Python dataclass)
5. Generate node specification (markdown)
6. Generate node code (Python functions)
7. Generate graph code (LangGraph workflow)
8. Generate main execution file

## Development Commands

### Running the Application
```bash
python app.py
```
The web interface will be available at http://localhost:5000

### Testing Generation Pipeline
Each generation module can be run independently for testing:
```bash
python gen_state_spec.py
python gen_state_code.py
# etc.
```

### Working with Graph Files
Graph specifications are stored in `graphs/` directory as `.txt` files. Example graphs are available in `graphs/examples/`.

## Key Dependencies

- **Flask**: Web framework for the UI
- **LangSmith**: For prompt management and LLM interactions
- **agno**: Agent framework for LLM interactions
- **LangChain**: For some prompt templates and LLM providers
- **colorama**: Terminal color output
- **questionary**: Interactive CLI prompts

## File Structure

- `graphs/`: Graph specification files (.txt)
- `templates/`: HTML templates and code generation templates
- `prompts/`: Local prompt files (most prompts are pulled from LangSmith hub)
- Generated artifacts are stored in folders named after the graph (configured via `base_folder` in default.yaml)

## Configuration Notes

- The `base_folder` in `default.yaml` determines where generated projects are created
- LLM configurations are split into three categories: `spec`, `node`, and `code` for different generation phases
- Prompts are primarily managed through LangSmith hub with format `hub:johannes/prompt-name`

## Working with Generated Code

Generated projects include:
- `state-spec.md` and `state_code.py`: State definition
- `node-spec.md` and `node_code.py`: Node implementations  
- `graph_code.py`: LangGraph workflow definition
- `main.py`: Execution entry point
- `*.yaml`: Configuration file
- `human_input.py` and `llm_cache.py`: Utility modules