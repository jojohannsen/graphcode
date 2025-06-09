# Graph Notation Design Patterns Reference

This document describes all supported design patterns in the graph notation language, detailing their syntax, implications, and implementation requirements.

## Pattern Categories

### 1. Sequential Flow Patterns

#### 1.1 Simple Transition
```
# Basic sequential execution from one node to another
node1 -> node2
```

**Description**: The most basic pattern representing sequential execution flow.

**Graph State Implications**:
- State is passed from `node1` to `node2` sequentially
- Any updates made by `node1` are available to `node2`
- No parallel execution or branching

**Node Requirements**:
- Both `node1` and `node2` must be implemented as node functions
- Each takes `state: StateClass` as primary argument
- Each returns `Dict[str, Any]` with state updates

**Conditional Edges**: None required

**Human-in-the-Loop**: Can be implemented in either node function through user input mechanisms

**Implementation Details**:
- Translates to `builder.add_edge('node1', 'node2')` in LangGraph
- Execution is synchronous and ordered

---

### 2. Parallel Execution Patterns

#### 2.1 Fork (Multiple Destinations)
```
# Fork execution into parallel branches
source_node -> dest1, dest2, dest3
```

**Description**: Spawns multiple nodes to execute concurrently after a single source node completes.

**Graph State Implications**:
- All destination nodes receive the same state from `source_node`
- Each parallel node can modify state independently
- State updates from parallel nodes are merged (order not guaranteed)

**Node Requirements**:
- Source node: Standard node function returning state updates
- Destination nodes: Each must handle receiving the same initial state
- All nodes run concurrently, so shared resource access must be thread-safe

**Conditional Edges**: None - all destinations execute unconditionally

**Human-in-the-Loop**: 
- Can occur in source node before forking
- Can occur in any/all of the parallel destination nodes
- Parallel human interactions may need coordination

**Implementation Details**:
- Translates to multiple `builder.add_edge('source_node', 'dest1')` calls
- All destination nodes start simultaneously
- Requires careful state field design to avoid conflicts

#### 2.2 Join (Multiple Sources)
```
# Multiple nodes converge to single destination
source1, source2, source3 -> destination_node
```

**Description**: Multiple independent source nodes all transition to the same destination node.

**Graph State Implications**:
- Destination node waits for ALL source nodes to complete
- State updates from all sources are merged before destination execution
- Field conflicts between sources need resolution strategy

**Node Requirements**:
- Source nodes: Independent node functions that can run separately
- Destination node: Must handle merged state from multiple sources
- Consider using `Annotated[list, operator.add]` for fields that accumulate

**Conditional Edges**: None - destination executes after all sources complete

**Human-in-the-Loop**:
- Can occur in any source node independently
- Destination node waits for ALL human interactions to complete
- May need timeout handling for user response coordination

**Implementation Details**:
- Each source gets separate `builder.add_edge()` call to destination
- LangGraph handles synchronization automatically
- State merging follows field annotation rules

---

### 3. Conditional Flow Patterns

#### 3.1 Routing Function (Conditional Branching)
```
# Dynamic routing based on state conditions
decision_node -> routing_function(option1, option2, END)
```

**Description**: Enables dynamic branching where the next node is determined by a routing function that evaluates the current state.

**Graph State Implications**:
- State is evaluated by routing function to determine next step
- Only one destination path is taken
- State flows unchanged to selected destination

**Node Requirements**:
- `decision_node`: Standard node function that prepares state for routing decision
- Destination nodes (`option1`, `option2`): Standard node functions
- `routing_function`: Special function that takes state and returns destination name

**Conditional Edges**: This IS the conditional edge implementation

**Human-in-the-Loop**:
- Can occur in `decision_node` to gather input for routing decision
- Can occur in any destination node
- Routing function itself typically doesn't involve human interaction

**Implementation Details**:
```python
def routing_function(state: StateClass) -> str:
    # Evaluation logic here
    if condition:
        return 'option1'
    elif other_condition:
        return 'option2'
    else:
        return 'END'

# In builder:
conditional_map = {'option1': 'option1', 'option2': 'option2', 'END': END}
builder.add_conditional_edges('decision_node', routing_function, conditional_map)
```

---

### 4. Parallel Processing Patterns

#### 4.1 Worker Nodes (List Processing)
```
# Process list elements concurrently
coordinator_node -> worker_node(StateClass.field_name)
```

**Description**: Processes each element of a list field concurrently using worker functions.

**Graph State Implications**:
- `StateClass.field_name` must be a list
- Worker processes each list element independently
- Results collected into `processed_<field_name>` list field
- Original list field remains unchanged

**Node Requirements**:
- `coordinator_node`: Standard node function that populates the list field
- `worker_node`: Special worker function signature:
  ```python
  def worker_node(field_value: ElementType, *, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
      # Process single field_value
      return {"processed_field_name": [result]}
  ```

**Conditional Edges**: None - all list elements are processed

**Human-in-the-Loop**:
- Can occur in coordinator node for list preparation
- Generally not suitable for worker nodes (too many parallel interactions)
- Consider aggregating human input needs in coordinator

**Implementation Details**:
- Requires `Annotated[list[ElementType], operator.add]` for processed field
- Worker function processes ONE element, returns list with single result
- LangGraph handles concurrency and result aggregation

---

### 5. Flow Control Patterns

#### 5.1 Termination
```
# End the graph execution
final_node -> END
```

**Description**: Explicitly terminates graph execution.

**Graph State Implications**:
- Final state from `final_node` becomes the graph's output
- No further processing occurs
- State should contain all required final results

**Node Requirements**:
- `final_node`: Standard node function that prepares final state
- Should ensure all required output fields are populated

**Conditional Edges**: Can be used as destination in routing functions

**Human-in-the-Loop**:
- Often appropriate for final confirmations or result presentation
- Last opportunity for user interaction

**Implementation Details**:
- `END` is a special LangGraph constant
- Translates to `builder.add_edge('final_node', END)`

#### 5.2 Explicit Start
```
# Explicitly define graph entry point
StateClass -> START
START -> first_node
```

**Description**: Explicitly defines the graph's starting node (though typically inferred).

**Graph State Implications**:
- Initial state setup occurs before first node
- Useful for complex state initialization scenarios

**Node Requirements**:
- `first_node`: Standard node function, receives initial state

**Conditional Edges**: None for START transitions

**Human-in-the-Loop**:
- Can occur in first node for initial user input
- State initialization might include user-provided data

**Implementation Details**:
- `START` is a special LangGraph constant
- Translates to `builder.add_edge(START, 'first_node')`

---

### 6. Documentation Patterns

#### 6.1 Comments
```
# This is a comment explaining the next operations
node1 -> node2
# Another comment describing the workflow section
node2 -> node3, node4
```

**Description**: Provides human-readable documentation within the graph specification.

**Graph State Implications**: None - comments don't affect execution

**Node Requirements**: None - comments are ignored during compilation

**Conditional Edges**: None

**Human-in-the-Loop**: Comments can document human interaction points

**Implementation Details**:
- Lines starting with `#` are completely ignored
- Can appear anywhere in the specification
- Useful for explaining complex routing logic or parallel sections

---

### 6. Chat and Messaging Patterns

#### 6.1 MessagesState Pattern (Chat Applications)
```
# Chat application using built-in MessagesState
ChatState -> get_user_input
# Process user message and generate response  
get_user_input -> generate_response
# Check if conversation should continue
generate_response -> continue_chat(get_user_input, END)
```

**Description**: Leverages LangGraph's built-in MessagesState for chat applications with automatic message handling.

**Graph State Implications**:
- Uses `MessagesState` as base class with built-in `messages` field
- Messages automatically handled with `add_messages` reducer
- Supports various message types (HumanMessage, AIMessage, SystemMessage, etc.)
- Can extend with additional fields for chat context

**Node Requirements**:
```python
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage

class ChatState(MessagesState):
    user_preferences: dict = {}
    conversation_context: str = ""

def get_user_input(state: ChatState):
    # In practice, this might wait for user input or process existing message
    return {"messages": []}  # Messages automatically appended

def generate_response(state: ChatState):
    # Access conversation history
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # Generate AI response
    response = AIMessage("Generated response based on conversation")
    return {"messages": [response]}
```

**Conditional Edges**: 
- `continue_chat` routing function determines if conversation continues
- Can evaluate message content, conversation length, or user preferences

**Human-in-the-Loop**:
- Natural pattern for human interaction through chat interface
- `get_user_input` node typically waits for human input
- Can implement typing indicators, message acknowledgments
- Supports real-time conversation flow

**Implementation Details**:
```python
from langgraph.graph import MessagesState, StateGraph, START, END

class ChatState(MessagesState):
    conversation_active: bool = True

def continue_chat(state: ChatState) -> str:
    # Check if user wants to continue or conversation should end
    last_message = state["messages"][-1] if state["messages"] else None
    
    if (last_message and 
        isinstance(last_message, HumanMessage) and 
        "goodbye" in last_message.content.lower()):
        return "END"
    return "get_user_input"

# Builder setup
builder = StateGraph(ChatState)
builder.add_node("get_user_input", get_user_input)
builder.add_node("generate_response", generate_response)
builder.add_edge(START, "get_user_input")
builder.add_edge("get_user_input", "generate_response")

conditional_map = {"get_user_input": "get_user_input", "END": END}
builder.add_conditional_edges("generate_response", continue_chat, conditional_map)
```

#### 6.2 Multi-Agent Chat Pattern
```
# Multi-agent conversation with message routing
ChatState -> message_router(agent_a, agent_b, human_input, END)
# Each agent processes and responds
agent_a, agent_b -> message_router
# Human can join conversation
human_input -> message_router
```

**Description**: Orchestrates conversation between multiple AI agents and human participants.

**Graph State Implications**:
- MessagesState tracks all participants' messages
- Additional fields may track current speaker, agent roles, conversation flow
- Message metadata can identify speaker/agent

**Node Requirements**:
- Each agent node: Processes conversation and adds their response
- `message_router`: Determines next speaker based on conversation state
- `human_input`: Handles human participant input

**Conditional Edges**: Central routing function manages conversation flow

**Human-in-the-Loop**: Integrated as equal participant in multi-agent conversation

**Implementation Details**:
- Message types can include custom metadata for agent identification
- Router evaluates conversation context to select next participant
- Supports complex conversation protocols and turn-taking rules

#### 6.3 Chat with Tools Pattern
```
# Chat application with tool integration
ChatState -> process_message
# Route based on whether tools are needed
process_message -> needs_tools(use_tools, generate_response)
# Use tools in parallel if needed
use_tools -> tool_a, tool_b, tool_c
# Aggregate tool results
tool_a, tool_b, tool_c -> aggregate_tools
# Generate final response with tool data
aggregate_tools, generate_response -> final_response
# Continue conversation
final_response -> continue_chat(process_message, END)
```

**Description**: Integrates external tools and function calling into chat conversation flow.

**Graph State Implications**:
- MessagesState for conversation history
- Additional fields for tool results, function calls, external data
- Tool outputs integrated into conversation context

**Node Requirements**:
- Tool nodes: Execute external functions, APIs, or services
- `needs_tools`: Analyzes message to determine if tools are required
- Response generation: Incorporates tool results into conversational response

**Conditional Edges**: Multiple routing points for tool usage and conversation flow

**Human-in-the-Loop**: Natural chat interface with enhanced capabilities through tools

**Implementation Details**:
- Tools can run in parallel when multiple are needed
- Tool results stored in state for response generation
- Supports complex tool orchestration and error handling

---

## Complex Pattern Combinations

### 7.1 Fork-Process-Join Pattern
```
# Collect input
input_node -> data_prep
# Fork into parallel processing
data_prep -> process_a, process_b, process_c  
# Join results back together
process_a, process_b, process_c -> result_aggregator
# Final output
result_aggregator -> END
```

### 7.2 Conditional with Parallel Processing
```
# Decide processing strategy
analyzer -> routing_function(simple_process, complex_process)
# Complex processing uses workers
complex_process -> worker_node(StateClass.tasks)
# Both paths converge
simple_process, worker_node -> final_output
final_output -> END
```

### 7.3 Human-in-the-Loop with Retry Pattern
```
# Get user input
user_input -> process_request
# Ask for confirmation
process_request -> confirm_with_user
# Route based on confirmation
confirm_with_user -> confirmation_router(finalize, user_input, END)
# Final step
finalize -> END
```

## State Design Considerations

### Field Types for Different Patterns
- **Sequential patterns**: Standard typed fields
- **Parallel patterns**: Use `Annotated[list[T], operator.add]` for accumulating results
- **Worker patterns**: Source list + `Annotated[list[T], operator.add]` processed list
- **Routing patterns**: Include fields that routing functions evaluate

### Naming Conventions
- Node functions: Descriptive action names (`process_data`, `get_user_input`)
- Routing functions: Question-like names (`should_continue`, `choose_method`)
- Worker functions: Processing action names (`analyze_item`, `transform_element`)
- State fields: Clear, descriptive names matching their purpose

## Implementation Best Practices

1. **Error Handling**: Each pattern should include error handling appropriate to its execution model
2. **Timeout Management**: Especially important for human-in-the-loop and parallel patterns
3. **State Validation**: Ensure state contracts are maintained across pattern boundaries
4. **Resource Management**: Consider resource usage in parallel patterns
5. **Testing**: Each pattern has different testing requirements and edge cases