from flask import Flask, render_template, jsonify
import os
from pathlib import Path

app = Flask(__name__)

# Sample graph content for display
SAMPLE_GRAPH = """# This is the main state machine
State -> initialization
# Initialize the system
initialization -> processing
processing -> validation
# Validate results
validation -> END
"""

# File paths
BASE_DIR = Path(__file__).parent
FILES = {
    'state_spec': 'state_spec.md',
    'state_code': 'state_code.py',
    'node_spec': 'node_spec.md',
    'node_code': 'node_code.py',
    'graph_definition': 'graph_definition.py',
    'main': 'main.py'
}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/graph')
def get_graph():
    """Return the graph definition"""
    return jsonify({'content': SAMPLE_GRAPH})

@app.route('/api/file/<file_key>')
def get_file(file_key):
    """Return file content based on the key"""
    if file_key not in FILES:
        return jsonify({'error': 'File not found'}), 404
    
    file_path = BASE_DIR / FILES[file_key]
    
    # For demo purposes, return sample content
    sample_content = get_sample_content(file_key)
    
    # In production, you would read from actual files:
    # if file_path.exists():
    #     with open(file_path, 'r') as f:
    #         content = f.read()
    #     return jsonify({'content': content})
    
    return jsonify({'content': sample_content})

def get_sample_content(file_key):
    """Return sample content for demo purposes"""
    samples = {
        'state_spec': """# State Specification

## Overview
States define the major phases of the graph execution.

## Properties
- **Name**: Unique identifier for the state
- **Entry Actions**: Actions performed when entering the state
- **Exit Actions**: Actions performed when leaving the state

## Example
```
State -> initialization
```
""",
        'state_code': '''class State:
    """Base class for all states in the graph"""
    
    def __init__(self, name):
        self.name = name
        self.entry_actions = []
        self.exit_actions = []
    
    def on_enter(self):
        """Execute entry actions"""
        for action in self.entry_actions:
            action()
    
    def on_exit(self):
        """Execute exit actions"""
        for action in self.exit_actions:
            action()
''',
        'node_spec': """# Node Specification

## Overview
Nodes represent individual processing steps in the graph.

## Types
1. **Processing Node**: Performs data transformation
2. **Decision Node**: Makes branching decisions
3. **Terminal Node**: Ends the graph execution (END)

## Transitions
Nodes can transition to other nodes using the `->` operator.
""",
        'node_code': '''class Node:
    """Base class for all nodes in the graph"""
    
    def __init__(self, name):
        self.name = name
        self.next_nodes = []
    
    def process(self, data):
        """Process data at this node"""
        raise NotImplementedError
    
    def add_transition(self, target_node):
        """Add a transition to another node"""
        self.next_nodes.append(target_node)
''',
        'graph_definition': '''from state_code import State
from node_code import Node

class GraphDefinition:
    """Define and manage the execution graph"""
    
    def __init__(self):
        self.states = {}
        self.nodes = {}
        self.start_state = None
    
    def add_state(self, state):
        """Add a state to the graph"""
        self.states[state.name] = state
        if self.start_state is None:
            self.start_state = state
    
    def add_node(self, node):
        """Add a node to the graph"""
        self.nodes[node.name] = node
    
    def parse_definition(self, definition_string):
        """Parse graph definition from string format"""
        lines = definition_string.strip().split('\\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if '->' in line:
                source, target = line.split('->')
                source = source.strip()
                target = target.strip()
                # Create connections here
''',
        'main': '''#!/usr/bin/env python3
"""Main entry point for the graph execution system"""

from graph_definition import GraphDefinition
import sys

def main():
    """Initialize and run the graph"""
    graph = GraphDefinition()
    
    # Load graph definition
    with open('graph.txt', 'r') as f:
        definition = f.read()
    
    graph.parse_definition(definition)
    
    # Execute the graph
    print("Starting graph execution...")
    # Implementation here
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    }
    return samples.get(file_key, '# File not found')

if __name__ == '__main__':
    app.run(debug=True)
