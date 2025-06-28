from flask import Flask, render_template, jsonify, request, send_file
import os
from pathlib import Path
import glob
import zipfile
import io
import datetime
import yaml
from gen_state_spec import generate_state_spec
from gen_state_code import generate_state_code
from gen_node_spec import generate_node_spec
from gen_node_code import generate_node_code
from gen_graph_code import generate_graph_code
from gen_main import generate_main

app = Flask(__name__)
app.config['DEFAULT_CONFIG'] = None

def load_config():
    """Load configuration from default.yaml"""
    if app.config['DEFAULT_CONFIG'] is None:
        default_config_path = Path("default.yaml")
        if default_config_path.exists():
            with open(default_config_path, "r") as f:
                app.config['DEFAULT_CONFIG'] = yaml.safe_load(f)
        else:
            app.config['DEFAULT_CONFIG'] = {}

@app.before_request
def before_request():
    """Load config before each request."""
    load_config()

def get_base_folder():
    """Get base folder from default.yaml configuration"""
    base_folder = app.config.get('DEFAULT_CONFIG', {}).get('base_folder')
    return os.path.expanduser(base_folder) if base_folder else None

def get_artifacts_base_dir():
    """Get the base directory where artifacts are stored"""
    base_folder = get_base_folder()
    return Path(base_folder) if base_folder else Path(__file__).parent

def extract_documentation(content):
    """Extract triple-quoted documentation block and return (readme, graph_content)"""
    lines = content.split('\n')
    readme_content = ""
    graph_start_index = 0
    
    # Look for triple-quoted block at the beginning
    if lines and lines[0].strip().startswith('"""'):
        in_doc_block = True
        doc_lines = []
        
        # Skip the opening triple quotes line
        for i, line in enumerate(lines[1:], 1):
            if line.strip().endswith('"""'):
                # Found closing triple quotes
                in_doc_block = False
                graph_start_index = i + 1
                break
            else:
                doc_lines.append(line)
        
        if not in_doc_block:  # Successfully found complete doc block
            readme_content = '\n'.join(doc_lines).strip()
    
    # Extract graph content (everything after documentation)
    graph_lines = lines[graph_start_index:]
    # Remove any leading empty lines
    while graph_lines and not graph_lines[0].strip():
        graph_lines.pop(0)
    
    graph_content = '\n'.join(graph_lines)
    
    return readme_content, graph_content

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

@app.route('/api/txt-files')
def list_txt_files():
    """Return a nested structure of .txt files in the 'graphs' directory and its subfolders"""
    def walk_dir(path):
        folders = []
        files = []
        for entry in sorted(os.listdir(path)):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                folders.append(walk_dir(full_path) | {"name": entry})
            elif entry.endswith('.txt'):
                files.append(entry)
        return {"folders": folders, "files": files}

    base_path = os.path.join(os.path.dirname(__file__), 'graphs')
    tree = walk_dir(base_path)
    return jsonify(tree)

@app.route('/api/graph/<path:filename>')
def get_graph(filename):
    """Return the graph definition from a txt file in the graphs directory or subfolders"""
    base_graphs = Path(__file__).parent / "graphs"
    file_path = base_graphs / filename
    if not file_path.exists():
        # fallback: try with .txt extension if not present
        if not str(file_path).endswith('.txt') and (file_path.with_suffix('.txt')).exists():
            file_path = file_path.with_suffix('.txt')
        else:
            return jsonify({'error': 'File not found'}), 404
    try:
        def copy_if_needed(src, dest):
            if src.exists() and not dest.exists():
                dest.write_text(src.read_text())

        artifacts_base_dir = get_artifacts_base_dir()
        if artifacts_base_dir:
            graph_name = filename[:-4] if filename.endswith('.txt') else filename
            graph_folder_path = artifacts_base_dir / graph_name
            graph_folder_path.mkdir(parents=True, exist_ok=True)

            # Copy default.yaml
            default_yaml_path = BASE_DIR / 'default.yaml'
            # Extract just the filename part for the YAML file
            yaml_filename = Path(graph_name).name
            new_yaml_path = graph_folder_path / f"{yaml_filename}.yaml"
            copy_if_needed(default_yaml_path, new_yaml_path)

            # Copy human_input.py and llm_cache.py
            copy_if_needed(BASE_DIR / 'human_input.py', graph_folder_path / 'human_input.py')
            copy_if_needed(BASE_DIR / 'llm_cache.py', graph_folder_path / 'llm_cache.py')

        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract and process documentation block
        readme_content, graph_content = extract_documentation(content)
        
        # Save README.md and cleaned graph if artifacts directory exists
        if artifacts_base_dir:
            # Save README.md if documentation was found
            if readme_content:
                readme_path = graph_folder_path / 'README.md'
                if not readme_path.exists():
                    readme_path.write_text(readme_content)
            
            # Save cleaned graph file
            graph_filename = Path(graph_name).name + '.txt'
            graph_file_path = graph_folder_path / graph_filename
            if not graph_file_path.exists():
                graph_file_path.write_text(graph_content)
        
        return jsonify({
            'content': graph_content, 
            'filename': filename,
            'readme': readme_content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph')
def get_default_graph():
    """Return the default sample graph"""
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

@app.route('/api/generate/state-spec', methods=['POST'])
def generate_state_spec_endpoint():
    """Generate state specification from graph file"""
    print("=== Starting state spec generation endpoint ===", flush=True)

    try:
        data = request.json
        filename = data.get('filename')
        print(f"Received request for filename: {filename}", flush=True)
        
        if not filename:
            print("ERROR: No filename provided", flush=True)
            return jsonify({'error': 'No filename provided'}), 400
        
        # Read the graph file from graphs/ directory
        graph_file_path = Path(__file__).parent / 'graphs' / filename
        print(f"Looking for file at: {graph_file_path}", flush=True)
        if not graph_file_path.exists():
            print("ERROR: File not found", flush=True)
            return jsonify({'error': 'File not found'}), 404
        
        with open(graph_file_path, 'r') as f:
            graph_spec = f.read()
        print(f"Successfully read graph spec ({len(graph_spec)} characters)", flush=True)
        
        # Extract graph name preserving folder structure (remove .txt extension)
        graph_name = filename[:-4] if filename.endswith('.txt') else filename
        print(f"Graph name: {graph_name}", flush=True)
        
        # Generate state spec
        try:
            print("Calling generate_state_spec function...", flush=True)
            state_spec_content = generate_state_spec(graph_name, graph_spec)
            print(f"Successfully generated state spec ({len(state_spec_content)} characters)", flush=True)
            return jsonify({
                'content': state_spec_content,
                'success': True
            })
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR in generate_state_spec: {error_details}", flush=True)
            return jsonify({'error': f'Generation failed: {str(e)}'}), 500
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in endpoint: {error_details}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate/state-code', methods=['POST'])
def generate_state_code_endpoint():
    """Generate state code from graph file and state spec"""
    print("=== Starting state code generation endpoint ===", flush=True)
    try:
        data = request.json
        filename = data.get('filename')
        print(f"Received request for filename: {filename}", flush=True)
        
        if not filename:
            print("ERROR: No filename provided", flush=True)
            return jsonify({'error': 'No filename provided'}), 400
        
        # Read the graph file from graphs/ directory
        graph_file_path = Path(__file__).parent / 'graphs' / filename
        print(f"Looking for file at: {graph_file_path}", flush=True)
        if not graph_file_path.exists():
            print("ERROR: File not found", flush=True)
            return jsonify({'error': 'File not found'}), 404
        
        with open(graph_file_path, 'r') as f:
            graph_spec = f.read()
        print(f"Successfully read graph spec ({len(graph_spec)} characters)", flush=True)
        
        # Extract graph name preserving folder structure (remove .txt extension)
        graph_name = filename[:-4] if filename.endswith('.txt') else filename
        print(f"Graph name: {graph_name}", flush=True)
        
        # Generate state code
        try:
            print("Calling generate_state_code function...", flush=True)
            state_code_content = generate_state_code(graph_name, graph_spec)
            print(f"Successfully generated state code ({len(state_code_content)} characters)", flush=True)
            return jsonify({
                'content': state_code_content,
                'success': True
            })
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR in generate_state_code: {error_details}", flush=True)
            return jsonify({'error': f'Generation failed: {str(e)}'}), 500
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in endpoint: {error_details}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate/node-spec', methods=['POST'])
def generate_node_spec_endpoint():
    """Generate node specification from graph file and state spec"""
    print("=== Starting node spec generation endpoint ===", flush=True)
    try:
        data = request.json
        filename = data.get('filename')
        print(f"Received request for filename: {filename}", flush=True)
        
        if not filename:
            print("ERROR: No filename provided", flush=True)
            return jsonify({'error': 'No filename provided'}), 400
        
        # Read the graph file from graphs/ directory
        graph_file_path = Path(__file__).parent / 'graphs' / filename
        print(f"Looking for file at: {graph_file_path}", flush=True)
        if not graph_file_path.exists():
            print("ERROR: File not found", flush=True)
            return jsonify({'error': 'File not found'}), 404
        
        with open(graph_file_path, 'r') as f:
            graph_spec = f.read()
        print(f"Successfully read graph spec ({len(graph_spec)} characters)", flush=True)
        
        # Extract graph name preserving folder structure (remove .txt extension)
        graph_name = filename[:-4] if filename.endswith('.txt') else filename
        print(f"Graph name: {graph_name}", flush=True)
        
        # Generate node spec
        try:
            print("Calling generate_node_spec function...", flush=True)
            node_spec_content = generate_node_spec(graph_name, graph_spec)
            print(f"Successfully generated node spec ({len(node_spec_content)} characters)", flush=True)
            return jsonify({
                'content': node_spec_content,
                'success': True
            })
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR in generate_node_spec: {error_details}", flush=True)
            return jsonify({'error': f'Generation failed: {str(e)}'}), 500
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in endpoint: {error_details}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate/node-code', methods=['POST'])
def generate_node_code_endpoint():
    """Generate node code from graph file and dependencies"""
    print("=== Starting node code generation endpoint ===", flush=True)
    try:
        data = request.json
        filename = data.get('filename')
        print(f"Received request for filename: {filename}", flush=True)
        
        if not filename:
            print("ERROR: No filename provided", flush=True)
            return jsonify({'error': 'No filename provided'}), 400
        
        # Read the graph file from graphs/ directory
        graph_file_path = Path(__file__).parent / 'graphs' / filename
        print(f"Looking for file at: {graph_file_path}", flush=True)
        if not graph_file_path.exists():
            print("ERROR: File not found", flush=True)
            return jsonify({'error': 'File not found'}), 404
        
        with open(graph_file_path, 'r') as f:
            graph_spec = f.read()
        print(f"Successfully read graph spec ({len(graph_spec)} characters)", flush=True)
        
        # Extract graph name preserving folder structure (remove .txt extension)
        graph_name = filename[:-4] if filename.endswith('.txt') else filename
        print(f"Graph name: {graph_name}", flush=True)
        
        # Generate node code
        try:
            print("Calling generate_node_code function...", flush=True)
            node_code_content = generate_node_code(graph_name, graph_spec)
            print(f"Successfully generated node code ({len(node_code_content)} characters)", flush=True)
            return jsonify({
                'content': node_code_content,
                'success': True
            })
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR in generate_node_code: {error_details}", flush=True)
            return jsonify({'error': f'Generation failed: {str(e)}'}), 500
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in endpoint: {error_details}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate/graph-code', methods=['POST'])
def generate_graph_code_endpoint():
    """Generate graph code from graph file and all dependencies"""
    print("=== Starting graph code generation endpoint ===", flush=True)
    try:
        data = request.json
        filename = data.get('filename')
        print(f"Received request for filename: {filename}", flush=True)
        
        if not filename:
            print("ERROR: No filename provided", flush=True)
            return jsonify({'error': 'No filename provided'}), 400
        
        # Read the graph file from graphs/ directory
        graph_file_path = Path(__file__).parent / 'graphs' / filename
        print(f"Looking for file at: {graph_file_path}", flush=True)
        if not graph_file_path.exists():
            print("ERROR: File not found", flush=True)
            return jsonify({'error': 'File not found'}), 404
        
        with open(graph_file_path, 'r') as f:
            graph_spec = f.read()
        print(f"Successfully read graph spec ({len(graph_spec)} characters)", flush=True)
        
        # Extract graph name preserving folder structure (remove .txt extension)
        graph_name = filename[:-4] if filename.endswith('.txt') else filename
        print(f"Graph name: {graph_name}", flush=True)
        
        # Generate graph code
        try:
            print("Calling generate_graph_code function...", flush=True)
            graph_code_content = generate_graph_code(graph_name, graph_spec)
            print(f"Successfully generated graph code ({len(graph_code_content)} characters)", flush=True)
            return jsonify({
                'content': graph_code_content,
                'success': True
            })
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR in generate_graph_code: {error_details}", flush=True)
            return jsonify({'error': f'Generation failed: {str(e)}'}), 500
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in endpoint: {error_details}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate/main', methods=['POST'])
def generate_main_endpoint():
    """Generate main code from graph file and all dependencies"""
    print("=== Starting main code generation endpoint ===", flush=True)
    try:
        data = request.json
        filename = data.get('filename')
        print(f"Received request for filename: {filename}", flush=True)
        
        if not filename:
            print("ERROR: No filename provided", flush=True)
            return jsonify({'error': 'No filename provided'}), 400
        
        # Read the graph file from graphs/ directory
        graph_file_path = Path(__file__).parent / 'graphs' / filename
        print(f"Looking for file at: {graph_file_path}", flush=True)
        if not graph_file_path.exists():
            print("ERROR: File not found", flush=True)
            return jsonify({'error': 'File not found'}), 404
        
        with open(graph_file_path, 'r') as f:
            graph_spec = f.read()
        print(f"Successfully read graph spec ({len(graph_spec)} characters)", flush=True)
        
        # Extract graph name preserving folder structure (remove .txt extension)
        graph_name = filename[:-4] if filename.endswith('.txt') else filename
        print(f"Graph name: {graph_name}", flush=True)
        
        # Generate main code
        try:
            print("Calling generate_main function...", flush=True)
            main_content = generate_main(graph_name, graph_spec)
            print(f"Successfully generated main code ({len(main_content)} characters)", flush=True)
            return jsonify({
                'content': main_content,
                'success': True
            })
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR in generate_main: {error_details}", flush=True)
            return jsonify({'error': f'Generation failed: {str(e)}'}), 500
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in endpoint: {error_details}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-file/<path:file_path>')
def check_file_exists(file_path):
    """Check if a file exists"""
    artifacts_base_dir = get_artifacts_base_dir()
    full_path = artifacts_base_dir / file_path
    
    if full_path.exists() and full_path.is_file():
        return jsonify({'exists': True}), 200
    else:
        return jsonify({'exists': False}), 404

@app.route('/api/get-file/<path:file_path>')
def get_existing_file(file_path):
    """Get content of an existing file"""
    artifacts_base_dir = get_artifacts_base_dir()
    full_path = artifacts_base_dir / file_path
    if not full_path.exists() or not full_path.is_file():
        return jsonify({'error': 'File not found'}), 404
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({'content': content}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-file/<path:file_path>', methods=['DELETE'])
def delete_file(file_path):
    """Delete a file"""
    artifacts_base_dir = get_artifacts_base_dir()
    full_path = artifacts_base_dir / file_path
    if not full_path.exists() or not full_path.is_file():
        return jsonify({'error': 'File not found'}), 404
    
    try:
        full_path.unlink()
        print(f"Deleted file: {full_path}", flush=True)
        return jsonify({'success': True, 'message': 'File deleted successfully'}), 200
    except Exception as e:
        print(f"Error deleting file {full_path}: {e}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-file/<path:file_path>', methods=['PUT'])
def save_file(file_path):
    """Save content to a file"""
    try:
        data = request.json
        content = data.get('content', '')
        
        artifacts_base_dir = get_artifacts_base_dir()
        full_path = artifacts_base_dir / file_path
        
        # Ensure parent directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Saved file: {full_path}", flush=True)
        return jsonify({'success': True, 'message': 'File saved successfully'}), 200
    except Exception as e:
        print(f"Error saving file {file_path}: {e}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-zip/<filename>')
def download_zip(filename):
    """Create and download a zip file containing all artifacts and graph specification"""
    try:
        # Extract graph name preserving folder structure (remove .txt extension)
        graph_name = filename[:-4] if filename.endswith('.txt') else filename
        artifacts_base_dir = get_artifacts_base_dir()

        # Create zip file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add the original graph specification file from graphs directory
            graph_file_path = BASE_DIR / 'graphs' / filename
            if graph_file_path.exists():
                zip_file.write(graph_file_path, f"{graph_name}/{filename}")
            
            # Define all possible artifact files, relative to the graph folder
            artifact_files = [
                'state-spec.md',
                'state_code.py',
                'node-spec.md',
                'node_code.py',
                'graph_code.py',
                'main.py',
                'lgcodegen_llm.py',
                f"{graph_name}.yaml",
                'human_input.py',
                'llm_cache.py'
            ]
            
            # Add existing artifact files to the zip from the artifacts directory
            graph_folder_path = artifacts_base_dir / graph_name
            for artifact_filename in artifact_files:
                full_path = graph_folder_path / artifact_filename
                if full_path.exists():
                    zip_file.write(full_path, f"{graph_name}/{artifact_filename}")
        
        zip_buffer.seek(0)
        
        # Send the zip file as download
        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name=f"{graph_name}_artifacts.zip",
            mimetype='application/zip'
        )
        
    except Exception as e:
        print(f"Error creating zip file: {e}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph-overview/<filename>')
def get_graph_overview(filename):
    """Get the graph overview with current artifact status"""
    try:
        # Extract graph name preserving folder structure (remove .txt extension)
        graph_name = filename[:-4] if filename.endswith('.txt') else filename
        artifacts_base_dir = get_artifacts_base_dir()
        
        # Read the overview template
        overview_template_path = BASE_DIR / 'templates' / 'graph_overview.md'
        if not overview_template_path.exists():
            return jsonify({'error': 'Overview template not found'}), 404
        
        with open(overview_template_path, 'r', encoding='utf-8') as f:
            overview_content = f.read()
        
        # Define artifact file mappings
        artifacts = [
            ('state_spec', "state-spec.md", "Generate"),
            ('state_code', "state_code.py", "Generate"),
            ('node_spec', "node-spec.md", "Generate"),
            ('node_code', "node_code.py", "Generate"),
            ('graph_code', "graph_code.py", "Generate"),
            ('main', "main.py", "Generate")
        ]
        
        # Define navigation button names for instructions
        nav_button_names = {
            'state_spec': 'state spec',
            'state_code': 'state code', 
            'node_spec': 'node spec',
            'node_code': 'node code',
            'graph_code': 'graph code',
            'main': 'main'
        }
        
        # Check status of each artifact
        status_data = {}
        graph_folder_path = artifacts_base_dir / graph_name
        for key, file_path, action in artifacts:
            full_path = graph_folder_path / file_path
            if full_path.exists():
                # Get file modification time
                mod_time = datetime.datetime.fromtimestamp(full_path.stat().st_mtime)
                status_data[f"{key}_status"] = f"Generated {mod_time.strftime('%Y-%m-%d %H:%M')}"
            else:
                button_name = nav_button_names[key]
                status_data[f"{key}_status"] = f"Click '{button_name}' to generate"
        
        # Replace placeholders in the overview content
        for key, value in status_data.items():
            overview_content = overview_content.replace(f"{{{key}}}", value)
        
        return jsonify({'content': overview_content}), 200
        
    except Exception as e:
        print(f"Error generating graph overview: {e}", flush=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Disable auto-reload to prevent interruption when files are generated
    app.run(debug=True, use_reloader=False)
