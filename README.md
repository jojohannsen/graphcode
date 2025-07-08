# GraphCode

**Transform simple graph notation into complete, runnable LangGraph applications.**

GraphCode is a web-based code generation tool that converts graph specifications written in an intuitive text notation into fully functional LangGraph implementations. Skip the boilerplate and focus on your workflow logic.

## 🚀 Quickstart

### Prerequisites

- Python 3.8+
- LLM API access (OpenAI, Anthropic, OpenRouter, or Google)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/graphcode.git
   cd graphcode
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your LLM settings**
   
   Copy and edit the configuration file:
   ```bash
   cp default.yaml.example default.yaml
   ```
   
   Update `default.yaml` with your API keys and preferred models:
   ```yaml
   llm_providers:
     openai:
       api_key: "your-openai-key"
     anthropic:
       api_key: "your-anthropic-key"
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:5000`

## 🎯 How It Works

### 1. Write Your Graph in Simple Notation

Create a `.txt` file in the `graphs/` directory:

```
# Simple workflow
START -> process_data -> validate
validate -> routing_func(valid, invalid, END)
invalid -> process_data
valid -> save_results -> END
```

### 2. Generate Complete LangGraph Code

GraphCode automatically generates:
- **State definitions** (Python dataclasses)
- **Node implementations** (Python functions)
- **Graph workflow** (LangGraph setup)
- **Main execution file** (Ready to run)

### 3. Advanced Patterns Supported

```
# Parallel processing (fork)
input -> analyze_text, extract_entities, sentiment_check
analyze_text, extract_entities, sentiment_check -> merge_results

# Worker node pattern
coordinator -> worker_node(StateClass.items)

# Human-in-the-loop
review -> human_input -> finalize

# Complex conditional routing
router -> routing_func(path_a, path_b, fallback, END)
```

## 📁 Project Structure

```
graphcode/
├── app.py                 # Flask web application
├── default.yaml           # Configuration (LLM settings, etc.)
├── graphs/                # Your graph specification files
│   └── examples/          # Example graphs
├── templates/             # HTML templates and code generation templates
├── prompts/               # Local prompt files
├── gen_*.py              # Generation pipeline modules
├── mk_utils.py           # Core utilities
└── requirements.txt      # Python dependencies
```

## 🔧 Usage

### Web Interface

1. Start the application: `python app.py`
2. Open `http://localhost:5000`
3. Upload or create a graph specification
4. View the visual representation
5. Generate and download your LangGraph code

### Command Line

Test individual generation steps:

```bash
# Generate state specification
python gen_state_spec.py

# Generate node implementations
python gen_node_code.py

# Full pipeline test
python gen_main.py
```

## 📝 Graph Notation Reference

| Pattern | Syntax | Description |
|---------|--------|-------------|
| Sequential flow | `node1 -> node2` | Direct transition between nodes |
| Entry point | `START -> first_node` | Graph entry point |
| Exit point | `last_node -> END` | Graph termination |
| Parallel fork | `source -> dest1, dest2, dest3` | Split execution to multiple nodes |
| Parallel join | `source1, source2 -> dest` | Merge multiple nodes to one |
| Conditional routing | `node -> routing_func(opt1, opt2, END)` | Branch based on routing function |
| Worker nodes | `coordinator -> worker_node(StateClass.field)` | Process list items in state |
| Human interaction | `node -> human_input -> next_node` | Human-in-the-loop workflows |

### Complete Example

```
# Multi-pattern workflow
START -> initialize -> coordinator

# Parallel processing with worker
coordinator -> worker_node(StateClass.tasks)
worker_node -> validator

# Conditional routing
validator -> check_results(approved, needs_review, failed)
approved -> finalize -> END
needs_review -> human_input -> coordinator
failed -> error_handler -> END

# Comments are preserved in generated code
# This creates a robust processing pipeline
```

## ⚙️ Configuration

The `default.yaml` file controls:

- **LLM Providers**: OpenAI, Anthropic, OpenRouter, Google
- **Models**: Separate configs for spec vs code generation
- **Output Location**: Where generated projects are saved
- **Caching**: File or SQLite-based prompt caching

Example configuration:
```yaml
base_folder: "./generated_projects"
llm_config:
  spec:
    provider: "openai"
    model: "gpt-4"
  code:
    provider: "anthropic"
    model: "claude-3-sonnet"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request

### Development Tips

- Each `gen_*.py` module can be run independently for testing
- Graph files go in `graphs/` directory
- Generated projects are self-contained and ready to run
- Use `CLAUDE.md` for additional development context

## 📄 License

[Add your license here]

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/graphcode/issues)
- **Documentation**: See `CLAUDE.md` for detailed development notes
- **Examples**: Check `graphs/examples/` for sample workflows

---

**Ready to turn your graph ideas into running code?** Start with the quickstart above and have your first LangGraph application running in minutes!