import unittest
from typing import List, Tuple, Optional
from enum import Enum
import json

# Note: llm_cache module is imported in main() when LLM features are used
# Import StateField and FieldCategory
try:
    from state_field import StateField, FieldCategory
    HAVE_STATE_FIELD = True
except ImportError:
    HAVE_STATE_FIELD = False
    StateField = None
    FieldCategory = None

class GraphPrimitive:
    """Base class for graph primitives"""
    pass

class TripleQuoteBlock(GraphPrimitive):
    """Represents a triple-quoted string block"""
    def __init__(self, content):
        self.content = content
    
    def __repr__(self):
        return f"TripleQuoteBlock({repr(self.content)})"
    
    def __eq__(self, other):
        return isinstance(other, TripleQuoteBlock) and self.content == other.content

class PythonComment(GraphPrimitive):
    """Represents one or more consecutive comment lines (without the # prefix) as a single text string"""
    def __init__(self, text):
        self.text = text if isinstance(text, str) else '\n'.join(text)
    
    def __repr__(self):
        return f"PythonComment({repr(self.text)})"
    
    def __eq__(self, other):
        return isinstance(other, PythonComment) and self.text == other.text

class GraphNotation(GraphPrimitive):
    """Represents a graph notation line containing '->'"""
    def __init__(self, line, is_start_notation=False):
        self.line = line
        self._is_start = is_start_notation
        self._lhs = None
        self._rhs = None
        self._parse_line()
    
    def _parse_line(self):
        """Parse the line to extract LHS and RHS"""
        if "->" in self.line:
            parts = self.line.split("->", 1)
            self._lhs = parts[0].strip()
            self._rhs = parts[1].strip()
        else:
            self._lhs = self.line.strip()
            self._rhs = ""
    
    @property
    def lhs(self):
        """Text preceding '->'"""
        return self._lhs
    
    @property
    def rhs(self):
        """Text after '->'"""
        return self._rhs
    
    @property
    def is_start(self):
        """True if this is the very first GraphNotation (LHS=State Class, RHS=START node)"""
        return self._is_start
    
    @property
    def rhs_parameters(self):
        """List of comma-separated parameters within parentheses on RHS, always returns list"""
        if not self.rhs:
            return []
        
        # Look for parentheses in RHS
        paren_start = self.rhs.find('(')
        paren_end = self.rhs.rfind(')')
        
        if paren_start == -1 or paren_end == -1 or paren_start >= paren_end:
            return []
        
        # Extract content between parentheses
        params_str = self.rhs[paren_start + 1:paren_end]
        
        if not params_str.strip():
            return []
        
        # Split by comma and strip whitespace
        return [param.strip() for param in params_str.split(',') if param.strip()]
    
    @property
    def is_worker(self):
        """True if worker (indicated by single RHS param ending with '*')"""
        params = self.rhs_parameters
        return len(params) == 1 and params[0].endswith('*')
    
    @property
    def is_conditional_edge(self):
        """True if not a worker and has at least 2 rhs_parameters"""
        return not self.is_worker and len(self.rhs_parameters) >= 2
    
    @property
    def source_nodes(self):
        """List of comma-separated words preceding '->'"""
        if not self.lhs:
            return []
        return [node.strip() for node in self.lhs.split(',') if node.strip()]
    
    @property
    def worker_function(self):
        """If is_worker, RHS excluding parentheses and content, otherwise None"""
        if not self.is_worker:
            return None
        
        # Find parentheses and return everything before them
        paren_start = self.rhs.find('(')
        if paren_start == -1:
            return self.rhs
        
        return self.rhs[:paren_start].strip()
    
    @property
    def rhs_function(self):
        """Extract function name from RHS (text before parentheses), or None if no parentheses"""
        if not self.rhs:
            return None
        
        # Find parentheses and return everything before them
        paren_start = self.rhs.find('(')
        if paren_start == -1:
            return None
        
        return self.rhs[:paren_start].strip()
    
    def dest_nodes(self, classification=None):
        """Destination nodes based on NotationClassification"""
        if classification == NotationClassification.NODE_TO_NODE:
            return [self.rhs] if self.rhs else []
        
        elif classification == NotationClassification.NODE_TO_WORKER:
            worker_func = self.worker_function
            return [worker_func] if worker_func else []
        
        elif classification == NotationClassification.NODE_TO_PARALLEL_NODES:
            # List of comma-separated values in RHS
            if not self.rhs:
                return []
            return [node.strip() for node in self.rhs.split(',') if node.strip()]
        
        elif classification == NotationClassification.NODE_TO_CONDITIONAL_EDGE:
            return self.rhs_parameters
        
        else:
            # Default case - return RHS as single item
            return [self.rhs] if self.rhs else []
    
    def __repr__(self):
        return f"GraphNotation({repr(self.line)})"
    
    def __eq__(self, other):
        return isinstance(other, GraphNotation) and self.line == other.line

class NotationClassification(Enum):
    """Classification types for graph notations"""
    NODE_TO_NODE = "NodeToNode"
    NODE_TO_WORKER = "NodeToWorker" 
    NODE_TO_PARALLEL_NODES = "NodeToParallelNodes"
    NODE_TO_CONDITIONAL_EDGE = "NodeToConditionalEdge"

class CommentNotationPair:
    """Represents a paired comment and notation with analysis data"""
    
    def __init__(self, comment: PythonComment, notation: GraphNotation):
        self.comment = comment
        self.notation = notation
        
        # Analysis data - initialized as empty, to be populated by analysis
        self.classification: Optional[NotationClassification] = None
        self.state_class: Optional[str] = None
        self.conditional_edge_function: Optional[str] = None
        self.conditional_edge_reasoning: Optional[str] = None
        self.implied_state_fields: List[str] = []
        self.errors: List[str] = []
        self.clarifications_needed: List[str] = []
    
    @property 
    def source_nodes(self):
        """Source nodes - empty list if this is a start notation, otherwise from notation"""
        if self.notation.is_start:
            return []
        return self.notation.source_nodes
    
    def __repr__(self):
        return f"CommentNotationPair(comment={self.comment}, notation={self.notation}, classification={self.classification})"
    
    def __eq__(self, other):
        if not isinstance(other, CommentNotationPair):
            return False
        return (self.comment == other.comment and 
                self.notation == other.notation and
                self.classification == other.classification)

class GraphStructure:
    """Higher-level representation of parsed graph with README and comment-notation pairs"""
    
    def __init__(self, primitives: List[GraphPrimitive], llm=None):
        self.llm = llm
        self.readme = self._extract_readme(primitives)
        self.pairs = self._create_pairs(primitives)
    
    def _extract_readme(self, primitives: List[GraphPrimitive]) -> str:
        """Extract README from first TripleQuoteBlock or provide default message"""
        for primitive in primitives:
            if isinstance(primitive, TripleQuoteBlock):
                return primitive.content
        
        return ("README missing. This should contain markdown documentation that describes "
                "the purpose and structure of this graph. The README helps users understand "
                "what this graph does and how it works.")
    
    def _create_pairs(self, primitives: List[GraphPrimitive]) -> List[CommentNotationPair]:
        """Create comment-notation pairs from primitives"""
        pairs = []
        
        # Filter out the first TripleQuoteBlock (used for README)
        filtered_primitives = []
        found_first_block = False
        
        for primitive in primitives:
            if isinstance(primitive, TripleQuoteBlock) and not found_first_block:
                found_first_block = True
                continue  # Skip the first block as it's used for README
            filtered_primitives.append(primitive)
        
        # Process remaining primitives to create pairs
        i = 0
        while i < len(filtered_primitives):
            primitive = filtered_primitives[i]
            
            if isinstance(primitive, GraphNotation):
                # Look backwards for a preceding PythonComment
                preceding_comment = None
                
                # Check the immediately preceding primitive
                if i > 0 and isinstance(filtered_primitives[i-1], PythonComment):
                    preceding_comment = filtered_primitives[i-1]
                
                # If no preceding comment found, create a default one
                if preceding_comment is None:
                    preceding_comment = PythonComment(
                        "Comment missing. This comment should provide context and explanation "
                        "for the graph transition that follows."
                    )
                
                # Create the pair
                pair = CommentNotationPair(preceding_comment, primitive)
                pairs.append(pair)
            
            i += 1
        
        return pairs
    
    def get_notations_count(self) -> int:
        """Return the number of notation pairs"""
        return len(self.pairs)
    
    def get_classified_pairs(self, classification: NotationClassification) -> List[CommentNotationPair]:
        """Return pairs with specific classification"""
        return [pair for pair in self.pairs if pair.classification == classification]
    
    def get_unclassified_pairs(self) -> List[CommentNotationPair]:
        """Return pairs that haven't been classified yet"""
        return [pair for pair in self.pairs if pair.classification is None]
    
    def _generate_implied_state_fields(self, pair: CommentNotationPair, current_state_fields: List = None) -> List:
        """
        Generate implied state fields using an LLM prompt.
        Returns StateField objects if available, otherwise string field names.
        """
        if current_state_fields is None:
            current_state_fields = []
            
        if not self.llm:
            # Fallback to placeholder logic if LLM is not available
            fields = []
            comment_words = pair.comment.text.lower().split()
            notation_line = pair.notation.line.lower()
            field_indicators = ['state', 'data', 'input', 'output', 'result', 'content', 'message', 'status']
            for indicator in field_indicators:
                if indicator in comment_words or indicator in notation_line:
                    fields.append(f"{indicator}_field")
            if pair.source_nodes:
                fields.append(f"{pair.source_nodes[0]}_data")
            return fields[:3]

        # Read graph-notation.md content
        try:
            with open('graph-notation.md', 'r', encoding='utf-8') as f:
                graph_notation_spec = f.read()
        except FileNotFoundError:
            graph_notation_spec = "Graph notation specification not available."
        
        # Format current state fields for the prompt
        current_fields_text = "No existing state fields."
        if current_state_fields:
            if HAVE_STATE_FIELD and all(hasattr(f, 'name') for f in current_state_fields):
                # StateField objects
                fields_list = []
                for field in current_state_fields:
                    fields_list.append(f"  - {field.name}: {field.type_annotation} ({field.category.value}) - {field.description}")
                current_fields_text = "\n".join(fields_list)
            else:
                # Simple strings
                current_fields_text = ", ".join(str(f) for f in current_state_fields)
        
        # Determine source and destination nodes
        source_nodes = pair.source_nodes if not pair.notation.is_start else []
        dest_nodes = []
        if pair.classification:
            dest_nodes = pair.notation.dest_nodes(pair.classification)
        elif pair.notation.rhs:
            dest_nodes = [pair.notation.rhs]
            
        prompt = f"""
You are an expert in analyzing LangGraph notation for state field identification. Your task is to identify all state fields that are impacted (read from or written to) by a specific comment/notation pair.

**Graph Notation Specification:**
{graph_notation_spec}

**Current Graph Context:**
- **Graph README:**
{self.readme}

- **Current State Fields:**
{current_fields_text}

**Current Comment/Notation Pair:**
- **Comment:** {pair.comment.text}
- **Notation:** {pair.notation.line}
- **Source Nodes:** {source_nodes}
- **Destination Nodes:** {dest_nodes}
- **Classification:** {pair.classification.value if pair.classification else 'Not classified'}

**Task:**
Analyze this comment/notation pair and identify ALL state fields that are impacted. This includes:
1. Fields that are READ by the source or destination nodes
2. Fields that are WRITTEN by the source or destination nodes  
3. New fields that need to be created based on the transition

For each field, determine:
- name: Valid Python identifier
- type_annotation: Python type (str, bool, int, List[str], Dict, etc.)
- category: One of "human_input", "model_generated", "graph"
- description: Clear explanation of the field's purpose
- read_by_nodes: Which nodes read this field (from source/dest nodes)
- written_by_nodes: Which nodes write this field (from source/dest nodes)

**Output Format:**
Return ONLY a valid JSON array of state field objects. Each object should have this structure:
```json
[
  {{
    "name": "field_name",
    "type_annotation": "str",
    "category": "model_generated",
    "description": "Description of what this field contains",
    "read_by_nodes": ["node1", "node2"],
    "written_by_nodes": ["node3"]
  }}
]
```

**Important:**
- Return valid JSON only, no explanatory text
- Include both existing fields that are impacted AND new fields that need to be created
- Use the exact category values: "human_input", "model_generated", or "graph"
- Node names should match the actual source/destination nodes from the notation

**State Fields JSON:**
"""
        
        try:
            response = self.llm.invoke(prompt.strip())
            
            # Check if the response object has a 'content' attribute
            if hasattr(response, 'content'):
                content = response.content.strip()
            else:
                content = str(response).strip()

            # Parse JSON response
            if content:
                try:
                    fields_data = json.loads(content)
                    if HAVE_STATE_FIELD:
                        # Convert to StateField objects
                        state_fields = []
                        for field_dict in fields_data:
                            try:
                                state_field = StateField(
                                    name=field_dict.get('name', ''),
                                    type_annotation=field_dict.get('type_annotation', 'str'),
                                    category=FieldCategory(field_dict.get('category', 'graph')),
                                    description=field_dict.get('description', ''),
                                    read_by_nodes=set(field_dict.get('read_by_nodes', [])),
                                    written_by_nodes=set(field_dict.get('written_by_nodes', []))
                                )
                                state_fields.append(state_field)
                            except Exception as e:
                                pair.errors.append(f"Failed to create StateField from {field_dict}: {e}")
                        return state_fields
                    else:
                        # Return field names only
                        return [field_dict.get('name', '') for field_dict in fields_data if field_dict.get('name')]
                except json.JSONDecodeError as e:
                    pair.errors.append(f"Failed to parse JSON response: {e}. Response was: {content[:200]}...")
                    # Fallback: try to extract field names from response
                    field_names = []
                    for line in content.split('\n'):
                        if '"name"' in line and ':' in line:
                            try:
                                name_part = line.split(':')[1].strip().strip('"').strip(',')
                                if name_part:
                                    field_names.append(name_part)
                            except:
                                continue
                    return field_names[:5]  # Limit fallback results
            
        except Exception as e:
            pair.errors.append(f"LLM call failed for implied state fields: {e}")

        return []
    
    def _generate_state_fields_structured(self, comments: str, notation: str, current_state_fields: List = None) -> List:
        """
        Generate state fields using structured output with StateField model.
        
        Args:
            comments: Comment text for the notation
            notation: Graph notation line
            current_state_fields: List of existing state fields for context
            
        Returns:
            List of StateField objects or empty list if LLM not available
        """
        if not self.llm or not HAVE_STATE_FIELD:
            return []
            
        if current_state_fields is None:
            current_state_fields = []
        
        # Read graph-notation.md content
        try:
            with open('graph-notation.md', 'r', encoding='utf-8') as f:
                graph_notation_spec = f.read()
        except FileNotFoundError:
            graph_notation_spec = "Graph notation specification not available."
        
        # Format current state fields for context
        current_fields_text = "No existing state fields."
        if current_state_fields:
            if all(hasattr(f, 'name') for f in current_state_fields):
                # StateField objects
                fields_list = []
                for field in current_state_fields:
                    fields_list.append(f"  - {field.name}: {field.type_annotation} ({field.category.value}) - {field.description}")
                current_fields_text = "\n".join(fields_list)
            else:
                # Simple strings
                current_fields_text = ", ".join(str(f) for f in current_state_fields)
        
        # Create structured output LLM
        try:
            structured_llm = self.llm._llm.with_structured_output(StateField)
        except AttributeError:
            # Fallback if structured output not available
            return []
        
        prompt = f"""You are an expert in analyzing LangGraph notation for state field identification. 

**Graph Notation Specification:**
{graph_notation_spec}

**Current Graph Context:**
- **Graph README:**
{self.readme}

- **Current State Fields:**
{current_fields_text}

**Current Comment/Notation:**
- **Comment:** {comments}
- **Notation:** {notation}

**Task:**
Analyze this comment/notation pair and identify ONE primary state field that is most directly impacted by this transition. Focus on the most important field that would be read from or written to.

Consider:
1. What data does this transition work with?
2. What field would be most central to this operation?
3. What type of data is implied by the comment and notation?

Return a single StateField with:
- name: Valid Python identifier for the most important field
- type_annotation: Appropriate Python type (str, bool, int, List[str], Dict, etc.)
- category: "human_input", "model_generated", or "graph" 
- description: Clear explanation of the field's purpose
- read_by_nodes: Empty set (will be populated later)
- written_by_nodes: Empty set (will be populated later)

Focus on the PRIMARY field only - the one most essential to this transition.
"""
        
        try:
            result = structured_llm.invoke(prompt.strip())
            return [result] if result else []
        except Exception as e:
            print(f"Structured output failed: {e}")
            return []
    
    def _generate_conditional_edge_reasoning(self, pair: CommentNotationPair) -> str:
        """
        Generate conditional edge reasoning using AI prompt (placeholder implementation)
        
        TODO: Replace with actual AI/LLM call that includes:
        - Graph notation specification  
        - Graph README (self.readme)
        - Current pair comments (pair.comment.text)
        - Current pair notation (pair.notation.line)
        """
        function_name = pair.conditional_edge_function or "unknown"
        options = pair.notation.rhs_parameters
        
        return (f"The {function_name} function evaluates conditions based on the current state "
                f"and routes to one of {len(options)} possible destinations: {', '.join(options)}. "
                f"The routing decision is made by examining state fields and applying business logic "
                f"as described in the comment: '{pair.comment.text}'")
    
    def auto_classify_pairs(self):
        """Automatically classify all unclassified pairs and populate analysis fields"""
        for pair in self.pairs:
            if pair.classification is None:
                notation = pair.notation
                
                # Set state class for start notation
                if notation.is_start:
                    pair.state_class = notation.lhs
                
                # Classify the notation
                if notation.is_worker:
                    pair.classification = NotationClassification.NODE_TO_WORKER
                elif notation.is_conditional_edge:
                    pair.classification = NotationClassification.NODE_TO_CONDITIONAL_EDGE
                    pair.conditional_edge_function = notation.rhs_function
                elif ',' in notation.rhs and not notation.rhs_parameters:
                    # Parallel nodes - comma-separated destinations without parentheses
                    pair.classification = NotationClassification.NODE_TO_PARALLEL_NODES
                else:
                    pair.classification = NotationClassification.NODE_TO_NODE
                
                # Generate implied state fields using AI prompt
                # Pass current state fields for context
                all_current_fields = []
                for p in self.pairs:
                    if hasattr(p, 'implied_state_fields') and p.implied_state_fields:
                        all_current_fields.extend(p.implied_state_fields)
                pair.implied_state_fields = self._generate_implied_state_fields(pair, all_current_fields)
                
                # Generate conditional edge reasoning if needed
                if pair.classification == NotationClassification.NODE_TO_CONDITIONAL_EDGE:
                    pair.conditional_edge_reasoning = self._generate_conditional_edge_reasoning(pair)
    
    def __repr__(self):
        return f"GraphStructure(readme_length={len(self.readme)}, pairs={len(self.pairs)})"

class GraphParser:
    """Parser for graph notation text that extracts blocks, comments, and notations"""
    
    def __init__(self):
        self.primitives = []
    
    def parse(self, graph_text):
        """
        Parse graph notation text and return a GraphParser object with primitives list
        
        Args:
            graph_text (str): The graph notation text to parse
            
        Returns:
            GraphParser: Object containing list of parsed primitives
        """
        self.primitives = []
        lines = graph_text.split('\n')
        i = 0
        first_notation_found = False
        
        while i < len(lines):
            line = lines[i]
            
            # Skip empty lines
            if not line.strip():
                i += 1
                continue
            
            # Check for triple-quoted string (block)
            if '"""' in line:
                block_content, new_i = self._parse_block(lines, i)
                if block_content is not None:
                    self.primitives.append(TripleQuoteBlock(block_content))
                    i = new_i + 1
                    continue
            
            # Check for comment line
            if line.strip().startswith('#'):
                comment_lines, new_i = self._parse_comment_group(lines, i)
                self.primitives.append(PythonComment(comment_lines))
                i = new_i + 1
                continue
            
            # Check for notation line (contains ->)
            if '->' in line:
                # Mark first notation as start notation
                is_start = not first_notation_found
                first_notation_found = True
                self.primitives.append(GraphNotation(line.strip(), is_start_notation=is_start))
            
            i += 1
        
        return self
    
    def _parse_block(self, lines, start_idx):
        """Parse a triple-quoted string block"""
        line = lines[start_idx]
        
        # Find the opening triple quotes
        triple_quote_pos = line.find('"""')
        if triple_quote_pos == -1:
            return None, start_idx
        
        # Check if it's a single-line block
        after_quotes = line[triple_quote_pos + 3:]
        closing_pos = after_quotes.find('"""')
        if closing_pos != -1:
            # Single line block
            content = after_quotes[:closing_pos]
            return content, start_idx
        
        # Multi-line block - collect content until closing quotes
        content_lines = []
        
        # Add remainder of first line after opening quotes
        remaining_first_line = line[triple_quote_pos + 3:]
        if remaining_first_line:  # Only add if there's content after """
            content_lines.append(remaining_first_line)
        
        i = start_idx + 1
        
        while i < len(lines):
            current_line = lines[i]
            if '"""' in current_line:
                # Found closing triple quotes
                closing_pos = current_line.find('"""')
                before_closing = current_line[:closing_pos]
                if before_closing:  # Only add if there's content before """
                    content_lines.append(before_closing)
                break
            else:
                content_lines.append(current_line)
            i += 1
        
        # Join content and strip leading/trailing whitespace
        content = '\n'.join(content_lines).strip()
        return content, i
    
    def _parse_comment_group(self, lines, start_idx):
        """Parse consecutive comment lines and combine them into a single text string"""
        comment_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#'):
                # Remove the # and any immediately following space
                comment_text = line[1:].lstrip(' ')
                comment_lines.append(comment_text)
                i += 1
            else:
                break
        
        # Combine all comment lines into a single string with newlines
        combined_text = '\n'.join(comment_lines)
        return combined_text, i - 1
    
    def get_blocks(self):
        """Return all block primitives"""
        return [p for p in self.primitives if isinstance(p, TripleQuoteBlock)]
    
    def get_comments(self):
        """Return all comment primitives"""
        return [p for p in self.primitives if isinstance(p, PythonComment)]
    
    def get_notations(self):
        """Return all notation primitives"""
        return [p for p in self.primitives if isinstance(p, GraphNotation)]
    
    def __repr__(self):
        return f"GraphParser(primitives={len(self.primitives)})"

def parse_graph_notation(graph_text):
    """
    Parse graph notation text and return a GraphParser object
    
    Args:
        graph_text (str): The graph notation text to parse
        
    Returns:
        GraphParser: Object containing list of parsed primitives (blocks, comments, notations)
    """
    parser = GraphParser()
    return parser.parse(graph_text)

def parse_graph_structure(graph_text, llm=None):
    """
    Parse graph notation text and return a higher-level GraphStructure object
    
    Args:
        graph_text (str): The graph notation text to parse
        llm: Optional LLM instance for AI-powered analysis
        
    Returns:
        GraphStructure: Object with README and comment-notation pairs
    """
    parser = parse_graph_notation(graph_text)
    return GraphStructure(parser.primitives, llm=llm)

# Test datasets for GraphNotation properties
NOTATION_PROPERTY_DATASETS = [
    {
        "name": "Simple node to node",
        "content": "A -> B",
        "is_start": False,
        "expected": {
            "lhs": "A",
            "rhs": "B", 
            "rhs_parameters": [],
            "is_worker": False,
            "is_conditional_edge": False,
            "source_nodes": ["A"],
            "worker_function": None,
            "rhs_function": None,
            "dest_nodes_node_to_node": ["B"]
        }
    },
    {
        "name": "Start notation",
        "content": "StateClass -> start_node",
        "is_start": True,
        "expected": {
            "lhs": "StateClass",
            "rhs": "start_node",
            "rhs_parameters": [],
            "is_worker": False,
            "is_conditional_edge": False,
            "source_nodes": ["StateClass"],
            "worker_function": None,
            "rhs_function": None,
            "dest_nodes_node_to_node": ["start_node"]
        }
    },
    {
        "name": "Worker notation",
        "content": "node1 -> worker_func(item*)",
        "is_start": False,
        "expected": {
            "lhs": "node1",
            "rhs": "worker_func(item*)",
            "rhs_parameters": ["item*"],
            "is_worker": True,
            "is_conditional_edge": False,
            "source_nodes": ["node1"],
            "worker_function": "worker_func",
            "rhs_function": "worker_func",
            "dest_nodes_worker": ["worker_func"]
        }
    },
    {
        "name": "Conditional edge notation",
        "content": "node1 -> route_decision(option1, option2, option3)",
        "is_start": False,
        "expected": {
            "lhs": "node1",
            "rhs": "route_decision(option1, option2, option3)",
            "rhs_parameters": ["option1", "option2", "option3"],
            "is_worker": False,
            "is_conditional_edge": True,
            "source_nodes": ["node1"],
            "worker_function": None,
            "rhs_function": "route_decision",
            "dest_nodes_conditional": ["option1", "option2", "option3"]
        }
    },
    {
        "name": "Parallel nodes notation",
        "content": "node1 -> node2, node3, node4",
        "is_start": False,
        "expected": {
            "lhs": "node1",
            "rhs": "node2, node3, node4",
            "rhs_parameters": [],
            "is_worker": False,
            "is_conditional_edge": False,
            "source_nodes": ["node1"],
            "worker_function": None,
            "rhs_function": None,
            "dest_nodes_parallel": ["node2", "node3", "node4"]
        }
    },
    {
        "name": "Multiple source nodes",
        "content": "node1, node2, node3 -> target",
        "is_start": False,
        "expected": {
            "lhs": "node1, node2, node3",
            "rhs": "target",
            "rhs_parameters": [],
            "is_worker": False,
            "is_conditional_edge": False,
            "source_nodes": ["node1", "node2", "node3"],
            "worker_function": None,
            "rhs_function": None,
            "dest_nodes_node_to_node": ["target"]
        }
    },
    {
        "name": "Complex conditional with whitespace",
        "content": "decision_node -> choose( path_a , path_b , path_c )",
        "is_start": False,
        "expected": {
            "lhs": "decision_node",
            "rhs": "choose( path_a , path_b , path_c )",
            "rhs_parameters": ["path_a", "path_b", "path_c"],
            "is_worker": False,
            "is_conditional_edge": True,
            "source_nodes": ["decision_node"],
            "worker_function": None,
            "rhs_function": "choose",
            "dest_nodes_conditional": ["path_a", "path_b", "path_c"]
        }
    },
    {
        "name": "Empty parameters",
        "content": "node1 -> func()",
        "is_start": False,
        "expected": {
            "lhs": "node1",
            "rhs": "func()",
            "rhs_parameters": [],
            "is_worker": False,
            "is_conditional_edge": False,
            "source_nodes": ["node1"],
            "worker_function": None,
            "rhs_function": "func",
            "dest_nodes_node_to_node": ["func()"]
        }
    },
    {
        "name": "No parentheses - no rhs_function",
        "content": "node1 -> simple_target",
        "is_start": False,
        "expected": {
            "lhs": "node1",
            "rhs": "simple_target",
            "rhs_parameters": [],
            "is_worker": False,
            "is_conditional_edge": False,
            "source_nodes": ["node1"],
            "worker_function": None,
            "rhs_function": None,
            "dest_nodes_node_to_node": ["simple_target"]
        }
    }
]

# Test datasets for GraphStructure testing
STRUCTURE_TEST_DATASETS = [
    {
        "name": "Complete graph with README and pairs",
        "content": '''"""
This is the README for the graph.
It explains what the graph does.
"""
# First comment about the transition
A -> B
# Second comment
B -> C''',
        "expected_readme": "This is the README for the graph.\nIt explains what the graph does.",
        "expected_pairs": [
            ("First comment about the transition", "A -> B"),
            ("Second comment", "B -> C")
        ]
    },
    {
        "name": "Graph without README",
        "content": '''# Comment for transition
A -> B''',
        "expected_readme": ("README missing. This should contain markdown documentation that describes "
                           "the purpose and structure of this graph. The README helps users understand "
                           "what this graph does and how it works."),
        "expected_pairs": [
            ("Comment for transition", "A -> B")
        ]
    },
    {
        "name": "Graph with notation but no preceding comment",
        "content": '''"""README content"""
A -> B
# Comment for second transition  
B -> C''',
        "expected_readme": "README content",
        "expected_pairs": [
            ("Comment missing. This comment should provide context and explanation for the graph transition that follows.", "A -> B"),
            ("Comment for second transition", "B -> C")
        ]
    },
    {
        "name": "Mixed content with multiple blocks (only first used as README)",
        "content": '''"""This is the README"""
# Comment 1
A -> B
"""This block is ignored"""
# Comment 2
B -> C''',
        "expected_readme": "This is the README",
        "expected_pairs": [
            ("Comment 1", "A -> B"),
            ("Comment 2", "B -> C")
        ]
    },
    {
        "name": "Original graph sample structure",
        "content": '''"""
This graph is the reasoning unit that translate a langgraph graph notation
into langgraph source code.
"""
# We invoke with a string containing the graph notation
GraphImplementation -> parse_graph
# Break the graph into block/comment/notation list
parse_graph -> make_README''',
        "expected_readme": "This graph is the reasoning unit that translate a langgraph graph notation\ninto langgraph source code.",
        "expected_pairs": [
            ("We invoke with a string containing the graph notation", "GraphImplementation -> parse_graph"),
            ("Break the graph into block/comment/notation list", "parse_graph -> make_README")
        ]
    },
    {
        "name": "Multiple consecutive comments (all combined with newlines)",
        "content": '''"""README"""
# First comment
# Second comment  
# Third comment
A -> B''',
        "expected_readme": "README",
        "expected_pairs": [
            ("First comment\nSecond comment\nThird comment", "A -> B")  # All comments combined with newlines
        ]
    }
]

# Test datasets for StateField verification
STATE_FIELD_TEST_DATASETS = [
    {
        "name": "Human input field",
        "comments": "Ask user whether to continue processing",
        "notation": "check_status -> should_continue(yes, no)",
        "expected_field": {
            "name": "user_continue_choice",
            "type_annotation": "bool",
            "category": "human_input",
            "description_contains": ["user", "continue", "choice"],
            "read_by_nodes": set(),
            "written_by_nodes": set()
        }
    },
    {
        "name": "Model generated content",
        "comments": "Generate a joke about the given topic",
        "notation": "get_topic -> generate_joke",
        "expected_field": {
            "name": "generated_joke",
            "type_annotation": "str",
            "category": "model_generated", 
            "description_contains": ["joke", "generated", "content"],
            "read_by_nodes": set(),
            "written_by_nodes": set()
        }
    },
    {
        "name": "Graph calculated data",
        "comments": "Calculate the total score from all inputs",
        "notation": "collect_scores -> calculate_total",
        "expected_field": {
            "name": "total_score",
            "type_annotation": "int",
            "category": "graph",
            "description_contains": ["total", "score", "calculated"],
            "read_by_nodes": set(),
            "written_by_nodes": set()
        }
    },
    {
        "name": "User request processing",
        "comments": "Store the user's search query for processing",
        "notation": "get_query -> process_search",
        "expected_field": {
            "name": "search_query",
            "type_annotation": "str",
            "category": "human_input",
            "description_contains": ["search", "query", "user"],
            "read_by_nodes": set(),
            "written_by_nodes": set()
        }
    },
    {
        "name": "LLM response data",
        "comments": "Generate response based on conversation history",
        "notation": "analyze_context -> generate_response",
        "expected_field": {
            "name": "ai_response",
            "type_annotation": "str",
            "category": "model_generated",
            "description_contains": ["response", "generated", "ai"],
            "read_by_nodes": set(),
            "written_by_nodes": set()
        }
    },
    {
        "name": "Validation result",
        "comments": "Check if the input data is valid and complete",
        "notation": "receive_data -> validate_input",
        "expected_field": {
            "name": "is_valid",
            "type_annotation": "bool", 
            "category": "graph",
            "description_contains": ["valid", "validation", "check"],
            "read_by_nodes": set(),
            "written_by_nodes": set()
        }
    },
    {
        "name": "Processing status",
        "comments": "Track the current processing state and progress",
        "notation": "start_processing -> update_status",
        "expected_field": {
            "name": "processing_status",
            "type_annotation": "str",
            "category": "graph",
            "description_contains": ["status", "processing", "progress"],
            "read_by_nodes": set(),
            "written_by_nodes": set()
        }
    },
    {
        "name": "List data structure",
        "comments": "Collect all error messages encountered during processing",
        "notation": "validate_data -> collect_errors",
        "expected_field": {
            "name": "error_messages",
            "type_annotation": "List[str]",
            "category": "graph",
            "description_contains": ["error", "messages", "collected"],
            "read_by_nodes": set(),
            "written_by_nodes": set()
        }
    }
]

def verify_state_field_match(state_fields: List, expected_field: dict) -> bool:
    """
    Verify that at least one StateField in the list matches the expected field criteria.
    
    Args:
        state_fields: List of StateField objects or field dictionaries
        expected_field: Dictionary with expected field properties
        
    Returns:
        bool: True if a matching field is found, False otherwise
    """
    if not state_fields:
        return False
        
    expected_name = expected_field.get("name")
    expected_type = expected_field.get("type_annotation") 
    expected_category = expected_field.get("category")
    expected_desc_contains = expected_field.get("description_contains", [])
    expected_read_nodes = expected_field.get("read_by_nodes", set())
    expected_write_nodes = expected_field.get("written_by_nodes", set())
    
    for field in state_fields:
        # Handle both StateField objects and dictionaries
        if hasattr(field, 'name'):
            # StateField object
            field_name = field.name
            field_type = field.type_annotation
            field_category = field.category.value if hasattr(field.category, 'value') else str(field.category)
            field_description = field.description.lower()
            field_read_nodes = field.read_by_nodes
            field_write_nodes = field.written_by_nodes
        elif isinstance(field, dict):
            # Dictionary
            field_name = field.get("name", "")
            field_type = field.get("type_annotation", "")
            field_category = field.get("category", "")
            field_description = field.get("description", "").lower()
            field_read_nodes = set(field.get("read_by_nodes", []))
            field_write_nodes = set(field.get("written_by_nodes", []))
        else:
            continue
        
        # Check name match (exact or contains expected name)
        name_match = (field_name == expected_name or 
                     (expected_name and expected_name.lower() in field_name.lower()))
        
        # Check type annotation match
        type_match = field_type == expected_type
        
        # Check category match  
        category_match = field_category == expected_category
        
        # Check description contains expected keywords
        desc_match = True
        if expected_desc_contains:
            desc_match = any(keyword.lower() in field_description for keyword in expected_desc_contains)
        
        # Check node sets (if specified)
        read_nodes_match = expected_read_nodes <= field_read_nodes if expected_read_nodes else True
        write_nodes_match = expected_write_nodes <= field_write_nodes if expected_write_nodes else True
        
        # A field matches if it has good name/type/category match and description keywords
        if ((name_match or field_name) and  # Has some reasonable name
            type_match and 
            category_match and 
            desc_match and
            read_nodes_match and 
            write_nodes_match):
            return True
    
    return False

def get_field_match_details(state_fields: List, expected_field: dict) -> dict:
    """
    Get detailed information about how well state fields match expected criteria.
    
    Args:
        state_fields: List of StateField objects or field dictionaries
        expected_field: Dictionary with expected field properties
        
    Returns:
        dict: Detailed match information for debugging
    """
    if not state_fields:
        return {"match": False, "reason": "No state fields provided"}
    
    expected_name = expected_field.get("name")
    expected_type = expected_field.get("type_annotation")
    expected_category = expected_field.get("category") 
    expected_desc_contains = expected_field.get("description_contains", [])
    
    best_match = None
    best_score = 0
    
    for i, field in enumerate(state_fields):
        # Handle both StateField objects and dictionaries
        if hasattr(field, 'name'):
            field_name = field.name
            field_type = field.type_annotation
            field_category = field.category.value if hasattr(field.category, 'value') else str(field.category)
            field_description = field.description.lower()
        elif isinstance(field, dict):
            field_name = field.get("name", "")
            field_type = field.get("type_annotation", "")
            field_category = field.get("category", "")
            field_description = field.get("description", "").lower()
        else:
            continue
        
        score = 0
        details = {}
        
        # Name scoring
        if field_name == expected_name:
            score += 3
            details["name"] = "exact_match"
        elif expected_name and expected_name.lower() in field_name.lower():
            score += 2  
            details["name"] = "partial_match"
        elif field_name:
            score += 1
            details["name"] = "has_name"
        else:
            details["name"] = "no_name"
        
        # Type scoring
        if field_type == expected_type:
            score += 2
            details["type"] = "exact_match"
        else:
            details["type"] = f"mismatch: got {field_type}, expected {expected_type}"
        
        # Category scoring
        if field_category == expected_category:
            score += 2
            details["category"] = "exact_match"
        else:
            details["category"] = f"mismatch: got {field_category}, expected {expected_category}"
        
        # Description scoring
        desc_matches = [kw for kw in expected_desc_contains if kw.lower() in field_description]
        if len(desc_matches) == len(expected_desc_contains):
            score += 2
            details["description"] = "all_keywords_found"
        elif desc_matches:
            score += 1
            details["description"] = f"partial_keywords: {desc_matches}"
        else:
            details["description"] = "no_keywords_found"
        
        if score > best_score:
            best_score = score
            best_match = {
                "index": i,
                "field": field,
                "score": score,
                "details": details,
                "field_name": field_name,
                "field_type": field_type,
                "field_category": field_category,
                "field_description": field_description[:100]
            }
    
    return {
        "match": best_score >= 7,  # Threshold for considering a good match
        "best_match": best_match,
        "total_fields": len(state_fields),
        "expected": expected_field
    }

# Test datasets for parameterized testing
TEST_DATASETS = [
    {
        "name": "Simple triple quote block",
        "content": '"""This is a block"""',
        "expected": [TripleQuoteBlock("This is a block")]
    },
    {
        "name": "Multi-line triple quote block",
        "content": '''"""
This is a multi-line
block with content
"""''',
        "expected": [TripleQuoteBlock("This is a multi-line\nblock with content")]
    },
    {
        "name": "Single comment line",
        "content": "# This is a comment",
        "expected": [PythonComment("This is a comment")]
    },
    {
        "name": "Multiple consecutive comments",
        "content": "# First comment\n# Second comment\n# Third comment",
        "expected": [PythonComment("First comment\nSecond comment\nThird comment")]
    },
    {
        "name": "Simple graph notation",
        "content": "A -> B",
        "expected": [GraphNotation("A -> B", is_start_notation=True)]  # First notation is start
    },
    {
        "name": "Complex graph notation",
        "content": "GraphImplementation -> parse_graph",
        "expected": [GraphNotation("GraphImplementation -> parse_graph", is_start_notation=True)]  # First notation is start
    },
    {
        "name": "Mixed content - block then comment then notation",
        "content": '''"""Block content"""
# Comment line
A -> B''',
        "expected": [
            TripleQuoteBlock("Block content"),
            PythonComment("Comment line"),
            GraphNotation("A -> B", is_start_notation=True)  # First notation is start
        ]
    },
    {
        "name": "Original graph sample",
        "content": '''"""
This graph is the reasoning unit that translate a langgraph graph notation
into langgraph source code.
"""
# We invoke with a string containing the graph notation
GraphImplementation -> parse_graph
# Break the graph into block/comment/notation list
parse_graph -> make_README''',
        "expected": [
            TripleQuoteBlock("This graph is the reasoning unit that translate a langgraph graph notation\ninto langgraph source code."),
            PythonComment("We invoke with a string containing the graph notation"),
            GraphNotation("GraphImplementation -> parse_graph", is_start_notation=True),  # First notation is start
            PythonComment("Break the graph into block/comment/notation list"),
            GraphNotation("parse_graph -> make_README", is_start_notation=False)  # Second notation is not start
        ]
    },
    {
        "name": "Empty lines should be ignored",
        "content": '''"""Block"""

# Comment

A -> B''',
        "expected": [
            TripleQuoteBlock("Block"),
            PythonComment("Comment"),
            GraphNotation("A -> B", is_start_notation=True)  # First notation is start
        ]
    },
    {
        "name": "Multiple separated comment groups",
        "content": '''# First group comment 1
# First group comment 2

# Second group comment 1
# Second group comment 2''',
        "expected": [
            PythonComment("First group comment 1\nFirst group comment 2"),
            PythonComment("Second group comment 1\nSecond group comment 2")
        ]
    }
]

class TestGraphParser(unittest.TestCase):
    """Parameterized tests for graph parser"""
    
    def test_parser_datasets(self):
        """Test parser against all datasets"""
        for dataset in TEST_DATASETS:
            with self.subTest(dataset=dataset["name"]):
                # Parse the content
                parser = parse_graph_notation(dataset["content"])
                
                # Check number of primitives matches
                self.assertEqual(
                    len(parser.primitives), 
                    len(dataset["expected"]),
                    f"Wrong number of primitives for '{dataset['name']}'. "
                    f"Got {len(parser.primitives)}, expected {len(dataset['expected'])}. "
                    f"Actual: {parser.primitives}"
                )
                
                # Check each primitive matches expected type and content
                for i, (actual, expected) in enumerate(zip(parser.primitives, dataset["expected"])):
                    self.assertEqual(
                        actual, 
                        expected,
                        f"Primitive {i} mismatch in '{dataset['name']}'. "
                        f"Got {actual}, expected {expected}"
                    )
    
    def test_structure_datasets(self):
        """Test GraphStructure against all structure datasets"""
        for dataset in STRUCTURE_TEST_DATASETS:
            with self.subTest(dataset=dataset["name"]):
                # Parse the structure
                structure = parse_graph_structure(dataset["content"], llm=None)
                
                # Check README matches
                self.assertEqual(
                    structure.readme,
                    dataset["expected_readme"],
                    f"README mismatch for '{dataset['name']}'. "
                    f"Got: {repr(structure.readme)}"
                )
                
                # Check number of pairs matches
                self.assertEqual(
                    len(structure.pairs),
                    len(dataset["expected_pairs"]),
                    f"Wrong number of pairs for '{dataset['name']}'. "
                    f"Got {len(structure.pairs)}, expected {len(dataset['expected_pairs'])}"
                )
                
                # Check each pair matches expected comment and notation
                for i, (pair, (expected_comment, expected_notation)) in enumerate(zip(structure.pairs, dataset["expected_pairs"])):
                    # Check comment text
                    self.assertEqual(
                        pair.comment.text,
                        expected_comment,
                        f"Comment mismatch in pair {i} for '{dataset['name']}'. "
                        f"Got: {repr(pair.comment.text)}, expected: {repr(expected_comment)}"
                    )
                    
                    # Check notation
                    self.assertEqual(
                        pair.notation.line,
                        expected_notation,
                        f"Notation mismatch in pair {i} for '{dataset['name']}'. "
                        f"Got: {pair.notation.line}, expected: {expected_notation}"
                    )
    
    def test_notation_properties(self):
        """Test GraphNotation properties against datasets"""
        for dataset in NOTATION_PROPERTY_DATASETS:
            with self.subTest(dataset=dataset["name"]):
                # Create a GraphNotation with the specified is_start value
                notation = GraphNotation(dataset["content"], is_start_notation=dataset["is_start"])
                expected = dataset["expected"]
                
                # Test basic properties
                self.assertEqual(notation.lhs, expected["lhs"], f"LHS mismatch for {dataset['name']}")
                self.assertEqual(notation.rhs, expected["rhs"], f"RHS mismatch for {dataset['name']}")
                self.assertEqual(notation.is_start, dataset["is_start"], f"is_start mismatch for {dataset['name']}")
                self.assertEqual(notation.rhs_parameters, expected["rhs_parameters"], f"rhs_parameters mismatch for {dataset['name']}")
                self.assertEqual(notation.is_worker, expected["is_worker"], f"is_worker mismatch for {dataset['name']}")
                self.assertEqual(notation.is_conditional_edge, expected["is_conditional_edge"], f"is_conditional_edge mismatch for {dataset['name']}")
                self.assertEqual(notation.source_nodes, expected["source_nodes"], f"source_nodes mismatch for {dataset['name']}")
                self.assertEqual(notation.worker_function, expected["worker_function"], f"worker_function mismatch for {dataset['name']}")
                self.assertEqual(notation.rhs_function, expected["rhs_function"], f"rhs_function mismatch for {dataset['name']}")
                
                # Test dest_nodes with different classifications
                if "dest_nodes_node_to_node" in expected:
                    actual = notation.dest_nodes(NotationClassification.NODE_TO_NODE)
                    self.assertEqual(actual, expected["dest_nodes_node_to_node"], f"dest_nodes NODE_TO_NODE mismatch for {dataset['name']}")
                
                if "dest_nodes_worker" in expected:
                    actual = notation.dest_nodes(NotationClassification.NODE_TO_WORKER)
                    self.assertEqual(actual, expected["dest_nodes_worker"], f"dest_nodes NODE_TO_WORKER mismatch for {dataset['name']}")
                
                if "dest_nodes_parallel" in expected:
                    actual = notation.dest_nodes(NotationClassification.NODE_TO_PARALLEL_NODES)
                    self.assertEqual(actual, expected["dest_nodes_parallel"], f"dest_nodes NODE_TO_PARALLEL_NODES mismatch for {dataset['name']}")
                
                if "dest_nodes_conditional" in expected:
                    actual = notation.dest_nodes(NotationClassification.NODE_TO_CONDITIONAL_EDGE)
                    self.assertEqual(actual, expected["dest_nodes_conditional"], f"dest_nodes NODE_TO_CONDITIONAL_EDGE mismatch for {dataset['name']}")
    
    def test_first_notation_is_start(self):
        """Test that first notation in parsed graph is marked as start"""
        content = '''"""README"""
# Comment 1
First -> Second
# Comment 2  
Second -> Third'''
        
        structure = parse_graph_structure(content, llm=None)
        structure.auto_classify_pairs()
        
        # Should have 2 pairs
        self.assertEqual(len(structure.pairs), 2)
        
        # First notation should be marked as start
        self.assertTrue(structure.pairs[0].notation.is_start)
        self.assertFalse(structure.pairs[1].notation.is_start)
        
        # Verify the start notation properties
        start_pair = structure.pairs[0]
        self.assertEqual(start_pair.notation.lhs, "First")
        self.assertEqual(start_pair.notation.rhs, "Second")
        self.assertEqual(start_pair.state_class, "First")  # State class set from LHS
        self.assertEqual(start_pair.source_nodes, [])  # Empty for start notation
        
        # Verify regular notation properties  
        regular_pair = structure.pairs[1]
        self.assertIsNone(regular_pair.state_class)  # No state class for regular notation
        self.assertEqual(regular_pair.source_nodes, ["Second"])  # Normal source nodes
    
    def test_implied_fields_population(self):
        """Test that implied fields are populated during auto-classification"""
        content = '''"""Test graph for state management"""
# Initialize the processing state with input data
StateClass -> process_input
# Route based on input type and validation results
process_input -> route_decision(valid_path, error_path, retry_path)
# Handle successful processing
valid_path -> complete'''
        
        structure = parse_graph_structure(content)
        structure.auto_classify_pairs()
        
        # Should have 3 pairs
        self.assertEqual(len(structure.pairs), 3)
        
        # Test first pair (start notation)
        start_pair = structure.pairs[0]
        self.assertTrue(len(start_pair.implied_state_fields) > 0)  # Should have generated fields
        self.assertIsNone(start_pair.conditional_edge_reasoning)  # Not a conditional edge
        
        # Test second pair (conditional edge)
        conditional_pair = structure.pairs[1]
        self.assertEqual(conditional_pair.classification, NotationClassification.NODE_TO_CONDITIONAL_EDGE)
        self.assertTrue(len(conditional_pair.implied_state_fields) > 0)
        self.assertIsNotNone(conditional_pair.conditional_edge_reasoning)  # Should have reasoning
        self.assertIn("route_decision", conditional_pair.conditional_edge_reasoning)
        
        # Test third pair (regular node)
        regular_pair = structure.pairs[2]
        self.assertEqual(regular_pair.classification, NotationClassification.NODE_TO_NODE)
        self.assertTrue(len(regular_pair.implied_state_fields) > 0)
        self.assertIsNone(regular_pair.conditional_edge_reasoning)  # Not a conditional edge
    
    def test_state_class_and_source_nodes(self):
        """Test that state_class is set correctly and source_nodes are empty for start notation"""
        content = '''"""Test graph"""
# Comment for start notation
MyStateClass -> start_node
# Comment for regular transition
start_node -> next_node'''
        
        structure = parse_graph_structure(content)
        structure.auto_classify_pairs()
        
        # Should have 2 pairs
        self.assertEqual(len(structure.pairs), 2)
        
        # First pair should be start notation with state class set and empty source nodes
        start_pair = structure.pairs[0]
        self.assertTrue(start_pair.notation.is_start)
        self.assertEqual(start_pair.state_class, "MyStateClass")
        self.assertEqual(start_pair.source_nodes, [])  # Empty for start notation
        self.assertEqual(start_pair.notation.source_nodes, ["MyStateClass"])  # Notation still has original value
        
        # Second pair should be regular notation with no state class and normal source nodes
        regular_pair = structure.pairs[1]
        self.assertFalse(regular_pair.notation.is_start)
        self.assertIsNone(regular_pair.state_class)
        self.assertEqual(regular_pair.source_nodes, ["start_node"])  # Normal source nodes
        self.assertEqual(regular_pair.notation.source_nodes, ["start_node"])  # Same as pair
    
    def test_conditional_edge_function_assignment(self):
        """Test that conditional_edge_function gets set correctly during auto-classification"""
        content = '''"""Test graph"""
# Comment for conditional edge
decision_node -> route_choice(path_a, path_b, path_c)
# Comment for regular node
A -> B'''
        
        structure = parse_graph_structure(content)
        structure.auto_classify_pairs()
        
        # Should have 2 pairs
        self.assertEqual(len(structure.pairs), 2)
        
        # First pair should be conditional edge with function set
        conditional_pair = structure.pairs[0]
        self.assertEqual(conditional_pair.classification, NotationClassification.NODE_TO_CONDITIONAL_EDGE)
        self.assertEqual(conditional_pair.conditional_edge_function, "route_choice")
        
        # Second pair should be node-to-node with no conditional function
        node_pair = structure.pairs[1]
        self.assertEqual(node_pair.classification, NotationClassification.NODE_TO_NODE)
        self.assertIsNone(node_pair.conditional_edge_function)
    
    def test_state_field_verification(self):
        """Test StateField verification functions with test datasets"""
        for dataset in STATE_FIELD_TEST_DATASETS:
            with self.subTest(dataset=dataset["name"]):
                # Test verify_state_field_match with a matching field
                if HAVE_STATE_FIELD:
                    # Create a StateField that should match
                    test_field = StateField(
                        name=dataset["expected_field"]["name"],
                        type_annotation=dataset["expected_field"]["type_annotation"],
                        category=FieldCategory(dataset["expected_field"]["category"]),
                        description=f"Test field for {' '.join(dataset['expected_field']['description_contains'])}",
                        read_by_nodes=dataset["expected_field"]["read_by_nodes"],
                        written_by_nodes=dataset["expected_field"]["written_by_nodes"]
                    )
                    
                    # Test that verification passes
                    self.assertTrue(
                        verify_state_field_match([test_field], dataset["expected_field"]),
                        f"StateField verification failed for {dataset['name']}"
                    )
                    
                    # Test detailed matching
                    details = get_field_match_details([test_field], dataset["expected_field"])
                    self.assertTrue(details["match"], f"Detailed matching failed for {dataset['name']}")
                    self.assertGreaterEqual(details["best_match"]["score"], 7, 
                                          f"Match score too low for {dataset['name']}")
                
                # Test with dictionary format
                test_field_dict = {
                    "name": dataset["expected_field"]["name"],
                    "type_annotation": dataset["expected_field"]["type_annotation"],
                    "category": dataset["expected_field"]["category"],
                    "description": f"Test field for {' '.join(dataset['expected_field']['description_contains'])}",
                    "read_by_nodes": list(dataset["expected_field"]["read_by_nodes"]),
                    "written_by_nodes": list(dataset["expected_field"]["written_by_nodes"])
                }
                
                self.assertTrue(
                    verify_state_field_match([test_field_dict], dataset["expected_field"]),
                    f"Dictionary verification failed for {dataset['name']}"
                )
                
                # Test that verification fails with wrong field
                wrong_field = {
                    "name": "wrong_name",
                    "type_annotation": "wrong_type",
                    "category": "graph",
                    "description": "wrong description",
                    "read_by_nodes": [],
                    "written_by_nodes": []
                }
                
                self.assertFalse(
                    verify_state_field_match([wrong_field], dataset["expected_field"]),
                    f"Verification should have failed for wrong field in {dataset['name']}"
                )

    

# Main program
if __name__ == "__main__":
    import argparse
    import sys
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Parse graph notation from a text file')
    parser.add_argument('filepath', nargs='?', help='Path to the text file containing graph notation')
    parser.add_argument('--test', action='store_true', help='Run tests instead of parsing a file')
    parser.add_argument('--structure', action='store_true', help='Show high-level structure analysis instead of primitives')
    
    # LLM configuration arguments
    parser.add_argument('--llm-provider', choices=['openai', 'anthropic', 'openrouter'], 
                       help='LLM provider for AI analysis')
    parser.add_argument('--llm-model', help='LLM model name (e.g., gpt-4, claude-3-sonnet-20240229)')
    parser.add_argument('--llm-cache', choices=['file', 'sqlite'], default='file',
                       help='LLM cache type (default: file)')
    
    args = parser.parse_args()
    
    if args.test:
        # Run tests
        unittest.main(argv=[''], exit=False, verbosity=2)
    elif args.filepath:
        # Set up LLM if requested
        llm = None
        if args.llm_provider and args.llm_model:
            try:
                # Import the LLM cache module
                from llm_cache import make_llm
                llm = make_llm(args.llm_provider, args.llm_model, args.llm_cache)
                print(f"Using LLM: {args.llm_provider}/{args.llm_model} with {args.llm_cache} cache")
            except ImportError:
                print("Warning: llm_cache module not found. AI analysis will use placeholder logic.")
            except Exception as e:
                print(f"Warning: Failed to initialize LLM: {e}. AI analysis will use placeholder logic.")
        
        try:
            # Read the file content
            with open(args.filepath, 'r', encoding='utf-8') as file:
                graph_text = file.read()
            
            if args.structure:
                # Parse and show structure with optional LLM
                structure = parse_graph_structure(graph_text, llm=llm)
                
                # Auto-classify all pairs before displaying
                structure.auto_classify_pairs()
                
                print(f"Graph Structure Analysis for '{args.filepath}':")
                print("=" * 50)
                
                print("\nREADME:")
                print("-" * 20)
                print(structure.readme)
                
                print(f"\nComment-Notation Pairs ({len(structure.pairs)}):")
                print("-" * 30)
                for i, pair in enumerate(structure.pairs, 1):
                    print(f"\n{i}. {pair.notation.line}")
                    print(f"   Comment: {repr(pair.comment.text)}")
                    print(f"   LHS: '{pair.notation.lhs}' | RHS: '{pair.notation.rhs}'")
                    print(f"   Is Start: {pair.notation.is_start}")
                    print(f"   State Class: {pair.state_class}")
                    print(f"   Source Nodes: {pair.source_nodes}")
                    print(f"   RHS Parameters: {pair.notation.rhs_parameters}")
                    print(f"   Is Worker: {pair.notation.is_worker}")
                    print(f"   Is Conditional Edge: {pair.notation.is_conditional_edge}")
                    print(f"   Worker Function: {pair.notation.worker_function}")
                    print(f"   RHS Function: {pair.notation.rhs_function}")
                    print(f"   Classification: {pair.classification.value if pair.classification else None}")
                    print(f"   Conditional Edge Function: {pair.conditional_edge_function}")
                    if pair.classification:
                        dest_nodes = pair.notation.dest_nodes(pair.classification)
                        print(f"   Destination Nodes: {dest_nodes}")
                    
                    # Analysis fields
                    print(f"   Implied State Fields: {pair.implied_state_fields}")
                    if pair.conditional_edge_reasoning:
                        print(f"   Conditional Edge Reasoning: {pair.conditional_edge_reasoning}")
                    
                    if pair.errors:
                        print(f"   Errors: {pair.errors}")
                    if pair.clarifications_needed:
                        print(f"   Clarifications: {pair.clarifications_needed}")
            else:
                # Parse and show primitives (original behavior)
                graph = parse_graph_notation(graph_text)
                
                # Display results
                print(f"Parsed {len(graph.primitives)} primitives from '{args.filepath}':")
                print(f"- {len(graph.get_blocks())} blocks")
                print(f"- {len(graph.get_comments())} comments")
                print(f"- {len(graph.get_notations())} notations")
                
                print("\nPrimitives:")
                for i, primitive in enumerate(graph.primitives):
                    print(f"{i+1}. {primitive}")
                
        except FileNotFoundError:
            print(f"Error: File '{args.filepath}' not found.", file=sys.stderr)
            sys.exit(1)
        except IOError as e:
            print(f"Error reading file '{args.filepath}': {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
