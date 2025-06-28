import unittest
from typing import List, Tuple, Optional
from enum import Enum

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
    """Represents one or more consecutive comment lines (without the # prefix)"""
    def __init__(self, lines):
        self.lines = lines if isinstance(lines, list) else [lines]
    
    def __repr__(self):
        return f"PythonComment({self.lines})"
    
    def __eq__(self, other):
        return isinstance(other, PythonComment) and self.lines == other.lines

class GraphNotation(GraphPrimitive):
    """Represents a graph notation line containing '->'"""
    def __init__(self, line):
        self.line = line
    
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
        self.implied_state_fields: List[str] = []
        self.implied_source_nodes: List[str] = []
        self.implied_destination_nodes: List[str] = []
        self.errors: List[str] = []
        self.clarifications_needed: List[str] = []
    
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
    
    def __init__(self, primitives: List[GraphPrimitive]):
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
                    preceding_comment = PythonComment([
                        "Comment missing. This comment should provide context and explanation "
                        "for the graph transition that follows."
                    ])
                
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
                self.primitives.append(GraphNotation(line.strip()))
            
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
        """Parse consecutive comment lines and group them together"""
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
        
        return comment_lines, i - 1
    
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

def parse_graph_structure(graph_text):
    """
    Parse graph notation text and return a higher-level GraphStructure object
    
    Args:
        graph_text (str): The graph notation text to parse
        
    Returns:
        GraphStructure: Object with README and comment-notation pairs
    """
    parser = parse_graph_notation(graph_text)
    return GraphStructure(parser.primitives)

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
        "name": "Multiple consecutive comments (only last one pairs with notation)",
        "content": '''"""README"""
# First comment
# Second comment  
# Third comment
A -> B''',
        "expected_readme": "README",
        "expected_pairs": [
            ("Third comment", "A -> B")  # Only the immediately preceding comment pairs
        ]
    }
]

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
        "expected": [PythonComment(["This is a comment"])]
    },
    {
        "name": "Multiple consecutive comments",
        "content": "# First comment\n# Second comment\n# Third comment",
        "expected": [PythonComment(["First comment", "Second comment", "Third comment"])]
    },
    {
        "name": "Simple graph notation",
        "content": "A -> B",
        "expected": [GraphNotation("A -> B")]
    },
    {
        "name": "Complex graph notation",
        "content": "GraphImplementation -> parse_graph",
        "expected": [GraphNotation("GraphImplementation -> parse_graph")]
    },
    {
        "name": "Mixed content - block then comment then notation",
        "content": '''"""Block content"""
# Comment line
A -> B''',
        "expected": [
            TripleQuoteBlock("Block content"),
            PythonComment(["Comment line"]),
            GraphNotation("A -> B")
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
            PythonComment(["We invoke with a string containing the graph notation"]),
            GraphNotation("GraphImplementation -> parse_graph"),
            PythonComment(["Break the graph into block/comment/notation list"]),
            GraphNotation("parse_graph -> make_README")
        ]
    },
    {
        "name": "Empty lines should be ignored",
        "content": '''"""Block"""

# Comment

A -> B''',
        "expected": [
            TripleQuoteBlock("Block"),
            PythonComment(["Comment"]),
            GraphNotation("A -> B")
        ]
    },
    {
        "name": "Multiple separated comment groups",
        "content": '''# First group comment 1
# First group comment 2

# Second group comment 1
# Second group comment 2''',
        "expected": [
            PythonComment(["First group comment 1", "First group comment 2"]),
            PythonComment(["Second group comment 1", "Second group comment 2"])
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
                structure = parse_graph_structure(dataset["content"])
                
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
                    # Check comment (handle both single string and list of strings)
                    if isinstance(expected_comment, str):
                        expected_comment_lines = [expected_comment]
                    else:
                        expected_comment_lines = expected_comment
                    
                    self.assertEqual(
                        pair.comment.lines,
                        expected_comment_lines,
                        f"Comment mismatch in pair {i} for '{dataset['name']}'. "
                        f"Got: {pair.comment.lines}, expected: {expected_comment_lines}"
                    )
                    
                    # Check notation
                    self.assertEqual(
                        pair.notation.line,
                        expected_notation,
                        f"Notation mismatch in pair {i} for '{dataset['name']}'. "
                        f"Got: {pair.notation.line}, expected: {expected_notation}"
                    )
    
    def test_classification_functionality(self):
        """Test classification and analysis functionality"""
        content = '''"""Test graph"""
# Comment 1
A -> B
# Comment 2  
B -> C'''
        
        structure = parse_graph_structure(content)
        
        # Test initial state - no classifications
        self.assertEqual(len(structure.get_unclassified_pairs()), 2)
        self.assertEqual(len(structure.get_classified_pairs(NotationClassification.NODE_TO_NODE)), 0)
        
        # Manually classify first pair
        structure.pairs[0].classification = NotationClassification.NODE_TO_NODE
        structure.pairs[0].implied_state_fields = ["state_a", "state_b"]
        structure.pairs[0].implied_source_nodes = ["A"]
        structure.pairs[0].implied_destination_nodes = ["B"]
        
        # Test classification queries
        self.assertEqual(len(structure.get_unclassified_pairs()), 1)
        self.assertEqual(len(structure.get_classified_pairs(NotationClassification.NODE_TO_NODE)), 1)
        
        # Test analysis data
        classified_pair = structure.get_classified_pairs(NotationClassification.NODE_TO_NODE)[0]
        self.assertEqual(classified_pair.implied_state_fields, ["state_a", "state_b"])
        self.assertEqual(classified_pair.implied_source_nodes, ["A"])
        self.assertEqual(classified_pair.implied_destination_nodes, ["B"])

# Main program
if __name__ == "__main__":
    import argparse
    import sys
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Parse graph notation from a text file')
    parser.add_argument('filepath', nargs='?', help='Path to the text file containing graph notation')
    parser.add_argument('--test', action='store_true', help='Run tests instead of parsing a file')
    parser.add_argument('--structure', action='store_true', help='Show high-level structure analysis instead of primitives')
    
    args = parser.parse_args()
    
    if args.test:
        # Run tests
        unittest.main(argv=[''], exit=False, verbosity=2)
    elif args.filepath:
        try:
            # Read the file content
            with open(args.filepath, 'r', encoding='utf-8') as file:
                graph_text = file.read()
            
            if args.structure:
                # Parse and show structure
                structure = parse_graph_structure(graph_text)
                
                print(f"Graph Structure Analysis for '{args.filepath}':")
                print("=" * 50)
                
                print("\nREADME:")
                print("-" * 20)
                print(structure.readme)
                
                print(f"\nComment-Notation Pairs ({len(structure.pairs)}):")
                print("-" * 30)
                for i, pair in enumerate(structure.pairs, 1):
                    print(f"\n{i}. Comment: {pair.comment.lines}")
                    print(f"   Notation: {pair.notation.line}")
                    print(f"   Classification: {pair.classification}")
                    if pair.implied_state_fields:
                        print(f"   State Fields: {pair.implied_state_fields}")
                    if pair.implied_source_nodes:
                        print(f"   Source Nodes: {pair.implied_source_nodes}")
                    if pair.implied_destination_nodes:
                        print(f"   Destination Nodes: {pair.implied_destination_nodes}")
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