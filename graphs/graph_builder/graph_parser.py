import unittest
from typing import List, Tuple

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

# Main program
if __name__ == "__main__":
    import argparse
    import sys
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Parse graph notation from a text file')
    parser.add_argument('filepath', nargs='?', help='Path to the text file containing graph notation')
    parser.add_argument('--test', action='store_true', help='Run tests instead of parsing a file')
    
    args = parser.parse_args()
    
    if args.test:
        # Run tests
        unittest.main(argv=[''], exit=False, verbosity=2)
    elif args.filepath:
        try:
            # Read the file content
            with open(args.filepath, 'r', encoding='utf-8') as file:
                graph_text = file.read()
            
            # Parse the graph
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