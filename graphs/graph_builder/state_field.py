from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Set
from enum import Enum

class FieldCategory(str, Enum):
    """Categories of state fields in graph specifications."""
    HUMAN_INPUT = "human_input"
    MODEL_GENERATED = "model_generated" 
    GRAPH_CALCULATED = "graph"



class StateField(BaseModel):
    """Represents a single field in the graph state."""
    
    name: str = Field(description="Field name (valid Python identifier)")
    type_annotation: str = Field(description="Python type annotation as string")
    category: FieldCategory = Field(description="Category of field")
    description: str = Field(description="Purpose and usage description")
    read_by_nodes: Set[str] = Field(
        default_factory=set,
        description="Set of node names that read this field"
    )
    written_by_nodes: Set[str] = Field(
        default_factory=set,
        description="Set of node names that write to this field"
    )
