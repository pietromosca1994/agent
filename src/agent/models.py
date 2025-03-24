from typing import List, Dict, Union, Literal, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class Message:
    role: str
    content: str

    def to_dict(self):
        """Convert the object to a dictionary."""
        return {"role": self.role, 
                "content": self.content}

    def __repr__(self):
        """String representation for debugging."""
        return f"Message(role='{self.role}', content='{self.content}')"

@dataclass
class Property:
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    nullable: bool = False

    def to_dict(self) -> Dict:
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}

@dataclass
class Parameters:
    type: str
    properties: Dict[str, Property]
    required: List[str]
    additionalProperties: bool = False

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "properties": {k: v.to_dict() for k, v in self.properties.items()},
            "required": self.required
        }

@dataclass
class Function:
    name: str
    description: str
    parameters: Parameters
    strict: bool=False

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_dict()
        }

@dataclass
class Tool:
    type: Literal["function"]
    function: Function

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "function": self.function.to_dict()
        }

@dataclass
class FunctionCall:
    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "arguments": self.arguments
        }

@dataclass
class ToolCall:
    id: str
    type: str
    function: Function

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_dict()
        }
    
@dataclass
class Descriptions:
    function: str
    properties: dict
