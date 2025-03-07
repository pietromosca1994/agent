import json
from typing import List, Callable
import pickle
from types import SimpleNamespace
from functools import wraps
import inspect
from typing import Union

from models import FunctionCall, ToolCall, Tool, Property, Function, Parameters, Descriptions

def to_json(string: str):
    string = string.replace('\n```', '')
    string = string.replace('```json\n', '')
    return json.dumps(string, indent=2)

def to_dict(string: str):
    string = string.replace('\n```', '')
    string = string.replace('```json\n', '')
    return json.loads(string)

def parse_tool_calls(tool_calls: str) -> List[ToolCall]:
    try:
        # Convert JSON string to Python list
        data = json.loads(tool_calls)
        tool_calls = []

        # Loop through each item in the list
        for item in data:
            func = item.get("function", {})
            arguments_str = func.get("arguments", "{}")
            
            # Parse the arguments JSON string into a dictionary
            arguments = json.loads(arguments_str)

            # Create Function and ToolCall instances
            function = FunctionCall(name=func.get("name"), arguments=arguments)
            tool_call = ToolCall(id=item.get("id"), type=item.get("type"), function=function)
            
            tool_calls.append(tool_call)

        return tool_calls
    
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error parsing tool_calls:\n{e}")
        return []

def save_pickle(obj, file_path):
    """
    Pickle an object and save it to the specified location.

    Args:
        obj: The object to be pickled.
        file_path: The file path where the pickle file will be saved.

    Returns:
        None
    """
    try:
        with open(file_path, 'wb') as file:  # 'wb' means write binary
            pickle.dump(obj, file)
        print(f"Object successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

import pickle

def load_pickle(file_path):
    """
    Load a pickled object from a file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        object: The unpickled object.

    Raises:
        Exception: If an error occurs during loading.
    """
    try:
        with open(file_path, 'rb') as file:  # 'rb' means read binary
            obj = pickle.load(file)
            print(f"Object successfully loaded from {file_path}")
            return obj
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pickle.PickleError:
        print("Error loading the pickle file")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def dict_to_object(dictionary: dict):
    return json.loads(json.dumps(dictionary), object_hook=lambda d: SimpleNamespace(**d))

def generate_tool(descriptions: Descriptions):
    def decorator(func: Callable):
        sig = inspect.signature(func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if name not in ['self', 'cls']:
                param_type = "string" if param.annotation == str else \
                            "number" if param.annotation in [int, float, Union[int, None]] else \
                            "boolean" if param.annotation == bool else \
                            "array" if param.annotation == List else \
                            "any"
                properties[name] = Property(type=param_type, description=descriptions.properties[name], )#
                if param.default == inspect.Parameter.empty:
                    required.append(name)

        parameters = Parameters(
            type="object",
            properties=properties,
            required=required,
            additionalProperties=False
        )

        # Attaching the tool to the function
        func.tool = Tool(
            type="function",
            function=Function(
                name=func.__name__,
                description=descriptions.function,
                strict=True,
                parameters=parameters
            )
        )

        # Wrapping the function with the tool functionality
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
    return decorator
