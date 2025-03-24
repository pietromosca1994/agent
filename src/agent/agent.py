import logging
from typing import List, Union
from abc import ABC, abstractmethod
from enum import Enum

from chatbot import BaseChatbot, OpenRouterChatbot, BASE_MODEL, Formats
from models import Tool, ToolCall, Message
from utils import parse_tool_calls

class StatusCode(Enum): 
    SUCCESS=0
    WAITING=1
    ARGPARSE_ERROR=2
    EXECUTION_ERROR=3
    NOT_IMPLEMENTED_ERROR=4

class BaseAgent(BaseChatbot):
    def __init__(self,
                 purpose: str, 
                 verbose: int = logging.INFO):
        
        # purpose of the agent (who are you?)
        self.purpose=purpose
        
        # state containing responses from functions
        self.reset_state()
        
        # list of tools available to the agent
        self.tools: List[Tool]=self.make_tools()

        # setup logger
        self.logger = logging.getLogger(self.__class__.__name__)  # Get a logger unique to the class
        self.logger.setLevel(verbose)  # Set the logging level
        
        # Check if handlers are already added (to prevent duplicate logs)
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

            # Console handler (logs to terminal)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # Optional: File logging (uncomment if needed)
            # file_handler = logging.FileHandler(f"{self.__class__.__name__}.log")
            # file_handler.setFormatter(formatter)
            # self.logger.addHandler(file_handler)

            # Prevent logs from propagating to the root logger
            self.logger.propagate = False

        pass 
    
    def reset_state(self):
        self.state={}

    def get_prompt(self, content: str)->str:
        # prompt=f"{self.purpose}\n{content}"
        prompt=f"{content}"
        return prompt #TODO 
    
    def chat(self,
             content: str,
             format: Formats=Formats.STRING,
             stream: bool = False)->Union[str, dict]:
        
        prompt=self.get_prompt(content)
        
        messages=[Message('system', self.purpose),
                  Message('user', prompt)]
        
        response=super().chat(messages,
                              tools=None, 
                              format=format, 
                              stream=stream)
        return response

    def make_tools(self) -> List[Tool]:
        # Collecting tools from decorated methods
        return [getattr(self, method).tool for method in dir(self) if hasattr(getattr(self, method), 'tool')]

    def call(self, actions: List[dict]):
        '''
        Executes a sequence of tool calls in the provided order.

        Each action is represented as a dictionary containing the tool's function name and its arguments.
        If an argument references the result of a previous action (indicated by a string starting with '$'), 
        the function resolves the dependency by replacing the reference with the corresponding result from the state.

        The function performs the following steps:
        1. Initializes an internal state dictionary to store results of executed actions.
        2. Iterates over the list of actions.
        3. Resolves dependencies in the function arguments by evaluating references to previous results.
        4. Executes the function if it is implemented.
        5. Stores the result of the function execution in the state dictionary if the result is not None.
        6. Logs execution results or errors.

        Args:
            actions (List[dict]): List of action dictionaries where each dictionary contains:
                - "id": Unique action identifier.
                - "function": Dictionary with:
                    - "name": Function name to be executed.
                    - "arguments": Dictionary of function arguments.

        Raises:
            ValueError: If argument resolution fails or referenced results are not available.
        '''

        self.state={}
        action_id_list=[f'${action["id"]}' for action in actions]
        
        for action in actions:
            self.state[action['id']]={'action': action,
                                      'result': None,
                                      'status': StatusCode.WAITING.value}
            
            function_name = action["function"]["name"]
            arguments = action["function"]["arguments"]

            # Resolve dependencies
            try:
                for key, value in arguments.items():
                    if isinstance(value, str):
                        mask=[s in value for s in action_id_list]
                        if any(mask): # value.startswith("$")
                            ref_key = [action_id_list for action_id_list, mask in zip(action_id_list, mask) if mask][0].replace('$', '')
                            if ref_key in self.state.keys():
                                arguments[key] = eval(value.replace(f'${ref_key}', 'self.state[ref_key]["result"]'))
            
            except Exception as e:
                self.state[action['id']]['status']=StatusCode.ARGPARSE_ERROR.value
                self.logger.error('Failed to parse function call arguments\n'
                f'Function name: {function_name}\n'
                f'Arguments: {arguments}\n'
                f'{e}')

            # Execute Function
            method = getattr(self, function_name, None)
            if method:
                try:
                    result = method(**arguments)
                    if result:
                        self.state[action["id"]]['result'] = result
                        self.logger.debug(f"Result from {function_name}: {result}")
                except Exception as e:
                    self.state[action['id']]['status']=StatusCode.EXECUTION_ERROR.value
                    self.logger.error(f'Failed to call method\n'
                                      f'Function name: {function_name}\n'
                                      f'Arguments: {arguments}\n'
                                      f'{e}')
            else:
                self.state[action['id']]['status']=StatusCode.NOT_IMPLEMENTED_ERROR.value
                self.logger.error(f"Function {function_name} not implemented")
                pass

    def execute(self, content: str)->dict:
        '''
        Method to ask the agent to eventually perform actions using the available tools  
        '''
        # make prompt 
        prompt=self.get_prompt(content)
        
        # request sequence of tool actions 
        messages=[Message('system', self.purpose), 
                  Message('user', prompt)]
        response=super().chat(messages, self.tools, Formats.DICT)
        self.logger.debug(response)

        # run functions 
        self.call(response)
        
        response=self.state
        return response

class OpenRouterAgent(BaseAgent, OpenRouterChatbot):
    def __init__(self,
                 purpose: str,
                 api_key: str=None,
                 model: str = BASE_MODEL,
                 verbose: int = logging.INFO):  
        OpenRouterChatbot.__init__(self, model, api_key, verbose)
        BaseAgent.__init__(self, purpose, verbose)


        