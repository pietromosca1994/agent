import os 
import requests
from typing import Tuple, List, Union
from time import sleep
import logging
from abc import ABC, abstractmethod
import pandas as pd
import json
from enum import Enum

from .models import Message, Tool
from .utils import to_json, to_dict

BASE_MODEL="deepseek/deepseek-chat:free"

class Formats(Enum):
    STRING='string'
    JSON='json'
    DICT='dict'

class BaseChatbot(ABC):
    def __init__(self, 
                 verbose=logging.INFO):
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
    
    @abstractmethod
    def chat(self, 
             messages: List[Message],
             tools: List[Tool]=None,
             format: Formats=Formats.STRING,
             stream: bool = False):
        pass 

    def _make_get_request(self, url: Union[str, bytes], *args, **kwargs)->Tuple[str, int]:
        response=requests.get(url, *args, **kwargs)
        
        if response.status_code == 200:
            self.logger.debug(f"Success calling {url} with {kwargs}")
        else:
            self.logger.error("Error:", response.status_code, response.text)

        return response, response.status_code
    
    def _make_post_request(self, url: Union[str, bytes], *args, **kwargs)->Tuple[str, int]:
        response=requests.post(url, *args, **kwargs)
        if response.status_code == 200:
            self.logger.debug(f"Success calling {url} with {kwargs}")
        else:
            self.logger.error("Error:", response.status_code, response.text)
        return response, response.status_code

class OpenRouterChatbot(BaseChatbot):
    def __init__(self, 
                 model:str = BASE_MODEL, 
                 api_key: str = None,
                 verbose: int = logging.INFO):
        super().__init__(verbose)

        if api_key is None:
            self.api_key=os.getenv('OPENROUTER_API_KEY')
        else:
            self.api_key=api_key
        
        if self.api_key is None:
            raise ValueError('Provide OPENROUTER_API_KEY')
        
        self.model=model
        self.is_model_free=self.model in self.get_free_model_list()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        model_id_list=self.get_model_list()
        if value not in model_id_list:
            model_list_str=[f"{model_id}\n" for model_id in model_id_list]
            raise ValueError(f'The selected model should be in\n{model_list_str}')
        else:
            self._model=value
            self.logger.info(f'Using the model {self._model}') 

    def get_model_info(self)->pd.DataFrame:
        url = "https://openrouter.ai/api/v1/models"
        response, status_code =self._make_get_request(url=url)
        return pd.json_normalize(response.json()['data'], sep='_')
    
    def get_model_list(self)->List[str]:
        model_info=self.get_model_info()
        model_id_list=model_info['id'].unique().tolist()
        model_id_list.sort()
        return model_id_list
    
    def get_free_model_info(self)->pd.DataFrame:
        model_info=self.get_model_info()
        return model_info[model_info['pricing_prompt'].astype('float')==0]
    
    def get_free_model_list(self)->List[str]:
        model_info=self.get_free_model_info()
        model_id_list=model_info['id'].unique().tolist()
        model_id_list.sort()
        return model_id_list

    @staticmethod
    def add_tools(tools: list[Tool]=None):
        prompt='''Based on the tools provided in the **Tools Description**, provide as a response *only** containing
        a sequence of function calls structured as an array of objects with the following **json** format: 
        [{
            id: <unique_id>,
            type: "function",
            function: {
                name: <function_name>,
                arguments: <arguments_in_json_format>
            }
        }]\n If a function produces results that are required by the next function reference the result using the "$id" syntax where id is the unique identifier of the previous function call.'''
        
        prompt+="### Tools Description\n"
        prompt+=json.dumps([tool.to_dict() for tool in tools], indent=2)
        return prompt
    
    def chat(self, 
             messages: List[Message],
             tools: list[Tool]=None,
             format: Formats=Formats.STRING,
             stream: bool = False)->Union[str, dict]:       
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # free models often don't support tools calling
        # Ref: https://openrouter.ai/docs/api-reference/overview
        if self.is_model_free:
            # add tools to the prompt
            if tools:
                for message in messages:
                    if message.role == 'user':
                        message.content+=self.add_tools(tools)

            data = {
                "model": self.model,
                "messages": [message.to_dict() for message in messages],
                "stream": stream
            }
        else:
            data = {
                "model": self.model,
                "messages": [message.to_dict() for message in messages],
                "stream": stream
            }
            if tools: 
                data["tools"]=[tool.to_dict() for tool in tools]

        self.logger.debug(json.dumps(data['messages']))

        # if tools:
        #     data['tools'] =[tool.to_dict() for tool in tools]

        url='https://openrouter.ai/api/v1/chat/completions'
        
        # loop till a response is generated or till the response doesn't satisfy the format
        content=''
        tool_calls=''
        while (content=='')&(tool_calls==''):
            response, status_code= self._make_post_request(url=url, 
                                                           json=data, 
                                                           headers=headers)
            json_response=response.json()
            
            if 'error' in json_response:
                content=None
                raise ValueError(f'Error in querying LLM: {json_response["error"]["message"]}')
            
            if 'content' in json_response['choices'][0]['message'].keys():
                content=json_response['choices'][0]['message']['content']
            
            if 'tool_calls' in json_response['choices'][0]['message'].keys():
                tool_calls=json_response['choices'][0]['message']['tool_calls']

            # if format==format.STRING

            if format==format.JSON:
                try: 
                    content=to_json(content)
                except Exception as e:
                    self.logger.warning(f'Failed to convert content to json format\n{e}')
                    content=''
            
            if format==format.DICT:
                try: 
                    content=to_dict(content)
                except Exception as e:
                    self.logger.warning(f'Failed to convert content to dict format\n{e}')
                    content=''

            if content=='':
                sleep_time=5
                self.logger.debug(f'Retrying in {sleep_time}s')
                sleep(sleep_time)

        return content #TODO add tool_calls
