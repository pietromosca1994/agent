#%%
# import modules
import dotenv
import os
from pathlib import Path
from glob import glob
import sys
import pandas as pd
import logging
import requests
from typing import List

sys.path.append('../src/') 
from agent import OpenRouterAgent
from utils import generate_tool
from models import Descriptions, Tool
from chatbot import BASE_MODEL

_=dotenv.load_dotenv(override=True)
print(f'Openrouter API key: {os.environ["OPENROUTER_API_KEY"]}')

#%%
# create a weather agent 
class WeatherAgent(OpenRouterAgent):
    def __init__(self,
                 purpose: str,
                 api_key: str=None,
                 model: str = BASE_MODEL,
                 verbose: int = logging.INFO):
        super().__init__(purpose, 
                         api_key, 
                         model,
                         verbose)
    
    @generate_tool(Descriptions(function="Get current temperature for provided coordinates in celsius.",
                                properties={'latitude': 'location latitude',
                                            'longitude': 'location longitude'}))
    def get_weather(self, latitude, longitude):
        response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
        data = response.json()
        return data['current']['temperature_2m']

    def make_tools(self) -> List[Tool]:
        # Collecting tools from decorated methods
        return [getattr(self, method).tool for method in dir(self) if hasattr(getattr(self, method), 'tool')]

    
    def call_function(self):
        pass 

#%%
# initialize the weather agent
weather_agent=WeatherAgent(purpose='You are a weather expert',
                           verbose=logging.DEBUG)

#%%
weather_agent.execute("What's the weather like in Paris today?")
# %%
