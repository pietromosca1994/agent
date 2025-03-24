# Agent 
### Introduction
A versatile agent framework

### Installation
1. Clone this repository  
``` bash 
git clone https://github.com/pietromosca1994/agent
```

2. Install the package  
```bash 
pip install .
```

3. Define a specialized agent  
```python 
from agent.agent import OpenRouterAgent
from agent.utils import generate_tool
from agent.models import Descriptions, Tool
from agent.chatbot import BASE_MODEL

_=dotenv.load_dotenv(override=True)
print(f'Openrouter API key: {os.environ["OPENROUTER_API_KEY"]}')

class WeatherAgent(OpenRouterAgent):
    def __init__(self,
                 api_key: str=None,
                 model: str = BASE_MODEL,
                 verbose: int = logging.INFO):
        purpose="You are a highly knowledgeable and reliable weather expert. " \
        "Your role is to provide accurate, up-to-date, and detailed weather forecasts, insights, and explanations. " \
        "Use clear and concise language while ensuring accessibility for all users. Offer safety advice during extreme conditions and explain meteorological concepts when needed. " \
        "Your responses should be professional, informative, and engaging."
        
        super().__init__(purpose, 
                         api_key, 
                         model,
                         verbose)
    
    # tools definition
    @generate_tool(Descriptions(function="Get current temperature for provided coordinates in celsius.",
                                properties={'latitude': 'location latitude',
                                            'longitude': 'location longitude'}))
    def get_weather(self, latitude, longitude):
        response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
        data = response.json()
        return data['current']['temperature_2m']

    def make_tools(self) -> List[Tool]:
        tools=super().make_tools()

        # Additional modifications to tools
        # ~~~
        
        return tools
```

4. Use the specialized agent  
```python 
weather_agent=WeatherAgent(verbose=logging.DEBUG)
weather_agent
weather_agent.execute("What's the weather like in Paris today?")
```  

### References 
[Openrouter API Reference](https://openrouter.ai/docs/api-reference/overview)  
[OpenAI API Reference](https://platform.openai.com/docs/overview)
