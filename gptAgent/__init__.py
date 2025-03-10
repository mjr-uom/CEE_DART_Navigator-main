import random
import time
from openai import AzureOpenAI

deployment_name = 'lunar-gpt-4o'
class OpenAIAgent:
   def __init__(
       self,
       agent_name="OpenAI Agent",
       system_role = "You are an expert in molecular genomics and cancer research. You always focus on specific proteins and genes of interest and the facts provided to you. Your reasoning is based on the context provided."
   ):
       self.client = AzureOpenAI(
           api_key="d8d1b0b5a5b94cc7a5fb12a49283fa",
           api_version="2024-10-21",
           azure_endpoint="https://lunarchatgpt.openai.azure.com/"
       )
       self.agent_name = agent_name
       self.system_role = system_role
       self.deployment_name = 'lunar-gpt-4o'
   def query(self, message):
       response = self.client.chat.completions.create(
           model=deployment_name,
           messages=[ {"role": "system", "content": self.system_role}, {"role": "user", "content": message}]
       )
       for word in response.choices[0].message.content.split():
           yield word + " "
           time.sleep(0.05)
       #return response.choices[0].message.content
