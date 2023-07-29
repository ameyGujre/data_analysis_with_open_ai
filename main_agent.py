##Importing all LLM libraries from langchian
from langchain.llms import AzureOpenAI, OpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
import os
import openai

#importing data analysis libraries 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##Setting up the OpenAI Key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_DEPLOYMENT_ENDPOINT = os.environ["OPENAI_API_BASE"]
OPENAI_DEPLOYMENT_NAME = os.environ["OPENAI_DEPLOYMENT_NAME"]
OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]
OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.environ["OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
OPENAI_EMBEDDING_MODEL_NAME = os.environ["OPENAI_EMBEDDING_MODEL_NAME"]
OPENAI_DEPLOYMENT_VERSION = os.environ["OPENAI_API_VERSION"]

# init Azure OpenAI
openai.api_type = "azure"
openai.api_version = OPENAI_DEPLOYMENT_VERSION
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY


##Function for reading csv into a dataframe
def load_data(file):
    df = pd.read_csv(file)
    return df


class DataAgent():

    def __init__(self, df):
        self.df = df

        self.llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                      model_name=OPENAI_MODEL_NAME,
                      openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                      openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                      openai_api_key=OPENAI_API_KEY,temperature=0.0, verbose=True)
        self.prefix = """
                        You are a data analyst who uses Pandas, Matplotlib, Seaborn to perform data analysis task,
                        asked by the user. When You have the final answer, just give the Final Answer.
                        Only answer to the user query, do not make assumptions and answer garbage.
                        

                        """
        self.suffix = """
                        To display the plot/graph or visualizations, Save the images in local temporary files, and 
                        wrap the image path around <img> tag of html
                        for example: <img src="temp/plot1.png">
                        """

        self.agent = create_pandas_dataframe_agent(llm=self.llm, df=self.df, prefix=self.prefix,  verbose=True,max_iterations=5)
    
    def generate_response(self,user_input):
        main_input = f"""
                Action : {user_input}
                Final Answer : 
                dataframe is given in df

"""

        return self.agent.run(main_input)
