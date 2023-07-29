##Importing all LLM libraries from langchian
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import os

#importing data analysis libraries 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##Setting up the OpenAI Key
os.environ['OPENAI_API_KEY'] = "XXXX"


##Function for reading csv into a dataframe
def load_data(file):
    df = pd.read_csv(file)
    return df


class DataAgent():

    def __init__(self, df):
        self.df = df

        self.llm = OpenAI(temperature=0.0, verbose=True)
        
        self.prefix = """
                        You are a data analyst who uses Pandas, Matplotlib, Seaborn to perform data analysis task,
                        asked by the user. When You have the final answer, just give the Final Answer.
                        Only answer to the user query, do not make assumptions and answer garbage.
                        """

        self.agent = create_pandas_dataframe_agent(llm=self.llm, df=self.df, prefix=self.prefix,  verbose=True,max_iterations=5)
    
    def generate_response(self,user_input):
        main_input = f"""
                Action : {user_input}
                Final Answer : 

"""

        return self.agent.run(main_input)
