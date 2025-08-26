from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.memory import RedisChatMessageHistory
import pymilvus
from pydantic import Field, BaseModel
from typing import TypedDict
from dotenv import load_dotenv
from operator import itemgetter
import os
import requests
from prompt_file import PLANNER_PROMPT, CONTEXT_REVIEW_PROMPT
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup



load_dotenv()



OPENAI_API_KEY = os.getenv("OPENAI_API")
KAKAO_API_KEY = os.getenv("KAKAO_MAP_API")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "hungry"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
llm = ChatOpenAI(model='gpt-4o',openai_api_key=OPENAI_API_KEY)
history = RedisChatMessageHistory(
    url="redis://localhost:6379",
    ttl=3600,
    session_id="user_123"
)





class State(TypedDict):

    user_query : str 
    messages : list[str]
    plan : str
    restaurant : str
    keyword : str
    location : str
    classifier : str 
    response : str



def planner(state: State):

    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

    chain = {"user_query": itemgetter("user_query") |  RunnablePassthrough()} | prompt | llm | StrOutputParser()

    response = chain.invoke({"user_query":state['user_query']})

    state['plan'] = response

    return state





@tool
def place_search(keyword: str):
    "if you have to find the restaurants, use this tool. the input keyword is related to the food."
    headers = {"Authorization":f"KakaoAK {KAKAO_API_KEY}"}
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json'

    response = requests.get(url,headers=headers,params={"query":keyword})
    if response.status_code == 200:

        names = []
        website = []
        address = response.json().get('documents')
        for i in range(len(address)):

            names.append(address[i].get('place_name'))
            website.append(address[i].get('place_url'))
        
     

        return 'this restaurant is good and review is good'

    else:
        return None


@tool
def context_review(documents: str):
    "if you want to know about the review of restaurant, use this tool. input document is the restaurant's information"

    prompt = ChatPromptTemplate.from_template(CONTEXT_REVIEW_PROMPT)
    chain = {"documents": itemgetter("documents") | RunnablePassthrough()} | prompt | llm | StrOutputParser()
    
    response = chain.invoke({"documents":documents})

    return response

def restaurant(state: State):


    restaurant_agent = create_react_agent(llm,tools=[place_search,context_review],
                                    prompt =("you are a assistant to run the plan."
                                            "if you want to take a information of the restaurants, use the context_review tool. input is only keyword of the restaurant's information"
                                            "if you want to search the place that user want, use the place_search tool. input is the keyword of the restaurant "
                                            "There is no required thing to using tool"))
    chunks = []
    for chunk in restaurant_agent.stream({"messages":state['plan']}):
        chunks.append(chunk)
        print(chunk)

    state['restaurant'] = chunks[-1]

    return state




graph_builder = StateGraph(State)
graph_builder.add_node("planner",planner)
graph_builder.add_node("restaurant",restaurant)
graph_builder.add_edge("planner","restaurant")
graph_builder.set_entry_point("planner")
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer = checkpointer)






# def ranker(state: State):



# def responder(state: State):
    



if __name__ == '__main__':

    config = {"configurable": {"thread_id": "1","checkpoint_ns":"restaurant"}}
    asc = State(user_query="I want to eat meat.")
    response = graph.invoke(asc,config=config)
    print(response)