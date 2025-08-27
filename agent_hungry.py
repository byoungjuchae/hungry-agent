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
from prompt_file import PLANNER_PROMPT, PLAN_SEARCH_PROMPT
from bs4 import BeautifulSoup as Soup
from selenium import webdriver
from selenium.webdriver.common.by import By
import time


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
    restaurant : list[str]
    keyword : str
    location : str
    classifier : str 
    response : str


class Review(BaseModel):

    place_name : list[str] = Field(default='')
    place_url : list[str] = Field(default='')
    total_reviewers : list[str] = Field(default='')
    rating : list[str] = Field(default='')
    blogging : list[str] = Field(default='')
    time_schedule_opening : list[str] = Field(default='')
    time_schedule_closed : list[str] = Field(default='')


def planner(state: State):

    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

    chain = {"user_query": itemgetter("user_query") |  RunnablePassthrough()} | prompt | llm | StrOutputParser()

    response = chain.invoke({"user_query":state['user_query']})

    state['plan'] = response

    return state


@tool
def place_search(keyword:str):
    "if you have to find the restaurants, use this tool. the input keyword is related to the food. keyword must be korean."
    headers = {"Authorization":f"KakaoAK {KAKAO_API_KEY}"}
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json'

    response = requests.get(url,headers=headers,params={f"query":{keyword}})
    

    if response.status_code == 200:

        names = []
        website = []
        documents = []
        information = []
        infor = {}
        address = response.json().get('documents')
    
        for i in range(1):
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("no-sandbox")
            options.add_argument('window-size=1920x1080')
            options.add_argument("disable-gpu")   # 가속 사용 x
            options.add_argument("lang=ko_KR")    
            options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36')
            driver = webdriver.Chrome(options=options)
            driver.get(address[i].get('place_url'))

            time.sleep(3)
            
            rating = driver.find_element(By.XPATH,'//*[@id="mainContent"]/div[1]/div[1]/div[2]/div[1]/a/span/span[2]').text
            total_reviewers = driver.find_element(By.XPATH,'//*[@id="mainContent"]/div[1]/div[1]/div[2]/div[2]/a/span[2]').text
            total_blogging = driver.find_element(By.XPATH,'//*[@id="mainContent"]/div[1]/div[1]/div[2]/div[3]/a/span[2]').text

            time_schedule_opening = driver.find_element(By.XPATH,'//*[@id="foldDetail2"]/div/div/span[2]').text
            time_schedule_closed = driver.find_element(By.XPATH,'//*[@id="foldDetail2"]/div/div/div[1]/span').text
            

            infor['place_name'] = address[i].get("place_name")
            infor['place_url'] = address[i].get("place_url")
            infor['rating'] = rating
            infor['total_reviewers'] = total_reviewers
            infor['blogging'] =  total_blogging
            infor['time_schedule_opening'] = time_schedule_opening
            infor['time_schedule_closed'] = time_schedule_closed
            
            driver.quit()
            information.append(infor)
  
        return information

    else:
        return None


@tool(args_schema=Review)
def context_review(place_name: list[str], place_url : list[str], total_reviewers:list[str], rating: list[str], blogging: list[str], time_schedule_opening : list[str], time_schedule_closed : list[str]):
    "if you want to know about the review of restaurant, use this tool. input document is the restaurant's information"
    
    prompt = ChatPromptTemplate.from_template(PLAN_SEARCH_PROMPT)
    chain = ({"place_name": itemgetter("place_name") | RunnablePassthrough(), "place_url":itemgetter("place_url") | RunnablePassthrough(),"rating": itemgetter("rating") | RunnablePassthrough(),"total_reviewers":itemgetter("total_reviewers") | RunnablePassthrough(), 
        "blogging": itemgetter("blogging") | RunnablePassthrough(), "opening_time": itemgetter("opening_time") | RunnablePassthrough(), "closed_time" : itemgetter("closed_time") | RunnablePassthrough()}
        | prompt | llm | StrOutputParser())
    response = chain.invoke({"place_name": place_name[0], "place_url": place_url[0],
                "rating":rating[0],"total_reviewers":total_reviewers[0],"blogging":blogging[0],
                "opening_time": time_schedule_opening[0],
                "closed_time":time_schedule_closed[0]})

    return response

def restaurant(state: State):


    restaurant_agent = create_react_agent(llm,tools=[place_search,context_review],
                                    prompt =("you are a assistant to run the plan."
                                            "if you want to take a information of the restaurants, use the context_review tool. input is only keyword of the restaurant's information"
                                            "if you want to search the place that user want, use the place_search tool. input is the keyword of the restaurant "))
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
    




   
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1","checkpoint_ns":"restaurant"}}
    asc = State(user_query="I want to eat meat.")
    response = graph.invoke(asc,config=config)
    print(response)