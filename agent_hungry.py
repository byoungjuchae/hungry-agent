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
from langchain_community.vectorstores import Milvus
import pymilvus
from pydantic import Field, BaseModel
from typing import TypedDict
from dotenv import load_dotenv
from operator import itemgetter
import os
import requests
from prompt_file import PLANNER_PROMPT, PLAN_SEARCH_PROMPT, LANGUAGE_PROMPT,LANGUAGE_CHECK_PROMPT
from bs4 import BeautifulSoup as Soup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import psycopg2



load_dotenv()


# embedding = OpenAIEmbeddings(model='text-')

# milvus = Milvus(
#     embedding_function = OpenAIEmbeddings(model='')
# )

OPENAI_API_KEY = os.getenv("OPENAI_API")
KAKAO_API_KEY = os.getenv("KAKAO_MAP_API")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "hungry"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
llm = ChatOpenAI(model='gpt-4o-mini',openai_api_key=OPENAI_API_KEY)

history = RedisChatMessageHistory(
    url="redis://localhost:6379",
    ttl=3600,
    session_id="user_123"
)

conn = psycopg2.connect(
    host='postgres',
    dbname='airflow',
    user='airflow',
    password='airflow',
    port=5432
)

cur = conn.cursor()
create_table_sql = """CREATE TABLE IF NOT EXISTS users(
                    id TEXT,
                    place_name TEXT,
                    place_url TEXT,
                    rating TEXT,
                    total_reviewers INTEGER,
                    blogging INTEGER,
                    time_schedule TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                    );
"""
cur.execute(create_table_sql)
conn.commit()





class State(TypedDict):

    user_query : str 
    messages : list[str]
    plan : str
    restaurant : list[str]
    language: str
    keyword : str
    location : str
    classifier : str 
    response : str
    final_response : str

class Review(BaseModel):

    place_name : list[str] = Field(default='')
    place_url : list[str] = Field(default='')
    total_reviewers : list[str] = Field(default='')
    rating : list[str] = Field(default='')
    blogging : list[str] = Field(default='')
    time_schedule_opening : list[str] = Field(default='')
    time_schedule_closed : list[str] = Field(default='')


async def planner(state: State):

    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

    chain = {"user_query": itemgetter("user_query") |  RunnablePassthrough()} | prompt | llm | StrOutputParser()

    response = await chain.ainvoke({"user_query":state['user_query']})

    state['plan'] = response

    return state


@tool
async def place_search(keyword:str):
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
    
        for i in range(5):
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("no-sandbox")
            options.add_argument('window-size=1920x1080')
            options.add_argument("disable-gpu") 
            options.add_argument("lang=ko_KR")    
            options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36')
            driver = webdriver.Chrome(options=options)
            driver.get(address[i].get('place_url'))
            wait = WebDriverWait(driver, 15)
            btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "button.btn_fold2[aria-controls='foldDetail2']")))
            if btn.get_attribute("aria-expanded") != "true":
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                driver.execute_script("arguments[0].click();", btn)  
            panel = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#foldDetail2")))

            time.sleep(3)
     
            rating = driver.find_element(By.XPATH,'//*[@id="mainContent"]/div[1]/div[1]/div[2]/div[1]/a/span/span[2]').text
            total_reviewers = driver.find_element(By.XPATH,'//*[@id="mainContent"]/div[1]/div[1]/div[2]/div[2]/a/span[2]').text
            total_blogging = driver.find_element(By.XPATH,'//*[@id="mainContent"]/div[1]/div[1]/div[2]/div[3]/a/span[2]').text

            ###### 광고 제외시키는 module 만들기. 블로거 내돈내산 아닌거 제외해서 허수 줄이기. 

            infor['place_name'] = address[i].get("place_name")
            infor['place_url'] = address[i].get("place_url")
            infor['rating'] = rating
            infor['total_reviewers'] = total_reviewers
            infor['blogging'] =  total_blogging
            infor['time_schedule'] = panel.text

            now = time.localtime()
            cur.execute("""INSERT INTO users (id, place_name, place_url, rating, total_reviewers, blogging, time_schedule, created_at) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) """,
                        ('1213',infor['place_name'],infor['place_url'],infor['rating'],int(infor['total_reviewers']),int(infor['blogging']),infor['time_schedule'],time.strftime('%Y%m%d',now)))
            
            conn.commit()
            driver.quit()
            information.append(infor)
  
        return information

    else:
        return None


@tool(args_schema=Review)
async def context_review(place_name: list[str], place_url : list[str], total_reviewers:list[str], rating: list[str], blogging: list[str], time_schedule : list[str]):
    "if you want to know about the review of restaurant, use this tool. input document is the restaurant's information"
    
    prompt = ChatPromptTemplate.from_template(PLAN_SEARCH_PROMPT)
    chain = ({"place_name": itemgetter("place_name") | RunnablePassthrough(), "place_url":itemgetter("place_url") | RunnablePassthrough(),"rating": itemgetter("rating") | RunnablePassthrough(),"total_reviewers":itemgetter("total_reviewers") | RunnablePassthrough(), 
        "blogging": itemgetter("blogging") | RunnablePassthrough(), "time_schedule": itemgetter("time_schedule") | RunnablePassthrough()}
        | prompt | llm | StrOutputParser())
    response = await chain.ainvoke({"place_name": place_name[0], "place_url": place_url[0],
                "rating":rating[0],"total_reviewers":total_reviewers[0],"blogging":blogging[0],
                "time_schedule": time_schedule_opening[0]})

    return response

async def restaurant(state: State):


    restaurant_agent = create_react_agent(llm,tools=[place_search],
                                    prompt =("you are a assistant to run the plan."
                                            "if you want to search the place that user want, use the place_search tool. input is the keyword of the restaurant"
                                            "you have to translate the answer into language of user's query."))
    chunks = []

    async for chunk in restaurant_agent.astream({"messages":state['plan']}):
        chunks.append(chunk)
        print(chunk)

    state['restaurant'] = chunks[-1]

    return state

async def check_language(state:State):

    prompt = ChatPromptTemplate.from_template(LANGUAGE_CHECK_PROMPT)

    chain = {"user_query": itemgetter("user_query") | RunnablePassthrough()} | prompt | llm | StrOutputParser()

    response = await chain.ainvoke({"user_query": state.get('user_query')})

    state['language'] = response
    
    return state

async def language_converter(state:State):

    prompt = ChatPromptTemplate.from_template(LANGUAGE_PROMPT)


    chain = {"language": itemgetter("language") | RunnablePassthrough(), "input_sentences": itemgetter("input_sentences") | RunnablePassthrough() } | prompt | llm | StrOutputParser()

    response = await chain.ainvoke({"input_sentences":state.get('restaurant'),"language":state.get('language')})

    state['final_response'] = response

    return state


graph_builder = StateGraph(State)
graph_builder.add_node("planner",planner)
graph_builder.add_node("restaurant",restaurant)
graph_builder.add_node("check_language",check_language)
graph_builder.add_node("language",language_converter)
graph_builder.add_edge("planner","restaurant")
graph_builder.add_edge("restaurant","check_language")
graph_builder.add_edge("check_language","language")
graph_builder.set_entry_point("planner")
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer = checkpointer)



async def chatting(query:str):
    config = {"configurable": {"thread_id": "1","checkpoint_ns":"restaurant"}}
    asc = State(user_query=query)
    response = await graph.ainvoke(asc,config=config)
  
    res = response.get('final_response')
    print(res)
    return res

# def ranker(state: State):



# def responder(state: State):
    

