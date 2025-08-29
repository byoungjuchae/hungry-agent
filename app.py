from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agent_hungry import chatting



app = FastAPI()

origins = [
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          
    allow_credentials=True,         
    allow_methods=["*"],            
    allow_headers=["*"],            
)
@app.post('/chat')
async def chat(query:str):


    response = await chatting(query=query)
    
    return response
