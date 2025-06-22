## BASIC FAST API ##
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

# Define the expected request body
class Message(BaseModel):
    message: str

# Define the POST endpoint
@app.post("/")
async def chat_endpoint(msg: Message):
    user_input = msg.message
    # Process the input and return a response
    #res = model(msg)
    return {"answer": f"HI, You said: {user_input} Hi"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=2000, reload=True)