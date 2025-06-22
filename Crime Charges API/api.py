from fastapi import FastAPI, Request
from pydantic import BaseModel
from vllm import LLM, SamplingParams

# Load model
llm = LLM(model="/data/huggingface_models/models--Qwen--Qwen3-8B")  # Replace with your model path or name

# Define FastAPI app
app = FastAPI()

# Input format
class PromptRequest():
    story: str
    
# Endpoint
@app.post("/eval")
async def generate_text(request: PromptRequest):
    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=1024
    )
    outputs = llm.generate(request.story, sampling_params)
    return {"answer": outputs[0].outputs[0].text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=2000, reload=True)
