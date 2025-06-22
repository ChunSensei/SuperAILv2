from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize FastAPI app
app = FastAPI()

# DEFINE MODELS ##
# model_name = "Qwen/Qwen3-0.6B"
model_name = "Qwen/Qwen3-8B"
cache_dir = "/home/siamai/lessons/huggingface_models/models--Qwen--Qwen3-8B"  # Replace with your desired path

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_dir
)

## LLM ANSWER
def llmanswer(query):
  messages = [
      {"role": "system", "content": "You are a helpful assistance. Please help answer or suggestion to the users requests"},
      {"role": "user", "content": query}
  ]

  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
  )

  model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

  # conduct text completion
  generated_ids = model.generate(
      **model_inputs,
      max_new_tokens=32768
  )

  output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

  # parsing thinking content
  try:
      # rindex finding 151668 (</think>)
      index = len(output_ids) - output_ids[::-1].index(151668)
  except ValueError:
      index = 0

  thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
  content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

  print("thinking content:", thinking_content)
  print("content:", content)

  return content

def answerYES(query):
    return f"YES! {query} YES!"

class UserInput(BaseModel):
    input_text: str

@app.post("/")
async def interact(user_input: UserInput):

    query = user_input.input_text

    response = answerYES(query)

    response = llmanswer(query)

    return {"response": response}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main_LLMHost:app",  # Use the import string for your app
        host="0.0.0.0",
        port=4000,
        reload=True  # Enable reloading (no workers)
    )

