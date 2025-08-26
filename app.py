from fastapi import FastAPI
# from transformers import AutoTokenizer, pipeline
from pydantic import BaseModel
# from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

Model_ID= "microsoft/phi-2"
# tokenizer = AutoTokenizer.from_pretrained(Model_ID)
# model = OVModelForCausalLM.from_pretrained(Model_ID)
# generator = pipeline("text-generation", model = model , tokenizer = tokenizer)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


app = FastAPI(title = "Phi2- API" , version="2.0")

class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 100

@app.post("/generate")
def generate_text(request: PromptRequest):
    result = generator(
        request.prompt,
        max_new_tokens=request.max_length,   # use new tokens instead of max_length
        truncation=True,                     # fix truncation warning
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id  # fix pad warning
    )
    return {"response": result[0]["generated_text"]}
