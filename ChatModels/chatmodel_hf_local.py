# from langchain_huggingface import HuggingFacePipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from dotenv import load_dotenv

# load_dotenv()

# # Load model and tokenizer
# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

# # Create pipeline
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     temperature=0.1,
#     top_p=0.95
# )

# # Create LangChain wrapper
# llm = HuggingFacePipeline(pipeline=pipe)

# # Use the model
# result = llm.invoke("What is the capital of Pakistan?")
# print(result)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

# Verify torch first
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load model (with error handling)
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        device=0 if device == "cuda" else -1
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Test query
    result = llm.invoke("What is the capital of Pakistan?")
    print(result)
    
except Exception as e:
    print(f"Error: {str(e)}")
    if "CUDA" in str(e):
        print("\nTry running with device='cpu' or check your CUDA installation")