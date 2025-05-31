from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Use a chat model endpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature=0.1,
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Explain the potentail impact of AI on the future of work.")
print(result.content)



# Use the model directly for text generation (not as a chat model)
# from langchain_huggingface import HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     max_new_tokens=512,
#     temperature=0.1
# )

# # Use directly for text generation
# result = llm.invoke("What is the capital of Pakistan?")
# print(result)