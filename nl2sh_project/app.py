from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Connect to Ollama Qwen 0.5B
llm = OllamaLLM(model="qwen:0.5b", temperature=0)

# 2. Load the index we just created
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.load_local("bash_index", embeddings, allow_dangerous_deserialization=True)

def translate_to_sh(user_query):
    # Retrieve the 2 most similar examples from your CSV 
    results = vector_db.similarity_search(user_query, k=2)
    context = "\n\n".join([res.page_content for res in results])

    # Augment the prompt for the small Qwen model
    prompt = f"""[INST] You are a Linux Command Generator. 
Use these examples to understand the syntax:
{context}

Task: {user_query}
Return ONLY the command. No explanations. No markdown code blocks. [/INST]"""

    return llm.invoke(prompt)

# Test Loop
query = input("What do you want to do? ")
print(f"Generated Shell Command: {translate_to_sh(query)}")