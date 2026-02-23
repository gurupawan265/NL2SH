import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load your NL2SH dataset 
df = pd.read_csv('C:\\Users\\LENOVO\\OneDrive\\Desktop\\NL2SH\\nl2sh_project\\nl2sh_train.csv')

# Format the rows so the AI sees the relationship between instructions and commands
docs = []
for _, row in df.iterrows():
    # Example mapping: "grep error from image355.sh" -> "grep INFO summary264.txt" 
    content = f"Instruction: {row['instruction']}\nCommand: {row['command']}"
    docs.append(Document(page_content=content))

# Use a lightweight embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create and save the local index
vector_db = FAISS.from_documents(docs, embeddings)
vector_db.save_local("bash_index")
print("Vector database created in VS Code!")