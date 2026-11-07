from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Prompt tuned for NL → Shell translation
NL2SH_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert Linux shell assistant.
Using the documentation below, translate the user's natural language request into the correct shell command.
Only respond with the shell command and a one-line explanation. Do not add markdown.

Documentation:
{context}

User Request: {question}

Shell Command:"""
)

def load_chain(model_name: str = "llama3"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        "vectorstore/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}      # top-4 relevant chunks
    )
    llm = ChatOllama(model=model_name, temperature=0)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": NL2SH_PROMPT}
    )
    return chain