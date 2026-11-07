from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest():
    # 1. Load man pages from /data/man_pages/
    loader = DirectoryLoader(
        "data/man_pages/",
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # 3. Embed using a local HuggingFace model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Build & save FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore/faiss_index")
    print("FAISS index saved!")

if __name__ == "__main__":
    ingest()