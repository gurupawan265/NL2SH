from rag_chain import load_chain

chain = load_chain(model_name="llama3")  # or mistral, codellama, etc.

print("NL2Sh ready! Type your request (ctrl+c to quit)\n")
while True:
    query = input("You: ")
    if not query.strip():
        continue
    result = chain.invoke({"query": query})
    print(f"\nShell: {result['result']}\n")