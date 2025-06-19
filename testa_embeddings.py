from langchain_community.embeddings import OllamaEmbeddings

# Use o nome do modelo de embeddings que você quer testar (ex: 'llama3')
embeddings = OllamaEmbeddings(model='mxbai-embed-large')
result = embeddings.embed_query("teste de embeddings ollama")
print(result)
