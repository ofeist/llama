# document_query.py
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain.chains import RetrievalQA
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser


ollama = Ollama (
    base_url='http://localhost:11434',
    model="llama3"
)
# print(ollama.invoke("why is the sky blue"))


loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
data = loader.load()

# print(data)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
all_splits = text_splitter.split_documents(data)

oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

question="Who is Neleus and who is in Neleus' family?"
docs = vectorstore.similarity_search(question)
len(docs)


qachain=GuardrailsOutputParser.from_chain_type(ollama, retriever=vectorstore.as_retriever())
qachain.invoke({"query": question})
