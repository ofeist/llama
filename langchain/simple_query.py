from langchain_community.llms import Ollama

ollama = Ollama(
    base_url='http://localhost:11434',
    model="llama3"
)
print(ollama.invoke("why is the sky blue"))

