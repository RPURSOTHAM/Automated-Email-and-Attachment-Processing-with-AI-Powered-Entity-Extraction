from ollama import Client

# Connect to Ollama's local server
client = Client(host='http://localhost:11434')

# Run a simple query on the LLaMA2 model
response = client.chat(model='llama2', messages=[{'role': 'user', 'content': 'Summarize the purpose of this email.'}])

print("Response from Local LLM (LLaMA2):\n", response['message']['content'])
