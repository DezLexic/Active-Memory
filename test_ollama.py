import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Say hello and confirm you are running locally."}],
)

print(response.message.content)
