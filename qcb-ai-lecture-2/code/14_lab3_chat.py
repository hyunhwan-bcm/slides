from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit": break
    messages.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="gpt-oss-20b", messages=messages,
    )
    
    assistant_msg = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_msg})
    print(f"Assistant: {assistant_msg}\n")
