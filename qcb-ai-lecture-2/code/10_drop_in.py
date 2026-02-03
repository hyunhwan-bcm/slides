from openai import OpenAI

# Option 1: OpenAI Cloud
client_cloud = OpenAI(api_key="sk-proj-xxx")

# Option 2: Local llama.cpp server
client_local = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # llama.cpp ignores this
)

# Same interface, different backend!
