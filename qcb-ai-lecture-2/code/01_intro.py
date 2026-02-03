# Before: Dependent on APIs
client = OpenAI(api_key="sk-proj-...")

# After: Independent Infrastructure
client = OpenAI(base_url="http://localhost:8080/v1")
