# Temperature: 0.0 (Deterministic)
response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    temperature=0.0  # Always same answer
)

# Temperature: 1.0 (Creative)
response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "Write a creative story"}],
    temperature=1.0  # More variety
)
