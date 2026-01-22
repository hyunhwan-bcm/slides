#!/usr/bin/env python3
"""
Fix corrupted slides.tex by restoring proper \inputminted commands
"""

def fix_slides():
    with open("slides.tex.original", "r") as f:
        content = f.read()
    
    # Map of replacements
    replacements = [
        # Intro code
        (
            r"""\begin{minted}{python}
# Before: Dependent on APIs
client = OpenAI(api_key="sk-proj-...")

# After: Independent Infrastructure
client = OpenAI(base_url="http://localhost:8080/v1")
\end{minted}""",
            r"\inputminted{python}{code/01_intro.py}"
        ),
        # Quantization
        (
            r"""\begin{minted}{python}
# Full Precision (FP16)
model_size = 20_000_000_000 * 2  # 2 bytes per param
print(f"{model_size / 1e9:.1f} GB")  # 40.0 GB

# 4-bit Quantized (Q4_K_M)
model_size = 20_000_000_000 * 0.5  # 0.5 bytes per param
print(f"{model_size / 1e9:.1f} GB")  # 10.0 GB (+ overhead = ~14GB)
\end{minted}""",
            r"\inputminted{python}{code/02_quantization.py}"
        ),
        # Brew install
        (
            r"""\begin{minted}{bash}
# macOS (Apple Silicon)
brew install llama.cpp

# Linux/macOS (Manual)
wget https://github.com/ggerganov/llama.cpp/releases/\
  latest/download/llama-cli-linux-x64.zip
unzip llama-cli-linux-x64.zip
chmod +x llama-cli
\end{minted}""",
            r"\inputminted{bash}{code/03_install_brew.sh}"
        ),
        # Source build
        (
            r"""\begin{minted}{bash}
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CUDA=1  # CUDA support
make LLAMA_METAL=1  # Metal (macOS)
\end{minted}""",
            r"\inputminted{bash}{code/04_install_source.sh}"
        ),
        # Download model
        (
            r"""\begin{minted}{bash}
# Using huggingface-cli (recommended)
pip install huggingface-hub

huggingface-cli download \
  TheBloke/gpt-oss-20B-GGUF \
  gpt-oss-20b.Q4_K_M.gguf \
  --local-dir ./models \
  --local-dir-use-symlinks False

# File will be at: ./models/gpt-oss-20b.Q4_K_M.gguf
\end{minted}""",
            r"\inputminted{bash}{code/05_download_model.sh}"
        ),
        # Lab 1 inference
        (
            r"""\begin{minted}{bash}
./llama-cli \
  -m models/gpt-oss-20b.Q4_K_M.gguf \
  -p "Write a Python function to check if a number is prime" \
  -n 512 \
  -ngl 99 \
  -c 4096
\end{minted}""",
            r"\inputminted{bash}{code/06_lab1_inference.sh}"
        ),
        # Start server
        (
            r"""\begin{minted}{bash}
./llama-server \
  -m models/gpt-oss-20b.Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -c 8192

# Server running at http://localhost:8080
\end{minted}""",
            r"\inputminted{bash}{code/07_start_server.sh}"
        ),
        # Test server
        (
            r"""\begin{minted}{bash}
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "user", "content": "Say hello"}
    ]
  }'
\end{minted}""",
            r"\inputminted{bash}{code/08_test_server.sh}"
        ),
        # Test response
        (
            r"""\begin{minted}{text}
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [{
    "message": {"role": "assistant", "content": "Hello!"}
  }]
}
\end{minted}""",
            r"\inputminted{text}{code/09_test_response.txt}"
        ),
        # Drop-in replacement
        (
            r"""\begin{minted}{python}
from openai import OpenAI

# Option 1: OpenAI Cloud
client_cloud = OpenAI(api_key="sk-proj-xxx")

# Option 2: Local llama.cpp server
client_local = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # llama.cpp ignores this
)

# Same interface, different backend!
\end{minted}""",
            r"\inputminted{python}{code/10_drop_in.py}"
        ),
        # Lab 2 client
        (
            r"""\begin{minted}{python}
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="local"
)

response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a function to reverse a string."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
\end{minted}""",
            r"\inputminted{python}{code/11_lab2_client.py}"
        ),
        # Streaming
        (
            r"""\begin{minted}{python}
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

stream = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "Explain async/await"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
\end{minted}""",
            r"\inputminted{python}{code/12_streaming.py}"
        ),
        # Temperature
        (
            r"""\begin{minted}{python}
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
\end{minted}""",
            r"\inputminted{python}{code/13_temperature.py}"
        ),
        # Lab 3 chat loop
        (
            r"""\begin{minted}{python}
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    messages.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=messages,
        temperature=0.7
    )
    
    assistant_msg = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_msg})
    print(f"Assistant: {assistant_msg}\n")
\end{minted}""",
            r"\inputminted{python}{code/14_lab3_chat.py}"
        ),
        # Tool schema
        (
            r"""\begin{minted}{python}
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]
\end{minted}""",
            r"\inputminted{python}{code/15_tool_schema.py}"
        ),
        # Weather function
        (
            r"""\begin{minted}{python}
def get_weather(city: str, units: str = "fahrenheit") -> dict:
    """Mock weather API call."""
    weather_data = {
        "New York": {"temp": 72, "condition": "Sunny"},
        "London": {"temp": 15, "condition": "Rainy"},
        "Tokyo": {"temp": 28, "condition": "Cloudy"}
    }
    
    data = weather_data.get(city, {"temp": 20, "condition": "Unknown"})
    
    if units == "celsius":
        data["temp"] = int((data["temp"] - 32) * 5/9)
    
    return {
        "city": city,
        "temperature": data["temp"],
        "condition": data["condition"],
        "units": units
    }
\end{minted}""",
            r"\inputminted{python}{code/16_weather_function.py}"
        ),
        # Tool request
        (
            r"""\begin{minted}{python}
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[
        {"role": "user", "content": "What's the weather in New York?"}
    ],
    tools=tools,
    tool_choice="auto"  # Let model decide when to use tools
)

print(response.choices[0].message)
\end{minted}""",
            r"\inputminted{python}{code/17_tool_request.py}"
        ),
        # Lab 4 tools
        (
            r"""\begin{minted}{python}
tools = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    }
}]

def calculate(expression: str) -> dict:
    try:
        return {"result": eval(expression), "expression": expression}
    except Exception as e:
        return {"error": str(e)}
\end{minted}""",
            r"\inputminted{python}{code/18_lab4_tools.py}"
        ),
        # Agent loop
        (
            r"""\begin{minted}{python}
def run_agent(user_query: str):
    messages = [
        {"role": "system", "content": "You are a math assistant."},
        {"role": "user", "content": user_query}
    ]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-oss-20b", messages=messages,
            tools=tools, tool_choice="auto"
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        if not message.tool_calls:
            return message.content
        
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            result = available_functions[func_name](**func_args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id,
                           "content": json.dumps(result)})
\end{minted}""",
            r"\inputminted{python}{code/19_agent_loop.py}"
        ),
        # Filesystem tools
        (
            r"""\begin{minted}{python}
def list_directory(path: str) -> dict:
    try:
        items = os.listdir(path)
        return {"path": path, "items": items, "count": len(items)}
    except Exception as e:
        return {"error": str(e)}

def read_file(filepath: str) -> dict:
    try:
        with open(filepath, 'r') as f:
            return {"filepath": filepath, "content": f.read()}
    except Exception as e:
        return {"error": str(e)}

def run_command(command: str) -> dict:
    try:
        result = subprocess.run(command, shell=True, 
                              capture_output=True, text=True)
        return {"stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return {"error": str(e)}
\end{minted}""",
            r"\inputminted{python}{code/20_filesystem_tools.py}"
        ),
        # Lab 5 analyzer
        (
            r"""\begin{minted}{python}
def code_analyzer_agent(task: str, max_iterations=10):
    messages = [{
        "role": "system",
        "content": """You are a code analysis agent. You can:
- list_directory: Browse folders
- read_file: Read source code
- run_command: Run tests or linters"""
    }, {"role": "user", "content": task}]
    
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-oss-20b", messages=messages,
            tools=tools, tool_choice="auto"
        )
        
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content
        
        # Execute tools and continue...
\end{minted}""",
            r"\inputminted{python}{code/21_lab5_analyzer.py}"
        ),
        # Config
        (
            r"""\begin{minted}{python}
from dataclasses import dataclass
import os

@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "local"
    model: str = "gpt-oss-20b"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    @classmethod
    def from_env(cls):
        return cls(base_url=os.getenv("LLM_BASE_URL", cls.base_url),
                   model=os.getenv("LLM_MODEL", cls.model))
        )

config = LLMConfig.from_env()
client = OpenAI(base_url=config.base_url, api_key=config.api_key)
\end{minted}""",
            r"\inputminted{python}{code/22_config.py}"
        ),
        # Lab 6 production
        (
            r"""\begin{minted}{python}
class ProductionAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = OpenAI(base_url="http://localhost:8080/v1")
        self.tools = {}
    
    def register_tool(self, schema: dict, func: Callable):
        func_name = schema["function"]["name"]
        self.tools[func_name] = {"schema": schema, "func": func}
    
    def execute_tool(self, name: str, args: dict) -> dict:
        try:
            return self.tools[name]["func"](**args)
        except Exception as e:
            return {"error": str(e)}
    
    def run(self, task: str) -> str:
        # Agent loop with error handling
        # ... implementation
\end{minted}""",
            r"\inputminted{python}{code/23_lab6_production.py}"
        ),
        # Dockerfile
        (
            r"""\begin{minted}{text}
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git build-essential

WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /app/llama.cpp
RUN make LLAMA_CUDA=1

EXPOSE 8080

CMD ["/app/llama.cpp/llama-server", \
     "-m", "/app/models/gpt-oss-20b.Q4_K_M.gguf", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "-ngl", "99"]
\end{minted}""",
            r"\inputminted{text}{code/24_dockerfile.txt}"
        ),
    ]
    
    # Apply replacements
    for old, new in replacements:
        content = content.replace(old, new)
    
    with open("slides.tex", "w") as f:
        f.write(content)
    
    print("Fixed slides.tex")

if __name__ == "__main__":
    fix_slides()
