# Local AI, Agents, and the `gpt-oss-20b` Stack

### Building Production-Grade AI Infrastructure

---

## Welcome

### What You'll Build Today

```python
# Before: Dependent on APIs
client = OpenAI(api_key="sk-proj-...")

# After: Independent Infrastructure
client = OpenAI(base_url="http://localhost:8080/v1")
```

**You will:**
- Run 20B parameter models on your laptop
- Build tool-calling agents from scratch
- Deploy autonomous coding assistants
- Save thousands in API costs

---

## Part 1: Foundation
### Understanding the Stack

---

## The Model Landscape (2024-2026)

```
┌─────────────────────────────────────────┐
│ Size vs. Capability Trade-off          │
├─────────────────────────────────────────┤
│ 7B   → Fast, but hallucinates tools    │
│ 13B  → Better, but still unreliable    │
│ 20B  → ✓ Reliable reasoning            │
│ 70B+ → Best, but requires A100         │
└─────────────────────────────────────────┘
```

**`gpt-oss-20b` is the sweet spot:**
- Fits on consumer hardware
- Production-grade reasoning
- Apache 2.0 license

---

## Hardware Requirements

### Minimum Specs for This Workshop

```yaml
Option 1 (NVIDIA):
  - RTX 3090 (24GB VRAM)
  - RTX 4090 (24GB VRAM)
  - Inference: ~40 tokens/sec

Option 2 (Apple Silicon):
  - M2/M3 with 32GB+ RAM
  - Inference: ~25 tokens/sec
  
Option 3 (CPU Only):
  - 32GB RAM minimum
  - Inference: ~5 tokens/sec (slow but works)
```

**We'll optimize for speed later in Part 2.**

---

## Quantization Deep Dive

### How We Fit 20B Parameters in 14GB

```python
# Full Precision (FP16)
model_size = 20_000_000_000 * 2  # 2 bytes per param
print(f"{model_size / 1e9:.1f} GB")  # 40.0 GB

# 4-bit Quantized (Q4_K_M)
model_size = 20_000_000_000 * 0.5  # 0.5 bytes per param
print(f"{model_size / 1e9:.1f} GB")  # 10.0 GB (+ overhead = ~14GB)
```

**Quality loss:** <2% perplexity increase  
**Speed gain:** 3x faster on consumer GPUs

---

## Quantization Formats Explained

```
GGUF Quantization Types:
┌─────────┬─────────┬──────────┬─────────┐
│ Format  │ Size    │ Quality  │ Use Case│
├─────────┼─────────┼──────────┼─────────┤
│ Q2_K    │ 5.5GB   │ Poor     │ Testing │
│ Q4_K_M  │ 14GB    │ Excellent│ Default │
│ Q5_K_M  │ 17GB    │ Best     │ Max GPU │
│ Q8_0    │ 21GB    │ Perfect  │ Rare    │
└─────────┴─────────┴──────────┴─────────┘
```

**For this workshop: Use Q4_K_M**

---

## Installing llama.cpp

### Method 1: Pre-built Binaries (Fast)

```bash
# macOS (Apple Silicon)
brew install llama.cpp

# Linux/macOS (Manual)
wget https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-cli-linux-x64.zip
unzip llama-cli-linux-x64.zip
chmod +x llama-cli
```

### Method 2: Build from Source (For GPU)

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# With CUDA support
make LLAMA_CUDA=1

# With Metal (macOS)
make LLAMA_METAL=1
```

---

## Downloading the Model

```bash
# Using huggingface-cli (recommended)
pip install huggingface-hub

huggingface-cli download \
  TheBloke/gpt-oss-20B-GGUF \
  gpt-oss-20b.Q4_K_M.gguf \
  --local-dir ./models \
  --local-dir-use-symlinks False

# File will be at: ./models/gpt-oss-20b.Q4_K_M.gguf
```

**Expected size:** ~13.8 GB  
**Download time:** 5-30 minutes (depends on connection)

---

## 🛠️ Lab 1: First Inference

### Running the Model (CLI Mode)

```bash
./llama-cli \
  -m models/gpt-oss-20b.Q4_K_M.gguf \
  -p "Write a Python function to check if a number is prime" \
  -n 512 \
  -ngl 99 \
  -c 4096
```

**Flag breakdown:**
- `-m`: Model path
- `-p`: Prompt
- `-n`: Max tokens to generate
- `-ngl`: GPU layers (99 = all)
- `-c`: Context window

---

## Understanding the Output

```
llama_model_loader: loaded meta data with 20 key-value pairs
llm_load_tensors: loaded 291/291 tensors (13.8 GB)
llm_load_print_meta: n_ctx = 4096
llm_load_print_meta: n_gpu_layers = 99

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
    
llama_print_timings: eval time = 3245.67 ms / 128 tokens (25.36 ms/token, 39.43 t/s)
```

**Key metrics:**
- **39.43 t/s** = Your inference speed
- Lower is slower, higher is better

---

## Performance Tuning

### CPU vs GPU Offloading

```bash
# No GPU (Baseline)
./llama-cli -m model.gguf -p "test" -ngl 0
# Speed: ~3 t/s

# Partial GPU (Some layers)
./llama-cli -m model.gguf -p "test" -ngl 33
# Speed: ~15 t/s

# Full GPU (All layers)
./llama-cli -m model.gguf -p "test" -ngl 99
# Speed: ~40 t/s
```

**Rule:** Use `-ngl 99` unless you hit OOM errors.

---

## Server Mode: The Game Changer

### Starting llama-server

```bash
./llama-server \
  -m models/gpt-oss-20b.Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -c 8192

# Server running at http://localhost:8080
```

**Now your model is an API server!**

---

## Testing the Server

```bash
# Using curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "user", "content": "Say hello"}
    ]
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [{
    "message": {"role": "assistant", "content": "Hello! How can I help you today?"}
  }]
}
```

---

## Python Integration
### OpenAI-Compatible Client

---

## The Drop-In Replacement Pattern

```python
# install the official client
# pip install openai

from openai import OpenAI

# Option 1: OpenAI Cloud
client_cloud = OpenAI(api_key="sk-proj-xxx")

# Option 2: Local llama.cpp server
client_local = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # llama.cpp ignores this
)

# Same interface, different backend!
```

---

## 🛠️ Lab 2: Your First Local Client

### Create `test_local.py`

```python
from openai import OpenAI

# Connect to your llama-server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="local"
)

# Make a simple request
response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to reverse a string."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

**Run it:**
```bash
python test_local.py
```

---

## Streaming Responses

### Real-Time Token Generation

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

stream = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "Explain async/await in Python"}],
    stream=True  # <-- Enable streaming
)

print("Assistant: ", end="", flush=True)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

**Output:** Tokens appear in real-time, just like ChatGPT!

---

## Prompt Engineering Basics

### System Prompts Matter

```python
# Bad: Vague system prompt
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Fix this code: print(x)"}
]

# Good: Specific instructions
messages = [
    {"role": "system", "content": """You are a Python debugging expert.
When shown code with errors:
1. Identify the specific issue
2. Explain why it fails
3. Provide corrected code
4. Add comments explaining the fix"""},
    {"role": "user", "content": "Fix this code: print(x)"}
]
```

---

## Temperature and Sampling

```python
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

# For code: Use 0.2-0.4
# For creative writing: Use 0.7-1.0
```

---

## Context Window Management

```python
def trim_context(messages, max_tokens=6000):
    """Keep only recent messages to fit context window."""
    # Always keep system message
    system_msg = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    
    # Rough token estimate: 1 token ≈ 4 chars
    total_chars = sum(len(str(m)) for m in other_msgs)
    
    while total_chars > max_tokens * 4 and len(other_msgs) > 2:
        # Remove oldest user/assistant pair
        other_msgs = other_msgs[2:]
        total_chars = sum(len(str(m)) for m in other_msgs)
    
    return system_msg + other_msgs
```

---

## 🛠️ Lab 3: Chat Loop with Memory

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("Chat started. Type 'exit' to quit.\n")

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
```

---

## Tool Calling & Agents
### Building Autonomous Systems

---

## What is Tool Calling?

```
Traditional LLM:
  User: "What's the weather in NYC?"
  LLM: "I don't have access to real-time data..."

Tool-Calling LLM:
  User: "What's the weather in NYC?"
  LLM: → Calls weather_api("NYC")
  LLM: "It's 72°F and sunny in New York City."
```

**The LLM decides WHEN and HOW to use tools.**

---

## Tool Schema Structure

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g., 'New York'"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

---

## Implementing Python Functions

```python
def get_weather(city: str, units: str = "fahrenheit") -> dict:
    """Mock weather API call."""
    # In production: Call real weather API
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
```

---

## Tool-Calling Request

```python
from openai import OpenAI
import json

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
```

---

## Parsing Tool Calls

```python
message = response.choices[0].message

if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"Model wants to call: {function_name}")
        print(f"With arguments: {function_args}")
        
        # Example output:
        # Model wants to call: get_weather
        # With arguments: {'city': 'New York', 'units': 'fahrenheit'}
```

---

## Executing Tools and Returning Results

```python
# Map function names to actual Python functions
available_functions = {
    "get_weather": get_weather,
}

# Execute the function
function_to_call = available_functions[function_name]
function_response = function_to_call(**function_args)

# Add tool result to conversation
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(function_response)
})

# Get final response from model
final_response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=messages
)
```

---

## 🛠️ Lab 4: Complete Tool-Calling Agent

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

# Define tools
tools = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g., '2 + 2' or '10 * 5'"
                }
            },
            "required": ["expression"]
        }
    }
}]

def calculate(expression: str) -> dict:
    """Safely evaluate math expressions."""
    try:
        # Only allow safe math operations
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression"}
        
        result = eval(expression)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e)}

available_functions = {"calculate": calculate}
```

---

## Agent Loop Implementation

```python
def run_agent(user_query: str):
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": user_query}
    ]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-oss-20b",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # Check if model wants to call a tool
        if not message.tool_calls:
            # No more tools, return final answer
            return message.content
        
        # Execute each tool call
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            result = available_functions[func_name](**func_args)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

# Test it
print(run_agent("What is 1234 times 567?"))
```

---

## Advanced Agent Patterns
### Real-World Applications

---

## Multi-Tool Agent: File System Navigator

```python
import os
import subprocess

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List contents of a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a text file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to file"}
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"}
                },
                "required": ["command"]
            }
        }
    }
]
```

---

## Tool Implementations

```python
def list_directory(path: str) -> dict:
    """List files in directory."""
    try:
        items = os.listdir(path)
        return {"path": path, "items": items, "count": len(items)}
    except Exception as e:
        return {"error": str(e)}

def read_file(filepath: str) -> dict:
    """Read file contents."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        return {"filepath": filepath, "content": content, "size": len(content)}
    except Exception as e:
        return {"error": str(e)}

def run_command(command: str) -> dict:
    """Execute shell command (BE CAREFUL!)."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=30
        )
        return {
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        return {"error": str(e)}
```

---

## 🛠️ Lab 5: Autonomous Code Analyzer

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

available_functions = {
    "list_directory": list_directory,
    "read_file": read_file,
    "run_command": run_command
}

def code_analyzer_agent(task: str, max_iterations=10):
    """Agent that can analyze codebases."""
    messages = [
        {
            "role": "system",
            "content": """You are a code analysis agent. You can:
- list_directory: Browse folders
- read_file: Read source code
- run_command: Run tests or linters

When analyzing code:
1. First explore the directory structure
2. Read relevant files
3. Provide detailed analysis
4. Suggest improvements"""
        },
        {"role": "user", "content": task}
    ]
    
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-oss-20b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        if not message.tool_calls:
            return message.content
        
        print(f"\n=== Iteration {iteration + 1} ===")
        
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            print(f"Calling: {func_name}({func_args})")
            
            result = available_functions[func_name](**func_args)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
    
    return "Max iterations reached"

# Run the agent
result = code_analyzer_agent("Analyze the Python files in ./src and check for security issues")
print("\n=== Final Analysis ===")
print(result)
```

---

## Error Handling in Agents

```python
def safe_tool_executor(func_name: str, func_args: dict) -> dict:
    """Wrapper with error handling and logging."""
    try:
        if func_name not in available_functions:
            return {"error": f"Unknown function: {func_name}"}
        
        # Add timeout protection
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Function execution timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        result = available_functions[func_name](**func_args)
        
        signal.alarm(0)  # Cancel timeout
        
        # Log successful execution
        print(f"✓ {func_name} completed successfully")
        
        return result
        
    except TimeoutError:
        return {"error": "Execution timeout (30s)"}
    except Exception as e:
        return {"error": f"Exception: {type(e).__name__}: {str(e)}"}
```

---

## Production Patterns
### Deploying Local AI

---

## Configuration Management

```python
# config.py
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "local"
    model: str = "gpt-oss-20b"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60
    
    @classmethod
    def from_env(cls):
        """Load config from environment variables."""
        return cls(
            base_url=os.getenv("LLM_BASE_URL", cls.base_url),
            api_key=os.getenv("LLM_API_KEY", cls.api_key),
            model=os.getenv("LLM_MODEL", cls.model),
            temperature=float(os.getenv("LLM_TEMPERATURE", cls.temperature)),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", cls.max_tokens)),
        )

# Usage
config = LLMConfig.from_env()
client = OpenAI(base_url=config.base_url, api_key=config.api_key)
```

---

## Async/Await Pattern

```python
import asyncio
from openai import AsyncOpenAI

async def async_inference(client: AsyncOpenAI, prompt: str):
    """Non-blocking inference."""
    response = await client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

async def batch_inference(prompts: list[str]):
    """Process multiple prompts concurrently."""
    client = AsyncOpenAI(
        base_url="http://localhost:8080/v1",
        api_key="local"
    )
    
    tasks = [async_inference(client, p) for p in prompts]
    results = await asyncio.gather(*tasks)
    
    return results

# Run it
prompts = [
    "What is Python?",
    "What is Rust?",
    "What is Go?"
]
results = asyncio.run(batch_inference(prompts))
```

---

## Caching Layer

```python
import hashlib
import json
from functools import wraps

class SimpleCache:
    def __init__(self):
        self.cache = {}
    
    def get_key(self, messages, temperature):
        """Generate cache key from messages."""
        content = json.dumps(messages, sort_keys=True) + str(temperature)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, messages, temperature):
        key = self.get_key(messages, temperature)
        return self.cache.get(key)
    
    def set(self, messages, temperature, response):
        key = self.get_key(messages, temperature)
        self.cache[key] = response

cache = SimpleCache()

def cached_completion(client, messages, temperature=0.7):
    """Check cache before API call."""
    # Try cache first
    cached_response = cache.get(messages, temperature)
    if cached_response:
        print("✓ Cache hit!")
        return cached_response
    
    # Cache miss, call API
    print("✗ Cache miss, calling API...")
    response = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=messages,
        temperature=temperature
    )
    
    # Store in cache
    cache.set(messages, temperature, response)
    
    return response
```

---

## Logging and Monitoring

```python
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_llm_call(messages, response, duration):
    """Log every LLM interaction."""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
        "response": response.choices[0].message.content if response.choices else None,
        "model": response.model,
        "duration_ms": duration * 1000,
        "usage": {
            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
            "completion_tokens": getattr(response.usage, 'completion_tokens', 0)
        }
    }
    
    logger.info(f"LLM_CALL: {json.dumps(log_data)}")
    
    return log_data
```

---

## 🛠️ Lab 6: Production-Ready Agent

```python
from openai import OpenAI
import json
import logging
from typing import List, Dict, Callable
from dataclasses import dataclass
import time

@dataclass
class AgentConfig:
    model: str = "gpt-oss-20b"
    temperature: float = 0.7
    max_iterations: int = 10
    timeout: int = 60

class ProductionAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = OpenAI(
            base_url="http://localhost:8080/v1",
            api_key="local"
        )
        self.logger = logging.getLogger(__name__)
        self.tools = {}
    
    def register_tool(self, schema: dict, func: Callable):
        """Register a tool with its schema and implementation."""
        func_name = schema["function"]["name"]
        self.tools[func_name] = {
            "schema": schema,
            "func": func
        }
        self.logger.info(f"Registered tool: {func_name}")
    
    def execute_tool(self, name: str, args: dict) -> dict:
        """Execute a tool with error handling."""
        try:
            if name not in self.tools:
                return {"error": f"Tool '{name}' not found"}
            
            start_time = time.time()
            result = self.tools[name]["func"](**args)
            duration = time.time() - start_time
            
            self.logger.info(f"Tool '{name}' executed in {duration:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Tool '{name}' failed: {e}")
            return {"error": str(e)}
    
    def run(self, task: str, system_prompt: str = None) -> str:
        """Run the agent on a task."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": task})
        
        tool_schemas = [t["schema"] for t in self.tools.values()]
        
        for iteration in range(self.config.max_iterations):
            self.logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                temperature=self.config.temperature
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            if not message.tool_calls:
                return message.content
            
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                result = self.execute_tool(func_name, func_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        
        return "Maximum iterations reached"
```

---

## Using the Production Agent

```python
# Initialize
config = AgentConfig(temperature=0.3, max_iterations=5)
agent = ProductionAgent(config)

# Register tools
agent.register_tool(
    schema={
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search documentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    func=lambda query: {"results": f"Found docs about: {query}"}
)

# Run task
result = agent.run(
    task="Find documentation about error handling",
    system_prompt="You are a helpful documentation assistant."
)

print(result)
```

---

## Advanced Topics
### Scaling and Optimization

---

## Model Switching Strategy

```python
class MultiModelClient:
    """Use different models for different tasks."""
    
    def __init__(self):
        self.fast_client = OpenAI(
            base_url="http://localhost:8080/v1",  # 8B model
            api_key="local"
        )
        self.smart_client = OpenAI(
            base_url="http://localhost:8081/v1",  # 20B model
            api_key="local"
        )
    
    def get_completion(self, messages, task_complexity="medium"):
        """Route to appropriate model based on complexity."""
        if task_complexity == "simple":
            # Use fast 8B model for simple tasks
            return self.fast_client.chat.completions.create(
                model="llama-8b",
                messages=messages
            )
        else:
            # Use smart 20B model for complex tasks
            return self.smart_client.chat.completions.create(
                model="gpt-oss-20b",
                messages=messages
            )
    
    def auto_route(self, messages):
        """Automatically determine complexity."""
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        
        # Simple heuristic: Check for keywords
        complex_keywords = ["analyze", "refactor", "debug", "explain"]
        is_complex = any(kw in user_msg.lower() for kw in complex_keywords)
        
        complexity = "complex" if is_complex else "simple"
        return self.get_completion(messages, complexity)
```

---

## Prompt Templating System

```python
from string import Template

class PromptTemplate:
    """Reusable prompt templates."""
    
    TEMPLATES = {
        "code_review": Template("""You are an expert code reviewer.

Review the following $language code:

```$language
$code
```

Provide:
1. Issues found
2. Security concerns
3. Performance improvements
4. Best practice violations"""),
        
        "debug_helper": Template("""You are a debugging assistant.

Error message:
$error

Code context:
```$language
$code
```

Help the user:
1. Understand the error
2. Identify the root cause
3. Provide a fix
4. Explain how to prevent similar errors"""),
        
        "test_generator": Template("""Generate unit tests for this $language function:

```$language
$code
```

Requirements:
- Use $test_framework
- Cover edge cases
- Include docstrings
- Test both success and failure cases""")
    }
    
    @classmethod
    def render(cls, template_name: str, **kwargs) -> str:
        """Render a template with variables."""
        if template_name not in cls.TEMPLATES:
            raise ValueError(f"Template '{template_name}' not found")
        
        return cls.TEMPLATES[template_name].substitute(**kwargs)

# Usage
prompt = PromptTemplate.render(
    "code_review",
    language="python",
    code="def add(a, b):\n    return a + b"
)
```

---

## 🛠️ Lab 7: Code Review Agent

```python
from openai import OpenAI

class CodeReviewAgent:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:8080/v1",
            api_key="local"
        )
    
    def review_file(self, filepath: str, language: str = "python"):
        """Review a code file."""
        # Read the file
        with open(filepath, 'r') as f:
            code = f.read()
        
        # Generate review prompt
        prompt = PromptTemplate.render(
            "code_review",
            language=language,
            code=code
        )
        
        # Get review
        response = self.client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a senior software engineer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def review_diff(self, diff_text: str):
        """Review a git diff."""
        prompt = f"""Review this code change:

```diff
{diff_text}
```

Focus on:
1. Potential bugs introduced
2. Breaking changes
3. Performance implications
4. Security concerns"""
        
        response = self.client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return response.choices[0].message.content

# Usage
agent = CodeReviewAgent()
review = agent.review_file("my_script.py", language="python")
print(review)
```

---

## RAG (Retrieval-Augmented Generation)

### Simple Document Search

```python
import numpy as np
from openai import OpenAI

class SimpleRAG:
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
        self.documents = []
        self.embeddings = []
    
    def add_document(self, text: str):
        """Add a document to the knowledge base."""
        self.documents.append(text)
        # Note: For production, use proper embedding model
        # This is a simple placeholder
        embedding = self._simple_embedding(text)
        self.embeddings.append(embedding)
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple word-based embedding (for demo only)."""
        # In production: Use sentence-transformers or OpenAI embeddings
        words = text.lower().split()
        # Create a simple frequency vector
        vector = np.random.rand(384)  # Placeholder
        return vector
    
    def search(self, query: str, top_k: int = 3) -> list:
        """Find most relevant documents."""
        query_emb = self._simple_embedding(query)
        
        # Calculate similarity (cosine similarity)
        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
            )
            similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k documents
        return [self.documents[i] for i, _ in similarities[:top_k]]
    
    def answer_question(self, question: str) -> str:
        """Answer question using retrieved context."""
        # Retrieve relevant documents
        context_docs = self.search(question, top_k=2)
        context = "\n\n".join(context_docs)
        
        # Generate answer with context
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
```

---

## Using RAG for Documentation

```python
# Initialize RAG system
rag = SimpleRAG()

# Add documentation
rag.add_document("""
Python's asyncio module provides infrastructure for writing 
concurrent code using async/await syntax. It is used for 
network I/O and other high-latency operations.
""")

rag.add_document("""
The requests library is used for making HTTP requests in Python.
It abstracts the complexities of making requests behind a simple API.
""")

rag.add_document("""
FastAPI is a modern web framework for building APIs with Python.
It uses type hints and provides automatic API documentation.
""")

# Ask a question
answer = rag.answer_question("How do I make HTTP requests in Python?")
print(answer)
```

---

## Real-World Use Cases

---

## Use Case 1: Automated Code Documentation

```python
class DocGenerator:
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
    
    def generate_docstring(self, function_code: str) -> str:
        """Generate docstring for a function."""
        prompt = f"""Generate a detailed Google-style docstring for this Python function:

```python
{function_code}
```

Include:
- Brief description
- Args section with types
- Returns section with type
- Raises section if applicable
- Example usage

Return ONLY the docstring, properly formatted."""
        
        response = self.client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def document_file(self, filepath: str) -> str:
        """Add docstrings to all functions in a file."""
        with open(filepath, 'r') as f:
            code = f.read()
        
        # Extract functions (simple regex, use AST for production)
        import re
        functions = re.findall(r'(def \w+\([^)]*\):)', code)
        
        documented_code = code
        for func_sig in functions:
            if '"""' not in code[code.find(func_sig):code.find(func_sig)+200]:
                # Function lacks docstring
                docstring = self.generate_docstring(func_sig)
                # Insert docstring (simplified)
                print(f"Generated docstring for: {func_sig}")
                print(docstring)
                print()
        
        return documented_code

# Usage
doc_gen = DocGenerator()
doc_gen.document_file("my_module.py")
```

---

## Use Case 2: Intelligent Log Analyzer

```python
class LogAnalyzer:
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
    
    def analyze_errors(self, log_file: str) -> dict:
        """Analyze error logs and provide insights."""
        with open(log_file, 'r') as f:
            logs = f.readlines()
        
        # Extract error lines
        errors = [line for line in logs if 'ERROR' in line or 'Exception' in line]
        
        if not errors:
            return {"status": "no_errors", "message": "No errors found in logs"}
        
        # Limit to recent errors
        recent_errors = errors[-50:]  # Last 50 errors
        error_text = '\n'.join(recent_errors)
        
        prompt = f"""Analyze these error logs:

```
{error_text}
```

Provide:
1. Summary of error types
2. Most frequent errors
3. Potential root causes
4. Recommended fixes
5. Priority level (low/medium/high/critical)

Format as JSON."""
        
        response = self.client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return response.choices[0].message.content
    
    def suggest_monitoring(self, log_analysis: str) -> list:
        """Suggest monitoring alerts based on log analysis."""
        prompt = f"""Based on this log analysis:

{log_analysis}

Suggest 5 monitoring alerts or dashboards that should be created.
Format as a Python list of dictionaries with keys: 'name', 'condition', 'severity'."""
        
        response = self.client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
```

---

## Use Case 3: API Client Generator

```python
class APIClientGenerator:
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
    
    def generate_from_openapi(self, openapi_spec: dict) -> str:
        """Generate Python client from OpenAPI spec."""
        import json
        spec_json = json.dumps(openapi_spec, indent=2)
        
        prompt = f"""Generate a complete Python client class for this OpenAPI specification:

```json
{spec_json}
```

Requirements:
- Use requests library
- Include all endpoints as methods
- Add type hints
- Include docstrings
- Add error handling
- Support authentication

Return complete, runnable Python code."""
        
        response = self.client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=3000
        )
        
        return response.choices[0].message.content

# Usage
generator = APIClientGenerator()

openapi_spec = {
    "openapi": "3.0.0",
    "info": {"title": "User API", "version": "1.0.0"},
    "paths": {
        "/users": {
            "get": {
                "summary": "List users",
                "responses": {"200": {"description": "Success"}}
            }
        }
    }
}

client_code = generator.generate_from_openapi(openapi_spec)
print(client_code)
```

---

## Debugging & Optimization

---

## Debugging Tool-Calling Issues

```python
class DebugAgent:
    """Agent with verbose logging for debugging."""
    
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
        self.conversation_log = []
    
    def run_with_debug(self, task: str, tools: list):
        """Run agent with detailed logging."""
        messages = [{"role": "user", "content": task}]
        
        print("=" * 60)
        print(f"TASK: {task}")
        print("=" * 60)
        
        for iteration in range(5):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            response = self.client.chat.completions.create(
                model="gpt-oss-20b",
                messages=messages,
                tools=tools,
                temperature=0.7
            )
            
            message = response.choices[0].message
            
            # Log the response
            print(f"Model response type: {type(message)}")
            print(f"Has tool_calls: {bool(message.tool_calls)}")
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    print(f"\nTool Call:")
                    print(f"  Name: {tool_call.function.name}")
                    print(f"  Arguments: {tool_call.function.arguments}")
                    
                    # Try to parse arguments
                    try:
                        args = json.loads(tool_call.function.arguments)
                        print(f"  Parsed args: {args}")
                    except json.JSONDecodeError as e:
                        print(f"  ERROR parsing arguments: {e}")
                        print(f"  Raw arguments: {tool_call.function.arguments}")
            else:
                print(f"\nFinal response: {message.content}")
                return message.content
            
            messages.append(message)
        
        return "Max iterations reached"
```

---

## Performance Profiling

```python
import time
from functools import wraps

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
    
    def profile_llm_call(func):
        """Decorator to profile LLM calls."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            result = func(self, *args, **kwargs)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            metrics = {
                "function": func.__name__,
                "duration": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "timestamp": time.time()
            }
            
            self.metrics.append(metrics)
            
            print(f"⏱️  {func.__name__} took {metrics['duration']:.2f}s")
            print(f"💾  Memory delta: {metrics['memory_delta']:.2f}MB")
            
            return result
        return wrapper
    
    @staticmethod
    def _get_memory_usage():
        """Get current memory usage in MB."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def get_stats(self):
        """Get performance statistics."""
        if not self.metrics:
            return "No metrics collected"
        
        total_time = sum(m["duration"] for m in self.metrics)
        avg_time = total_time / len(self.metrics)
        
        return {
            "total_calls": len(self.metrics),
            "total_time": total_time,
            "average_time": avg_time,
            "slowest_call": max(self.metrics, key=lambda x: x["duration"])
        }
```

---

## Token Usage Optimization

```python
class TokenOptimizer:
    """Optimize prompts to reduce token usage."""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    @staticmethod
    def compress_code(code: str) -> str:
        """Remove unnecessary whitespace from code."""
        lines = code.split('\n')
        # Remove empty lines and excessive whitespace
        compressed = '\n'.join(
            line.strip() for line in lines 
            if line.strip()
        )
        return compressed
    
    @staticmethod
    def truncate_context(messages: list, max_tokens: int = 4000) -> list:
        """Truncate old messages to fit token budget."""
        # Always keep system message
        system_msgs = [m for m in messages if m["role"] == "system"]
        other_msgs = [m for m in messages if m["role"] != "system"]
        
        # Estimate tokens
        total_tokens = sum(
            TokenOptimizer.estimate_tokens(str(m)) 
            for m in other_msgs
        )
        
        # Remove oldest messages if over budget
        while total_tokens > max_tokens and len(other_msgs) > 2:
            removed = other_msgs.pop(0)
            total_tokens -= TokenOptimizer.estimate_tokens(str(removed))
        
        return system_msgs + other_msgs
    
    @staticmethod
    def summarize_long_output(client, long_text: str, max_length: int = 500) -> str:
        """Summarize long tool outputs."""
        if len(long_text) <= max_length:
            return long_text
        
        response = client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[{
                "role": "user",
                "content": f"Summarize this in under {max_length} characters:\n\n{long_text}"
            }],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content
```

---

## Deployment Strategies

---

## Docker Deployment

```dockerfile
# Dockerfile for llama-server
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp
WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /app/llama.cpp
RUN make LLAMA_CUDA=1

# Download model
WORKDIR /app/models
RUN wget https://huggingface.co/TheBloke/gpt-oss-20B-GGUF/resolve/main/gpt-oss-20b.Q4_K_M.gguf

# Expose port
EXPOSE 8080

# Start server
CMD ["/app/llama.cpp/llama-server", \
     "-m", "/app/models/gpt-oss-20b.Q4_K_M.gguf", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "-ngl", "99", \
     "-c", "8192"]
```

---

## docker-compose.yml

```yaml
version: '3.8'

services:
  llama-server:
    build: .
    ports:
      - "8080:8080"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  agent-api:
    build: ./agent
    ports:
      - "5000:5000"
    environment:
      - LLM_BASE_URL=http://llama-server:8080/v1
    depends_on:
      - llama-server
    restart: unless-stopped
```

---

## Kubernetes Deployment

```yaml
# llama-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llama-server
  template:
    metadata:
      labels:
        app: llama-server
    spec:
      containers:
      - name: llama-server
        image: my-registry/llama-server:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: models
          mountPath: /app/models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: llama-server-service
spec:
  selector:
    app: llama-server
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

---

## Conclusion

---

## What You've Learned

### Technical Skills
- ✅ Running 20B parameter models on consumer hardware
- ✅ Building OpenAI-compatible inference servers
- ✅ Implementing tool-calling agents from scratch
- ✅ Production patterns for error handling and logging
- ✅ Deployment strategies for local AI

### Architecture Patterns
- ✅ Drop-in replacement pattern for cloud APIs
- ✅ ReAct agent loop implementation
- ✅ Multi-tool coordination
- ✅ Caching and optimization strategies

---

## Cost Comparison

```
Traditional Cloud AI (OpenAI GPT-4):
  - $0.03 per 1K input tokens
  - $0.06 per 1K output tokens
  - 1M tokens = $30-60

Local gpt-oss-20b:
  - Hardware: RTX 3090 (~$1000)
  - Electricity: ~$0.50/day
  - Unlimited tokens
  - Break-even: ~50M tokens

For heavy users: Local pays off in weeks!
```

---

## Performance Benchmark

```
Model Comparison (Coding Tasks):

GPT-4:
  - Tokens/sec: 40-60
  - Cost: $0.03/1K tokens
  - Privacy: ❌ Cloud

gpt-oss-20b (Local):
  - Tokens/sec: 30-45
  - Cost: $0 (after hardware)
  - Privacy: ✅ Local

Claude 3:
  - Tokens/sec: 50-70
  - Cost: $0.015/1K tokens
  - Privacy: ❌ Cloud
```

---

## Production Checklist

### Before Deploying

```markdown
□ Performance Testing
  □ Measure tokens/sec on target hardware
  □ Test with realistic prompt sizes
  □ Profile memory usage

□ Reliability
  □ Implement error handling
  □ Add retry logic
  □ Set up logging and monitoring

□ Security
  □ Sanitize user inputs
  □ Validate tool outputs
  □ Implement rate limiting

□ Documentation
  □ API documentation
  □ Tool schemas documented
  □ Runbook for operations

□ Monitoring
  □ Request latency metrics
  □ Error rate tracking
  □ Resource utilization alerts
```

---

## Common Pitfalls

### 1. Context Window Overflow
```python
# Bad: No context management
messages.append(new_message)  # Grows forever

# Good: Trim old messages
messages = TokenOptimizer.truncate_context(messages, max_tokens=6000)
messages.append(new_message)
```

### 2. Tool Hallucination
```python
# Bad: Trust all tool calls
result = eval(tool_call.function.arguments)  # Dangerous!

# Good: Validate tool calls
if tool_call.function.name not in allowed_tools:
    return {"error": "Unknown tool"}
```

### 3. No Error Recovery
```python
# Bad: Crashes on API error
response = client.chat.completions.create(...)

# Good: Handle errors gracefully
try:
    response = client.chat.completions.create(...)
except Exception as e:
    logger.error(f"LLM call failed: {e}")
    return fallback_response
```

---

## Resources

### Essential Links
- **llama.cpp GitHub:** https://github.com/ggerganov/llama.cpp
- **Model Download:** https://huggingface.co/TheBloke/gpt-oss-20B-GGUF
- **OpenAI Python SDK:** https://github.com/openai/openai-python

### Community
- **llama.cpp Discord:** Active community for optimization tips
- **r/LocalLLaMA:** Reddit community for local AI enthusiasts
- **HuggingFace Forums:** Model-specific discussions

### Further Learning
- **Prompt Engineering Guide:** https://www.promptingguide.ai/
- **LangChain Docs:** For advanced agent patterns
- **OpenAI Cookbook:** API best practices

---

## Next Steps

### Immediate (This Week)
1. Download and run `gpt-oss-20b` locally
2. Build a simple tool-calling agent
3. Deploy llama-server with Docker

### Short Term (This Month)
1. Integrate into your development workflow
2. Build custom tools for your use case
3. Optimize prompts for your domain

### Long Term (This Quarter)
1. Deploy to production with monitoring
2. Fine-tune models for specialized tasks
3. Build a team library of reusable agents

---

## Final Project Ideas

### 1. Code Review Bot
- Reads git diffs
- Identifies potential issues
- Suggests improvements
- Posts comments on PRs

### 2. Documentation Generator
- Scans codebase
- Generates API docs
- Creates usage examples
- Maintains knowledge base

### 3. DevOps Assistant
- Monitors logs
- Diagnoses issues
- Suggests fixes
- Automates common tasks

### 4. Data Analysis Agent
- Loads datasets
- Performs exploratory analysis
- Generates visualizations
- Writes reports

---

## Thank You!

### You now have the skills to:
- 🚀 Run production-grade AI locally
- 🛠️ Build autonomous coding agents
- 💰 Save thousands in API costs
- 🔒 Maintain data privacy
- ⚡ Deploy scalable AI infrastructure

### Questions?
Let's discuss your specific use cases and challenges.

### Stay in Touch
- GitHub: [workshop-repo-link]
- Discord: [community-invite]
- Email: [instructor-email]

---

## Bonus: Quick Reference

### Essential Commands

```bash
# Start llama-server
./llama-server -m model.gguf --port 8080 -ngl 99

# Test server
curl http://localhost:8080/health

# Monitor GPU
nvidia-smi -l 1

# Check Docker logs
docker logs -f llama-server
```

### Essential Python Snippets

```python
# Basic client
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="x")

# Simple completion
response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "Hello"}]
)

# With tools
response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)
```
