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
