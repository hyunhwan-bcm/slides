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
