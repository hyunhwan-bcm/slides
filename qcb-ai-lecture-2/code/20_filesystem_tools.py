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
