./llama-server \
  -m models/gemma-3-4b-it-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -c 8192

# Server running at http://localhost:8080
