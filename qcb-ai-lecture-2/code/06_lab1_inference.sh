./llama-cli \
  -m models/gemma-3-4b-it-Q4_K_M.gguf \
  -p "Write a Python function to check if a number is prime" \
  -n 512 \
  -ngl 99 \
  -c 4096
