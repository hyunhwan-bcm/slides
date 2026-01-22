# Using huggingface-cli (recommended)
pip install huggingface-hub

huggingface-cli download \
  ggml-org/gpt-oss-20b-GGUF \
  gpt-oss-20b-mxfp4.gguf \
  --local-dir ./models \
  --local-dir-use-symlinks False

# File will be at: ./models/gpt-oss-20b.Q4_K_M.gguf
