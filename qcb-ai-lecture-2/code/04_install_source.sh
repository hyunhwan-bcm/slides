git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CUDA=1  # CUDA support
make LLAMA_METAL=1  # Metal (macOS)
