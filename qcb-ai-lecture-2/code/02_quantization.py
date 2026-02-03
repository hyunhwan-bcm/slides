# Full Precision (FP16)
model_size = 20_000_000_000 * 2  # 2 bytes per param
print(f"{model_size / 1e9:.1f} GB")  # 40.0 GB

# 4-bit Quantized (Q4_K_M)
model_size = 20_000_000_000 * 0.5  # 0.5 bytes per param
print(f"{model_size / 1e9:.1f} GB")  # 10.0 GB (+ overhead = ~14GB)
