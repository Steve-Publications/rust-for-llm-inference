# NVIDIA Transformer Engine FP8 configuration
from transformer_engine.common.recipe import Format, DelayedScaling
fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,  # E4M3 forward, E5M2 backward
    amax_history_len=16,
    amax_compute_algo="max"
)
