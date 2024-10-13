from jax import numpy as jnp


def judge(MAT_A, MAT_B, tolerance=1e-5):
    difference = MAT_A - MAT_B
    max_diff = jnp.max(jnp.abs(difference))
    if max_diff < tolerance:
        print("手动计算的与测试答案一致。")
    else:
        print("手动计算的与测试答案不一致。")
    print(f"最大差异: {max_diff}")