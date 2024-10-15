"""
@author: tetean
@time: 2024/10/15 20:31
@info: 
"""

import jax
import jax.numpy as jnp
from util.Log import Log
import time
from util.Conf import load_config, write_res
import numpy as np


def init_params(layers):
    """
    初始化网络参数

    参数:
    layers: 一个列表,包含每层的神经元数量

    返回:
    初始化后的网络参数列表
    """
    keys = jax.random.split(jax.random.PRNGKey(0), len(layers) - 1)
    params = []
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        lb, ub = -(1 / jnp.sqrt(n_in)), (1 / jnp.sqrt(n_in))
        W = lb + (ub - lb) * jax.random.uniform(key, shape=(n_out, n_in))
        B = jax.random.uniform(key, shape=(n_out,))
        params.append({'W': W, 'B': B})
    return params


@jax.jit
def condense_B(x):
    # 假设x的形状是(m, n, n, n, n)
    # 取出x[:, j, j, k, k]，并将结果存储到y
    y = x[:, jnp.arange(x.shape[1]), jnp.arange(x.shape[1]), :, :]
    y = y[:, :, jnp.arange(x.shape[1]), jnp.arange(x.shape[1])]
    return y

@jax.jit
def MLP(params, x):
    for layer in params[:-1]:
        W = layer['W']
        b = layer['B']
        x = jnp.tanh(W @ x + b)
    W = params[-1]['W']
    b = params[-1]['B']
    return W @ x + b


conf = load_config(skip=True)

for exp in conf.values():
    layers = exp['layers']
    CNT = exp['CNT']
    params = init_params(layers)

    # 使用 jax 计算 bihar 并计算执行时间
    start_time = time.time()
    X = jax.random.uniform(jax.random.PRNGKey(0), shape=(layers[0],))
    for _ in range(CNT):
        bihar = jax.hessian(jax.hessian(MLP, argnums=1), argnums=1)(params, X)
        bihar = condense_B(bihar)
    duration = time.time() - start_time
    print(f'jax 计算 {CNT} 次，共用时：{duration}')
    exp['running time'] = {'jax': duration}
write_res(conf)






