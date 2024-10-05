"""
@author: tetean
@time: 2024/10/5 23:16
@info: 
"""
import lapjax as jax
from lapjax import LapTuple
import lapjax.numpy as jnp
import time
from util.Conf import load_config


def init_params(layers):
    keys = jax.random.split(jax.random.PRNGKey(0), len(layers) - 1)
    params = []
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        lb, ub = -(1 / jnp.sqrt(n_in)), (1 / jnp.sqrt(n_in))
        W = lb + (ub - lb) * jax.random.uniform(key, shape=(n_out, n_in))
        b = jax.random.uniform(key, shape=(n_out,))
        params.append([W, b])

    params[-1][1] = jnp.zeros_like(params[-1][1])  # 最后一层需要置 0
    return params


def MLP_lapjax(x, params):
    for W, b in params:
        x = jnp.dot(W, x)
        x = jnp.tanh(x + b)
    return x

# Log = Log().logger
conf = load_config()
layers = conf['layers']
X = jax.random.uniform(jax.random.PRNGKey(0), shape=(layers[0],))
CNT = conf['CNT']
params = init_params(layers)

print('----------------------------通过 Lapjax 求 Laplacian结果---------------------------------')


start_time = time.time()
for _ in range(CNT):
    input_laptuple = LapTuple(X, is_input=True)
    output_laptuple = MLP_lapjax(input_laptuple, params)
    Lap = output_laptuple.lap
duration = time.time() - start_time
print('Laplacian: ', Lap)
print(f'Lapjax 计算 {CNT} 次，共用时：{duration}')