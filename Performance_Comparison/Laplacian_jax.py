'''
@author: tetean
@time: 2024/10/04 20:01 PM
@info: 对比 lapjax 求 Laplacian 与 FwdHess 求 Laplacian，以及直接求 hessian 再求 Laplacian 的运行时间
'''

import jax
import jax.numpy as jnp
from util.Log import Log
import time
from util.Conf import load_config, write_res


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


def MLP(x, params):
    for W, b in params:
        x = jnp.tanh(W @ x + b)
    return x


# Log = Log().logger
conf = load_config(skip=True)


for exp in conf.values():
    layers = exp['layers']
    # X = jax.random.uniform(jax.random.PRNGKey(0), shape=(layers[0],))
    CNT = exp['CNT']
    params = init_params(layers)

    print('----------------------------Jax 通过 Hessian 求 Laplacian结果---------------------------------')

    # 使用 jax 计算 Hessian 并计算执行时间
    start_time = time.time()
    for _ in range(CNT):
        X = jax.random.uniform(jax.random.PRNGKey(0), shape=(layers[0],))
        hess = jax.hessian(MLP)(X, params)
        Lap = jnp.trace(hess, axis1=-1, axis2=-2)
    duration = time.time() - start_time
    # print('Laplacian: ', Lap)
    # print("Hessian 张量:\n", hess)
    print(f'普通 Hessian 计算 {CNT} 次，共用时：{duration}')
    exp['running time'] = {'jax': duration}
write_res(conf)






