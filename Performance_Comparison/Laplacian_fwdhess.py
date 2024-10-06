"""
@author: tetean
@time: 2024/10/5 23:17
@info: 
"""

import jax
import jax.numpy as jnp
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

def dtanh(x):
    """计算双曲正切函数的导数"""
    return 1 - jnp.tanh(x) ** 2


def tanh_hessian(A, x, b):
    """
    计算给定输入的二阶导数 Hessian 张量

    参数：
    A -- 权重矩阵，形状 (n, m)
    x -- 输入向量，形状 (m,)
    b -- 偏置向量，形状 (n,)

    返回：
    H -- Hessian 张量，形状 (n, m, m)
    """
    # 计算 z = A @ x + b
    z = A @ x + b

    # 计算一阶导数 sech²(z) = 1 - tanh²(z)
    sech2_z = 1 - jnp.tanh(z) ** 2

    # 计算二阶导数 -2 * tanh(z) * sech²(z)
    tanh_prime = -2 * jnp.tanh(z) * sech2_z

    H = A[:, :, None] * A[:, None, :] * tanh_prime[:, None, None]

    return H


class FHess:
    """
    Hessian 类，用于存储向量及其雅可比和 Hessian
    """
    def __init__(self, x=None, jac=None, hess=None):
        self.x = x      # 向量
        self.jac = jac  # 雅可比矩阵
        self.hess = hess  # Hessian 张量



def fwd_hess(hess_last, J_uF, jac_last, H_uF, shape):
    """
    计算前向 Hessian 张量

    参数：
    hess_last -- 上一层的 Hessian
    J_uF -- 当前层的雅可比
    jac_last -- 上一层的雅可比
    H_uF -- 当前层的 Hessian
    shape -- 输出形状 (p, n)

    返回：
    hessian_F -- 当前层的 Hessian 张量
    """
    # 初始化 Hessian 张量，形状 (p, n, n)
    hessian_F = jnp.zeros((shape[0], shape[1], shape[1]))

    for i in range(shape[0]):
        # 计算第一项：∇x² u_k * ∇u F_i,k
        term1 = jnp.tensordot(J_uF[i], hess_last, axes=1)  # 结果形状 (n, n)
        # term1 = jnp.einsum('i,ijk->jk', J_uF[i], hess_last)

        # 计算第二项：∇x u * ∇u² F_i * ∇x u
        term2 = jnp.dot(jac_last.T, jnp.dot(H_uF[i], jac_last))  # 结果形状 (n, n)
        # term2 = jnp.einsum('ij,jk,kl->il', jac_last.T, H_uF[i], jac_last)

        # 将两部分相加得到 F_i 的 Hessian
        hessian_F = hessian_F.at[i].set(term1 + term2)

    return hessian_F

def MLP(x):
    """
    计算每一层的输出、雅可比和 Hessian

    参数：
    x -- 输入向量
    params -- 模型参数（权重和偏置）

    返回：
    info -- 每一层的输出信息（包含 x、雅可比和 Hessian）
    """
    global in_dim
    info = []

    for W, b in params:
        # 计算当前层的输出
        x1 = W @ x + b
        x2 = jnp.tanh(x1)

        if not len(info):
            # 第一层的雅可比和 Hessian
            jac = jnp.diag(dtanh(x1)) @ W
            hess = tanh_hessian(W, x, b)
        else:
            # 计算后续层的雅可比和 Hessian
            jac = jnp.diag(dtanh(x1)) @ W @ info[-1].jac
            H_uF = tanh_hessian(W, x, b)
            hess = fwd_hess(info[-1].hess, jnp.diag(dtanh(x1)) @ W, info[-1].jac, H_uF, [W.shape[0], in_dim])

        x = x2
        info.append(FHess(x2, jac, hess))

    return info


# 定义模型参数
conf = load_config()
layers = conf['layers']; in_dim = layers[0]
X = jax.random.uniform(jax.random.PRNGKey(0), shape=(layers[0],))
CNT = conf['CNT']
params = init_params(layers)


print('----------------------------前向 Hessian 结果---------------------------------')
# 计算前向 Hessian 的执行时间
start_time = time.time()
for _ in range(CNT):
    hess = MLP(X)[-1].hess
    Lap = jnp.trace(hess, axis1=-1, axis2=-2)
duration = time.time() - start_time
print('Laplacian: ', Lap)
print(f'前向 Hessian 计算 {CNT} 次，共用时：{duration}')



