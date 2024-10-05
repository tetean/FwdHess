'''
@author: tetean
@time: 2024/10/04 11:44 PM
@info: 计算给定参数和输入向量的 Hessian 张量
'''

import jax
import jax.numpy as jnp
import time
from jax import vmap, jit
from jax import tree_util

# 设置计算次数
CNT = int(1e2)

def dtanh(x):
    """计算双曲正切函数的导数"""
    return 1 - jnp.tanh(x) ** 2

def F_jax(x, params):
    """
    向量值函数 F 的定义：
    F(x) = tanh(C * tanh(Ax + b) + d)

    参数：
    x -- 输入向量，形状 (n,)
    params -- 包含层的权重和偏置的元组列表

    返回：
    输出向量，形状 (p,)
    """
    for W, b in params:
        x = jnp.tanh(jnp.dot(W, x) + b)
    return x

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

    # 获取输入的维度
    n = len(b)  # 输出维度
    m = len(x)  # 输入维度

    H = A[:, :, None] * A[:, None, :] * tanh_prime[:, None, None]

    return H

@tree_util.register_pytree_node_class
class FHess:
    """
    Hessian 类，用于存储向量及其雅可比和 Hessian
    """
    def __init__(self, x=None, jac=None, hess=None):
        self.x = x      # 向量，形状可能为 (m,)
        self.jac = jac  # 雅可比矩阵，形状可能为 (m, n)
        self.hess = hess  # Hessian 张量，形状可能为 (m, m)

    def tree_flatten(self):
        """
        将类实例拆分为子元素和辅助数据。
        返回一个元组（子元素的元组, 辅助数据）
        """
        children = (self.x, self.jac, self.hess)
        aux_data = None  # 如果有需要，可以包含辅助数据
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        从子元素和辅助数据中重建类实例。
        """
        x, jac, hess = children
        return cls(x, jac, hess)

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

def f(x):
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

# 定义维度
n = 3  # 输入维度
in_dim = n
m = 2  # 第一层输出维度
p = 2  # 第二层输出维度
q = 3  # 第三层输出维度

# 初始化随机参数
key = jax.random.PRNGKey(0)
A = jax.random.normal(key, (m, n))
C = jax.random.normal(key, (p, m))
b = jax.random.normal(key, (m,))
d = jax.random.normal(key, (p,))

E = jax.random.normal(key, (q, p))
g = jax.random.normal(key, (q,))

# 定义输入向量
x = jnp.array([0.1, 0.2, 0.3])

# 定义模型参数
params = [(A, b), (C, d), (C, d), (E, g)]

# 计算 Hessian 张量
hessian_F = f(x)

print('----------------------------前向 Hessian 结果---------------------------------')
print("Hessian 张量形状:", hessian_F[-1].hess.shape)
print("Hessian 张量:\n", hessian_F[-1].hess)

# 计算前向 Hessian 的执行时间
start_time = time.time()
hessian_F = f(x)
duration = time.time() - start_time
print(f'前向 Hessian 计算 {CNT} 次，共用时：{duration}')

print('----------------------------jax 的 Hessian 结果---------------------------------')

# 使用 jax 计算 Hessian 并计算执行时间
start_time = time.time()
hess = jax.hessian(F_jax)(x, params)
duration = time.time() - start_time
print("Hessian 张量:\n", hess)
print(f'普通 Hessian 计算 {CNT} 次，共用时：{duration}')



vmap_f = jax.jit(vmap(f))


x = jnp.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

start_time = time.time()
outputs = vmap_f(x)
duration = time.time() - start_time
print(f'并行前向 Hessian 计算 {CNT} 次，共用时：{duration}')
print(outputs[-1].hess)
