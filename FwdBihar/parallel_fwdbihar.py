'''
@Autior: tetean
@Time: 2024/10/17 5:35 PM
@Info:
'''
import jax
import jax.numpy as jnp
from jax import tree_util
import numpy as np
from util.Jug import judge
import time
from util.Conf import load_config, write_res

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


@tree_util.register_pytree_node_class
class FHess:
    """
    Hessian 类，用于存储向量及其雅可比和 Hessian
    """
    def __init__(self, x=None, jac=None, hess=None, trd=None, bihar=None):
        self.x = x
        self.jac = jac
        self.hess = hess
        self.trd = trd
        self.bihar = bihar

    def tree_flatten(self):
        """
        将类实例拆分为子元素和辅助数据。
        返回一个元组（子元素的元组, 辅助数据）
        """
        children = (self.x, self.jac, self.hess, self.trd, self.bihar)
        aux_data = None  # 辅助数据
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        从子元素和辅助数据中重建类实例。
        """
        x, jac, hess, trd, bihar = children
        return cls(x, jac, hess, trd, bihar)

@jax.jit
def dtanh(x):
    """计算双曲正切函数的导数"""
    return 1 - jnp.tanh(x) ** 2

# def condense_B(x):
#     # x的形状是(m, n, n, n, n)
#     y = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             for k in range(x.shape[1]):
#                 y[i][j][k] = x[i][j][j][k][k]
#     return jnp.array(y)


@jax.jit
def condense_B(x):
    # 假设x的形状是(m, n, n, n, n)
    # 取出x[:, j, j, k, k]，并将结果存储到y
    y = x[:, jnp.arange(x.shape[1]), jnp.arange(x.shape[1]), :, :]
    y = y[:, :, jnp.arange(x.shape[1]), jnp.arange(x.shape[1])]
    return y

@jax.jit
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

def final_hess(J_uF, hess_last, shape):
    hessian_F = jnp.zeros((shape[0], shape[1], shape[1]))

    for i in range(shape[0]):
        # 计算第一项：∇x² u_k * ∇u F_i,k
        tmp = jnp.tensordot(J_uF[i], hess_last, axes=1)  # 结果形状 (n, n)

        hessian_F = hessian_F.at[i].set(tmp)
    return hessian_F

@jax.jit
def fwd_trd(JF_u, Ju_x, HF_u, Hu_x, TF_u, Tu_x):
    term1 = jnp.einsum('hj,j123->h123', JF_u, Tu_x)
    term2 = jnp.einsum('hjk,j12,k3->h123', HF_u, Hu_x, Ju_x) + jnp.einsum('hjk,j13,k2->h123', HF_u, Hu_x, Ju_x) \
            + jnp.einsum('hjk,j23,k1->h123', HF_u, Hu_x, Ju_x)
    term3 = jnp.einsum('hjkl,j1,k2,l3->h123', TF_u, Ju_x, Ju_x, Ju_x, )

    TF_x = term1 + term2 + term3
    return TF_x

@jax.jit
def fwd_bihar(JF_u, Ju_x, HF_u, Hu_x, TF_u, Tu_x, BF_u, Bu_x_cond):
    # def condense_T(x):
    #     # x的形状是(m, n, n, n)
    #     y = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    #     for i in range(x.shape[0]):
    #         for j in range(x.shape[1]):
    #             for k in range(x.shape[1]):
    #                 y[i][j][k] = x[i][j][j][k]
    #     return jnp.array(y)
    #
    # def condense_H(x):
    #     # x的形状是(m, n, n)
    #     y = np.zeros((x.shape[0], x.shape[1]))
    #     for i in range(x.shape[0]):
    #         for j in range(x.shape[1]):
    #             y[i][j] = x[i][j][j]
    #     return jnp.array(y)
    @jax.jit
    def condense_T(x):
        # x的形状是(m, n, n, n)
        # 使用切片操作直接提取需要的元素
        y = x[:, jnp.arange(x.shape[1]), jnp.arange(x.shape[1]), :]
        return y
    @jax.jit
    def condense_H(x):
        # x的形状是(m, n, n)
        # 使用切片操作直接提取需要的元素
        y = x[:, jnp.arange(x.shape[1]), jnp.arange(x.shape[1])]
        return y

    Tu_x_cond = condense_T(Tu_x)
    Hu_x_cond = condense_H(Hu_x)
    term1 = jnp.einsum('hj,j13->h13', JF_u, Bu_x_cond)
    term2 = 2 * (jnp.einsum('hjk,j13,k3->h13', HF_u, Tu_x_cond, Ju_x) + jnp.einsum('hjk,j31,k1->h13', HF_u, Tu_x_cond,
                                                                                   Ju_x))
    term3 = jnp.einsum('hjk,j1,k3->h13', HF_u, Hu_x_cond, Hu_x_cond) + 2 * jnp.einsum('hjk,j13,k13->h13', HF_u, Hu_x,
                                                                                      Hu_x)
    term4 = jnp.einsum('hjkl,j1,k3,l3->h13', TF_u, Hu_x_cond, Ju_x, Ju_x) \
            + 4 * jnp.einsum('hjkl,j13,k1,l3->h13', TF_u, Hu_x, Ju_x, Ju_x) \
            + jnp.einsum('hjkl,j3,k1,l1->h13', TF_u, Hu_x_cond, Ju_x, Ju_x)
    term5 = jnp.einsum('hjklm,j1,k1,l3,m3->h13', BF_u, Ju_x, Ju_x, Ju_x, Ju_x)

    BF_x = term1 + term2 + term3 + term4 + term5
    return BF_x

@jax.jit
def MLP_jax(params, x):
    for layer in params[:-1]:
        W = layer['W']
        b = layer['B']
        x = jnp.tanh(W @ x + b)
    W = params[-1]['W']
    b = params[-1]['B']
    return W @ x + b

@jax.jit
def MLP(x, params):
    info = []
    in_dim = -1

    def simple_layer(x, A, b):
        return jnp.tanh(A @ x + b)

    for layer in params[:-1]:
        W = layer['W']
        b = layer['B']
        x1 = W @ x + b
        x2 = jnp.tanh(x1)

        if in_dim < 0:
            in_dim = W.shape[1]

        if not len(info):
            # 第一层的雅可比和 Hessian
            jac = jnp.diag(dtanh(x1)) @ W
            hess = tanh_hessian(W, x, b)
            trd = jax.jacobian(jax.hessian(simple_layer))(x, W, b)


            bihar = jax.hessian(jax.hessian(simple_layer))(x, W, b)
            bihar = condense_B(bihar)
        else:
            # 计算后续层的雅可比和 Hessian

            JF_u = jnp.diag(dtanh(x1)) @ W
            jac = JF_u @ info[-1].jac

            HF_u = tanh_hessian(W, x, b)
            hess = fwd_hess(info[-1].hess, jnp.diag(dtanh(x1)) @ W, info[-1].jac, HF_u, [W.shape[0], in_dim])

            TF_u = jax.jacobian(jax.hessian(simple_layer))(x, W, b)
            trd = fwd_trd(JF_u, info[-1].jac, HF_u, info[-1].hess, TF_u, info[-1].trd)

            BF_u = jax.hessian(jax.hessian(simple_layer))(x, W, b)
            bihar = fwd_bihar(JF_u, info[-1].jac, HF_u, info[-1].hess, TF_u, info[-1].trd, BF_u, info[-1].bihar)
        x = x2
        info.append(FHess(x2, jac, hess, trd, bihar))

    W = params[-1]['W']
    b = params[-1]['B']


    if not len(info):
        jac = W
        hess = jnp.zeros((W.shape[0], x.shape[0], x.shape[0]))
        trd = jnp.zeros((W.shape[0], x.shape[0], x.shape[0], x.shape[0]))
        bihar = jnp.zeros((W.shape[0], x.shape[0], x.shape[0]))
    else:

        JF_u = W
        jac = JF_u @ info[-1].jac

        HF_u = jnp.zeros((W.shape[0], x.shape[0], x.shape[0]))
        hess = final_hess(W, info[-1].hess, [W.shape[0], in_dim])

        TF_u = jnp.zeros((W.shape[0], x.shape[0], x.shape[0], x.shape[0]))
        trd = fwd_trd(JF_u, info[-1].jac, HF_u, info[-1].hess, TF_u, info[-1].trd)

        BF_u = jnp.zeros((W.shape[0], x.shape[0], x.shape[0], x.shape[0], x.shape[0]))
        bihar = fwd_bihar(JF_u, info[-1].jac, HF_u, info[-1].hess, TF_u, info[-1].trd, BF_u, info[-1].bihar)
    x = W @ x + b
    info.append(FHess(x, jac, hess, trd, bihar))
    return info

layers = [2, 3, 2]
CNT = 1
params = init_params(layers)


vmap_MLP = jax.jit(jax.vmap(MLP, in_axes=(0, None)))

vmap_MLP_jax = jax.hessian(jax.hessian(MLP_jax, argnums=1), argnums=1)

vmap_MLP_jax = jax.jit(jax.vmap(vmap_MLP_jax, in_axes=(None, 0)))

# 计算前向 bihar 的执行时间
start_time = time.time()
for _ in range(CNT):
    X = jax.random.uniform(jax.random.PRNGKey(0), shape=(2, layers[0]))

    bihar = vmap_MLP(X, params)[-1].bihar

    bihar_jax = vmap_MLP_jax(params, X)

    # judge(bihar, tmp)
    # for _ in range(2):
    #     tmp = condense_B(bihar_jax[_])
    #     # print(tmp)
    #     # print(bihar)
    #     judge(bihar[_], tmp)
duration = time.time() - start_time
print(f'前向 bihar 计算 {CNT} 次，共用时：{duration}')

