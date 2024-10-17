import sys
from ruamel.yaml import YAML
from typing import Dict, Any
import time, os

# import lapjax
# import lapjax.numpy as jnp
# from lapjax import vmap, grad, jit, config

import optax
import jaxopt

import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, config
from jax import tree_util

config.update("jax_enable_x64", True)
yaml = YAML(typ='safe')
yaml.default_flow_style = False

@ jax.jit
def MLP(params, x):
    """
    多层感知机函数
    
    参数:
    params: 网络参数列表,每个元素是一个包含'W'(权重)和'B'(偏置)的字典
    x: 输入数据

    返回:
    网络的输出
    """
    for layer in params[:-1]:
        x = jnp.tanh(jnp.dot(layer['W'], x) + layer['B'])
    return jnp.dot(params[-1]['W'], x) + params[-1]['B']

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
def MLP_fwdbihar(x, params):
    info = None
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

        if info is None:
            # 第一层的雅可比和 Hessian
            jac = jnp.diag(dtanh(x1)) @ W
            hess = tanh_hessian(W, x, b)
            trd = jax.jacobian(jax.hessian(simple_layer))(x, W, b)


            bihar = jax.hessian(jax.hessian(simple_layer))(x, W, b)
            bihar = condense_B(bihar)
        else:
            # 计算后续层的雅可比和 Hessian

            JF_u = jnp.diag(dtanh(x1)) @ W
            jac = JF_u @ info.jac

            HF_u = tanh_hessian(W, x, b)
            hess = fwd_hess(info.hess, jnp.diag(dtanh(x1)) @ W, info.jac, HF_u, [W.shape[0], in_dim])

            TF_u = jax.jacobian(jax.hessian(simple_layer))(x, W, b)
            trd = fwd_trd(JF_u, info.jac, HF_u, info.hess, TF_u, info.trd)

            BF_u = jax.hessian(jax.hessian(simple_layer))(x, W, b)
            bihar = fwd_bihar(JF_u, info.jac, HF_u, info.hess, TF_u, info.trd, BF_u, info.bihar)
        x = x2
        info = FHess(x2, jac, hess, trd, bihar)

    W = params[-1]['W']
    # b = params[-1]['B']


    if info is None:
        # jac = W
        # hess = jnp.zeros((W.shape[0], x.shape[0], x.shape[0]))
        # trd = jnp.zeros((W.shape[0], x.shape[0], x.shape[0], x.shape[0]))
        bihar = jnp.zeros((W.shape[0], x.shape[0], x.shape[0]))
    else:

        JF_u = W
        # jac = JF_u @ info.jac

        HF_u = jnp.zeros((W.shape[0], x.shape[0], x.shape[0]))
        # hess = final_hess(W, info.hess, [W.shape[0], in_dim])

        TF_u = jnp.zeros((W.shape[0], x.shape[0], x.shape[0], x.shape[0]))
        # trd = fwd_trd(JF_u, info.jac, HF_u, info.hess, TF_u, info.trd)

        BF_u = jnp.zeros((W.shape[0], x.shape[0], x.shape[0], x.shape[0], x.shape[0]))
        bihar = fwd_bihar(JF_u, info.jac, HF_u, info.hess, TF_u, info.trd, BF_u, info.bihar)
    # x = W @ x + b

    bihar = jnp.sum(bihar, axis=(1, 2))
    info = bihar
    return info

def init_params(layers):
    """
    初始化网络参数
    
    参数:
    layers: 一个列表,包含每层的神经元数量

    返回:
    初始化后的网络参数列表
    """
    keys = jax.random.split(jax.random.PRNGKey(0), len(layers)-1)
    params = []
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        lb, ub = -(1 / jnp.sqrt(n_in)), (1 / jnp.sqrt(n_in))
        W = lb + (ub-lb) * jax.random.uniform(key, shape=(n_out, n_in))
        B = jax.random.uniform(key, shape=(n_out,))
        params.append({'W': W, 'B': B})
    return params

@jit
def h(x, d):
    """
    边界条件函数
    
    参数:
    x: 输入数据
    d: 维度

    返回:
    边界条件函数的值
    """
    return (jnp.sum(x, axis=-1) / d) ** 2 + jnp.sin(jnp.sum(x, axis=-1) / d)

# @jit
# def f(x, d):
#     """
#     源项函数
#
#     参数:
#     x: 输入数据
#     d: 维度
#
#     返回:
#     源项函数的值
#     """
#
#     return jnp.sin(jnp.sum(x, axis=-1) / d) / d - 2 / d


@jit
def f(x, d):
    """
    源项函数

    参数:
    x: 输入数据
    d: 维度

    返回:
    源项函数的值
    """

    return - 1 / d ** 2 * jnp.sin(jnp.sum(x, axis=-1) / d)

@jit
def u(x, d):
    """
    精确解函数
    
    参数:
    x: 输入数据
    d: 维度

    返回:
    精确解函数的值
    """
    s = jnp.sum(x, axis=-1) / d
    return (s) ** 2 + jnp.sin(s)

def get_laplacian_function_lapjax(func):
    """
    使用lapjax获取拉普拉斯算子函数
    
    参数:
    func: 原始函数

    返回:
    拉普拉斯算子函数
    """
    from lapjax import LapTuple
    def lap(data):
        input_laptuple = LapTuple(data, is_input=True)
        output_laptuple = func(input_laptuple)
        return output_laptuple.lap
    return lap

def get_laplacian_function_orig(func):
    """
    使用原始方法获取拉普拉斯算子函数
    
    参数:
    func: 原始函数

    返回:
    拉普拉斯算子函数
    """
    def lap(data):
        hess = jax.hessian(func)(data)
        return jnp.trace(hess, axis1=-1, axis2=-2)
    return lap

def get_bihar_func(func):
    @jax.jit
    def condense_B(x):
        # 假设x的形状是(m, n, n, n, n)
        # 取出x[:, j, j, k, k]，并将结果存储到y
        y = x[:, jnp.arange(x.shape[1]), jnp.arange(x.shape[1]), :, :]
        y = y[:, :, jnp.arange(x.shape[1]), jnp.arange(x.shape[1])]
        return y

    def bihar(data):
        B = jax.hessian(jax.hessian(func))(data)
        B = condense_B(B)

        return jnp.sum(B, axis=(1, 2))
    return bihar


def loss_fn(params, interior_points, boundary_points, d, laplacian_method):
    """
    损失函数
    
    参数:
    params: 网络参数
    interior_points: 内部点
    boundary_points: 边界点
    d: 维度
    laplacian_method: 拉普拉斯算子计算方法

    返回:
    损失值
    """
    def u_net(x):
        return MLP(params, x)
    
    if laplacian_method == 'fwdbihar':
        bihar_u = jit(vmap(MLP_fwdbihar, in_axes=(0, None)))
        vmap_bihar = bihar_u
    else:

        bihar_u = get_bihar_func(u_net)
        vmap_bihar = jit(vmap(bihar_u))

    interior_loss = jnp.mean((-vmap_bihar(interior_points).reshape(-1) - f(interior_points, d)) ** 2)
    boundary_loss = jnp.mean((u_net(boundary_points).reshape(-1) - h(boundary_points, d)) ** 2)
    return interior_loss + boundary_loss

def generate_data(num_interior, num_boundary, d):
    """
    生成训练或测试数据
    
    参数:
    num_interior: 内部点数量
    num_boundary: 边界点数量
    d: 维度

    返回:
    interior_points: 内部点
    boundary_points: 边界点
    """
    interior_points = jax.random.uniform(jax.random.PRNGKey(0), (num_interior, d), minval=-1, maxval=1)

    boundary_points = []
    for i in range(d):
        for val in [-1, 1]:
            points = jax.random.uniform(jax.random.PRNGKey(i), (num_boundary // (2*d), d), minval=-1, maxval=1)
            points = points.at[:, i].set(val)
            boundary_points.append(points)

    boundary_points = jnp.concatenate(boundary_points, axis=0)

    return interior_points, boundary_points

def train_adam(
    params,
    interior_points, boundary_points,
    adam_args,
    num_epochs,
    d,
    laplacian_method
):
    """
    使用Adam优化器训练网络
    
    参数:
    params: 初始网络参数
    interior_points: 内部点
    boundary_points: 边界点
    adam_args: Adam优化器参数
    num_epochs: 训练轮数
    d: 维度
    laplacian_method: 拉普拉斯算子计算方法

    返回:
    优化后的网络参数
    """
    optimizer = optax.adam(**adam_args)
    opt_state = optimizer.init(params)

    @jit
    def step(params, opt_state, interior_points, boundary_points):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, interior_points, boundary_points, d, laplacian_method))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for epoch in range(num_epochs):
        params, opt_state, loss = step(params, opt_state, interior_points, boundary_points)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return params

def train_lm(
    params,
    interior_points, boundary_points,
    d,
    lm_args,
    laplacian_method
):
    """
    使用Levenberg-Marquardt算法训练网络
    
    参数:
    params: 初始网络参数
    interior_points: 内部点
    boundary_points: 边界点
    d: 维度
    lm_args: Levenberg-Marquardt算法参数
    laplacian_method: 拉普拉斯算子计算方法

    返回:
    优化后的网络参数
    """
    params_flat, params_unravel = jax.flatten_util.ravel_pytree(params)

    def residual_fun(params_flat):
        params = params_unravel(params_flat)
        def u_net(x):
            return MLP(params, x)

        if laplacian_method == 'fwdbihar':
            bihar_u = jax.jit(jax.vmap(MLP_fwdbihar, in_axes=(0, None)))
            vmap_bihar = bihar_u
        else:
            bihar_u = get_bihar_func(u_net)
            vmap_bihar = jit(vmap(bihar_u))


        u_net = jax.jit(jax.vmap(u_net))
        if laplacian_method == 'fwdbihar':
            # interior_residuals = - f(interior_points, d)
            interior_residuals = (-vmap_bihar(interior_points, params).reshape(-1) - f(interior_points, d))
        else:
            interior_residuals = (-vmap_bihar(interior_points).reshape(-1) - f(interior_points, d))

        boundary_residuals = (u_net(boundary_points).reshape(-1) - h(boundary_points, d))

        return jnp.concatenate([interior_residuals, boundary_residuals])

    solver = jaxopt.LevenbergMarquardt(
        residual_fun,
        verbose=True,
        jit=True,
        **lm_args
    )

    params_flat_optimized, state = solver.run(params_flat)
    params_optimized = params_unravel(params_flat_optimized)

    return params_optimized

def train_lbfgs(
    params,
    interior_points, boundary_points,
    d,
    lbfgs_args,
    laplacian_method
):
    """
    使用L-BFGS算法训练网络
    
    参数:
    params: 初始网络参数
    interior_points: 内部点
    boundary_points: 边界点
    d: 维度
    lbfgs_args: L-BFGS算法参数
    laplacian_method: 拉普拉斯算子计算方法

    返回:
    优化后的网络参数
    """
    def objective(params):
        return loss_fn(params, interior_points, boundary_points, d, laplacian_method)

    solver = jaxopt.LBFGS(
        fun=objective,
        verbose=True,
        jit=True,
        **lbfgs_args
    )

    params_optimized, state = solver.run(params)

    return params_optimized

def calculate_error(params, test_points, d, error_type, norm_type):
    """
    计算误差
    
    参数:
    params: 网络参数
    test_points: 测试点
    d: 维度
    error_type: 误差类型 ('relative' 或 'absolute')
    norm_type: 范数类型 ('L2' 或 'Linf')

    返回:
    计算得到的误差
    """
    def u_net(x):
        return MLP(params, x)
    
    true_values = u(test_points, d)

    u_net = jax.jit(jax.vmap(u_net))
    predicted_values = u_net(test_points).reshape(-1)
    
    if norm_type == 'L2':
        if error_type == 'relative':
            error = jnp.linalg.norm(true_values - predicted_values) / jnp.linalg.norm(true_values)
        else:  # absolute
            error = jnp.linalg.norm(true_values - predicted_values)
    elif norm_type == 'Linf':
        if error_type == 'relative':
            error = jnp.max(jnp.abs(true_values - predicted_values) / jnp.abs(true_values))
        else:  # absolute
            error = jnp.max(jnp.abs(true_values - predicted_values))
    else:
        raise ValueError("Unsupported norm type. Choose 'L2' or 'Linf'.")
    
    return float(error)

def run_experiment(
    d,
    optimizer,
    training_args,
    optimizer_args,
    num_interior,
    num_boundary,
    layers,
    laplacian_method='lapjax'):
    """
    运行单个实验
    
    参数:
    d: 维度
    optimizer: 优化器类型 ('adam', 'lm' 或 'lbfgs')
    training_args: 训练参数（仅限Adam，epochs数）
    optimizer_args: 优化器参数
    num_interior: 内部点数量
    num_boundary: 边界点数量
    layers: 网络层结构
    laplacian_method: 拉普拉斯算子计算方法

    返回:
    训练后的网络参数
    """
    params = init_params(layers)
    interior_points, boundary_points = generate_data(num_interior, num_boundary, d)

    if optimizer == 'adam':
        trained_params = train_adam(params, interior_points, boundary_points, optimizer_args, training_args["epochs"], d, laplacian_method)
    elif optimizer == 'lm':
        trained_params = train_lm(params, interior_points, boundary_points, d, optimizer_args, laplacian_method)
    elif optimizer == 'lbfgs':
        trained_params = train_lbfgs(params, interior_points, boundary_points, d, optimizer_args, laplacian_method)
    else:
        raise ValueError("Unsupported optimizer. Choose 'adam' or 'lm'.")

    return trained_params

def test_model(trained_params, d, test_interior, test_boundary, error_type, norm_type):
    """
    测试模型
    
    参数:
    trained_params: 训练后的网络参数
    d: 维度
    test_interior: 测试用内部点数量
    test_boundary: 测试用边界点数量
    error_type: 误差类型
    norm_type: 范数类型

    返回:
    interior_error: 内部点误差
    boundary_error: 边界点误差
    """
    test_in, test_b = generate_data(test_interior, test_boundary, d)
    
    interior_error = calculate_error(trained_params, test_in, d, error_type, norm_type)
    boundary_error = calculate_error(trained_params, test_b, d, error_type, norm_type)
    
    return interior_error, boundary_error

def load_config(file_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    参数:
    file_path: 配置文件路径

    返回:
    配置字典
    """
    with open(file_path, 'r', encoding='UTF-8') as file:
        return yaml.load(file)

def run_experiments(config: Dict[str, Any]):
    """
    运行多个实验
    
    参数:
    config: 配置字典

    返回:
    实验结果列表
    """
    results = {}
    for exp_name, exp_config in config['experiments'].items():
        print(f"[INFO] Running experiment: {exp_name}")
        
        d = exp_config['d']
        optimizer = exp_config['optimizer']
        optimizer_args = exp_config['optimizer_args']
        training_args = exp_config['training_args'] if 'training_args' in exp_config else None
        num_interior = exp_config['num_interior']
        num_boundary = exp_config['num_boundary']
        layers = exp_config['layers']
        laplacian_method = exp_config['laplacian_method']

        begin_t = time.time()
        trained_params = run_experiment(
            d, optimizer, training_args, optimizer_args, 
            num_interior, num_boundary, layers, laplacian_method
        )
        end_t = time.time()

        duration_seconds = end_t - begin_t

        print("[INFO] Training completed. Now testing the model...")

        test_config = config['test']
        test_interior = test_config['test_interior']
        test_boundary = test_config['test_boundary']

        l2_rel_in, l2_rel_b = test_model(
            trained_params, d, test_interior, test_boundary, "relative", "L2"
        )
        l2_abs_in, l2_abs_b = test_model(
            trained_params, d, test_interior, test_boundary, "absolute", "L2"
        )
        linf_rel_in, linf_rel_b = test_model(
            trained_params, d, test_interior, test_boundary, "relative", "Linf"
        )
        linf_abs_in, linf_abs_b = test_model(
            trained_params, d, test_interior, test_boundary, "absolute", "Linf"
        )

        results[exp_name] = {
            'l2_error':
            {
                'interior': {'relative': l2_rel_in, 'absolute': l2_abs_in},
                'boundary': {'relative': l2_rel_b, 'absolute': l2_abs_b}
            },
            'linf_error':
            {
                'interior': {'relative': linf_rel_in, 'absolute': linf_abs_in},
                'boundary': {'relative': linf_rel_b, 'absolute': linf_abs_b}
            },
            'duration_seconds': duration_seconds
        }

    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to the YAML config file.")
        sys.exit(1)

    config_file = sys.argv[1]
    config = load_config(config_file)
    
    results = run_experiments(config)

    for exp_name, exp_result in results.items():
        config['experiments'][exp_name]['result'] = exp_result

    res_dir = './result/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    total_files = len([file for file in os.listdir(res_dir)])
    hibar_res_dir = os.path.join(res_dir, f'{total_files + 1}')
    os.makedirs(hibar_res_dir)
    res_file = os.path.join(hibar_res_dir, 'result.yaml')
    with open(res_file, 'w') as file:
        yaml.dump(config, file)