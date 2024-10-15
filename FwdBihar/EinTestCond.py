import jax
import jax.numpy as jnp

from util.Jug import judge
import numpy as np

key = jax.random.PRNGKey(0)

# 定义维度
dim_x = 3  # 输入向量 x 的维度
dim_u = 5  # 中间向量 u 的维度
dim_F = 2  # 输出向量 F 的维度

# 初始化参数
A = jax.random.normal(key, (dim_u, dim_x))
b = jax.random.normal(key, (dim_u,))
C = jax.random.normal(key, (dim_F, dim_u))
d = jax.random.normal(key, (dim_F,))


def MLP(x, A, b):
    x = jnp.tanh(A @ x + b)
    return x

# 定义前向计算函数
def MLP_2layer(x, A, b, C, d):
    x = jnp.tanh(A @ x + b)
    x = jnp.tanh(C @ x + d)
    return x

x = jax.random.normal(key, (dim_x, ))
u = MLP(x, A, b)

BF_u = jax.hessian(jax.hessian(MLP))(u, C, d)
Bu_x = jax.hessian(jax.hessian(MLP))(x, A, b)
HF_u = jax.hessian(MLP)(u, C, d)
Hu_x = jax.hessian(MLP)(x, A, b)
TF_u = jax.hessian(jax.jacobian(MLP))(u, C, d)
Tu_x = jax.hessian(jax.jacobian(MLP))(x, A, b)
JF_u = jax.jacobian(MLP)(u, C, d)
Ju_x = jax.jacobian(MLP)(x, A, b)

# term1 = jnp.einsum('ajklm,j1,k2,l3,m4->a1234', BF_u, Ju_x, Ju_x, Ju_x, Ju_x)
# term2 = 6 * jnp.einsum('ajkl,j12,k3,l4->a1234', TF_u, Hu_x, Ju_x, Ju_x)
# term3 = 3 * jnp.einsum('ajk,j12,k34->a1234', HF_u, Hu_x, Hu_x)
# term4 = 4 * jnp.einsum('ajk,j123,k4->a1234', HF_u,Tu_x, Ju_x)
# term5 = jnp.einsum('aj,j1234->a1234', JF_u, Bu_x)
# BF_x = term1 + term2 + term3 + term4 + term5

# term1 = jnp.einsum('ajklm,j1,k2,l3,m4->a1234', BF_u, Ju_x, Ju_x, Ju_x, Ju_x)
# term2 = jnp.einsum('ajkl,j12,k3,l4->a1234', TF_u, Hu_x, Ju_x, Ju_x) + jnp.einsum('ajkl,j13,k2,l4->a1234', TF_u, Hu_x, Ju_x, Ju_x) \
#     + jnp.einsum('ajkl,j14,k2,l3->a1234', TF_u, Hu_x, Ju_x, Ju_x) + jnp.einsum('ajkl,j23,k1,l4->a1234', TF_u, Hu_x, Ju_x, Ju_x) \
#     + jnp.einsum('ajkl,j24,k1,l3->a1234', TF_u, Hu_x, Ju_x, Ju_x) + jnp.einsum('ajkl,j34,k1,l2->a1234', TF_u, Hu_x, Ju_x, Ju_x)
# term3 = jnp.einsum('ajk,j12,k34->a1234', HF_u, Hu_x, Hu_x) + jnp.einsum('ajk,j13,k24->a1234', HF_u, Hu_x, Hu_x) \
#     + jnp.einsum('ajk,j14,k23->a1234', HF_u, Hu_x, Hu_x)
# term4 = jnp.einsum('ajk,j123,k4->a1234', HF_u,Tu_x, Ju_x) + jnp.einsum('ajk,j124,k3->a1234', HF_u,Tu_x, Ju_x) \
#     + jnp.einsum('ajk,j134,k2->a1234', HF_u,Tu_x, Ju_x) + jnp.einsum('ajk,j234,k1->a1234', HF_u,Tu_x, Ju_x)
# term5 = jnp.einsum('aj,j1234->a1234', JF_u, Bu_x)
# BF_x = term1 + term2 + term3 + term4 + term5
#
#
# BF_x_jax = jax.hessian(jax.hessian(MLP_2layer))(x, A, b, C, d)



def condense_B(x):
    y = np.zeros((x.shape[0], x.shape[1], x.shape[1]))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[1]):
                y[i][j][k]= x[i][j][j][k][k]
    return jnp.array(y)


def condense_H(x):
    y = np.zeros((x.shape[0], x.shape[1]))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[i][j] = x[i][j][j]
    return jnp.array(y)

def restore_H(x):
    y = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[1]):
                y[i][j][k] = x[i][j]
    return jnp.array(y)


def condense_T(x):
    y = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[1]):
                y[i][j][k] = x[i][j][j][k]
    return jnp.array(y)
def condense_T_(x):
    y = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[1]):
                y[i][j][k] = x[i][j][k][k]
    return jnp.array(y)

def condense_T1(x):
    y = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                y[i][j] = x[i][i][j]
    return jnp.array(y)


Bu_x_cond = condense_B(Bu_x)
Tu_x_cond = condense_T(Tu_x)
Hu_x_cond = condense_H(Hu_x)

print('condensed_H')
term1 = jnp.einsum('ik,kj->ij',JF_u, Hu_x_cond)
term2 = jnp.einsum('kj,ikm,mj->ij',Ju_x, HF_u, Ju_x)
HF_x = term1 + term2
HF_x_jax = condense_H(jax.hessian(MLP_2layer)(x, A, b, C, d))
judge(HF_x, HF_x_jax)

print('condensed_T')
term1 = jnp.einsum('hj,j13->h13', JF_u, Tu_x_cond)
term2 = jnp.einsum('hjk,j1,k3->h13', HF_u, Hu_x_cond, Ju_x) + jnp.einsum('hjk,j13,k1->h13', HF_u, Hu_x, Ju_x) \
    + jnp.einsum('hjk,j13,k1->h13', HF_u, Hu_x, Ju_x)
term3 = jnp.einsum('hjkl,j1,k1,l3->h13', TF_u, Ju_x, Ju_x, Ju_x)

TF_x = term1 + term2 + term3
TF_x_jax = condense_T(jax.hessian(jax.jacobian(MLP_2layer))(x, A, b, C, d))
judge(TF_x, TF_x_jax)

print('condensed_B')
term1 = jnp.einsum('hj,j13->h13', JF_u, Bu_x_cond)
term2 = 2 * (jnp.einsum('hjk,j13,k3->h13', HF_u, Tu_x_cond, Ju_x) + jnp.einsum('hjk,j31,k1->h13', HF_u, Tu_x_cond, Ju_x))
term3 = jnp.einsum('hjk,j1,k3->h13', HF_u, Hu_x_cond, Hu_x_cond) + 2 * jnp.einsum('hjk,j13,k13->h13', HF_u, Hu_x, Hu_x)
term4 = jnp.einsum('hjkl,j1,k3,l3->h13', TF_u, Hu_x_cond, Ju_x, Ju_x) \
    + 4 * jnp.einsum('hjkl,j13,k1,l3->h13', TF_u, Hu_x, Ju_x, Ju_x) \
    + jnp.einsum('hjkl,j3,k1,l1->h13', TF_u, Hu_x_cond, Ju_x, Ju_x)
term5 = jnp.einsum('hjklm,j1,k1,l3,m3->h13', BF_u, Ju_x, Ju_x, Ju_x, Ju_x)



# term2 = jnp.einsum('ajk,j123,k4->a1234', HF_u,Tu_x, Ju_x) + jnp.einsum('ajk,j124,k3->a1234', HF_u,Tu_x, Ju_x) \
#     + jnp.einsum('ajk,j134,k2->a1234', HF_u,Tu_x, Ju_x) + jnp.einsum('ajk,j234,k1->a1234', HF_u,Tu_x, Ju_x)
# term2 = condense_B(term2)

BF_x_cond = term1 + term2 + term3 + term4 + term5
BF_x_jax_cond = condense_B(jax.hessian(jax.hessian(MLP_2layer))(x, A, b, C, d))
judge(BF_x_cond, BF_x_jax_cond)
