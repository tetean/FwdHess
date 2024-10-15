import jax
import jax.numpy as jnp
from util.Jug import judge
import numpy as np

key = jax.random.PRNGKey(0)

# 定义维度
dim_x = 2  # 输入向量 x 的维度
dim_u = 2  # 中间向量 u 的维度
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

term1 = jnp.einsum('ajklm,j1,k2,l3,m4->a1234', BF_u, Ju_x, Ju_x, Ju_x, Ju_x)
term2 = jnp.einsum('ajkl,j12,k3,l4->a1234', TF_u, Hu_x, Ju_x, Ju_x) + jnp.einsum('ajkl,j13,k2,l4->a1234', TF_u, Hu_x, Ju_x, Ju_x) \
    + jnp.einsum('ajkl,j14,k2,l3->a1234', TF_u, Hu_x, Ju_x, Ju_x) + jnp.einsum('ajkl,j23,k1,l4->a1234', TF_u, Hu_x, Ju_x, Ju_x) \
    + jnp.einsum('ajkl,j24,k1,l3->a1234', TF_u, Hu_x, Ju_x, Ju_x) + jnp.einsum('ajkl,j34,k1,l2->a1234', TF_u, Hu_x, Ju_x, Ju_x)
term3 = jnp.einsum('ajk,j12,k34->a1234', HF_u, Hu_x, Hu_x) + jnp.einsum('ajk,j13,k24->a1234', HF_u, Hu_x, Hu_x) \
    + jnp.einsum('ajk,j14,k23->a1234', HF_u, Hu_x, Hu_x)
term4 = jnp.einsum('ajk,j123,k4->a1234', HF_u,Tu_x, Ju_x) + jnp.einsum('ajk,j124,k3->a1234', HF_u,Tu_x, Ju_x) \
    + jnp.einsum('ajk,j134,k2->a1234', HF_u,Tu_x, Ju_x) + jnp.einsum('ajk,j234,k1->a1234', HF_u,Tu_x, Ju_x)
term5 = jnp.einsum('aj,j1234->a1234', JF_u, Bu_x)
BF_x = term1 + term2 + term3 + term4 + term5


BF_x_jax = jax.hessian(jax.hessian(MLP_2layer))(x, A, b, C, d)



def condense_B(x):
    y = np.zeros((x.shape[0], x.shape[1], x.shape[1]))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                y[i][j][k]= x[i][j][j][k][k]
    return jnp.array(y)


judge(BF_x, BF_x_jax)


