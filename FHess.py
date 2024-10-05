"""
@author: tetean
@time: 2024/10/5 21:59
@info: 
"""
class FHess:
    """
    Hessian 类，用于存储向量及其雅可比和 Hessian
    """
    def __init__(self, x=None, jac=None, hess=None):
        self.x = x      # 向量
        self.jac = jac  # 雅可比矩阵
        self.hess = hess  # Hessian 张量

    def tree_flatten(self):
        """
        将类实例拆分为子元素和辅助数据。
        返回一个元组（子元素的元组, 辅助数据）
        """
        children = (self.x, self.jac, self.hess)
        aux_data = None  # 辅助数据
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        从子元素和辅助数据中重建类实例。
        """
        x, jac, hess = children
        return cls(x, jac, hess)

