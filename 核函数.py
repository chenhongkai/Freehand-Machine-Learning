from numpy import  ndarray, exp, zeros

class LinearKernel:
    """
    线性核函数（Linear kernel function）
    核函数值 K(x_, z_) = x_ @ z_
    """
    def __call__(self, X__: ndarray, Z__: ndarray) -> ndarray:
        if X__.ndim==1 and 1<=Z__.ndim<=2:     # 若X__为1维ndarray，且Z__为1维或2维ndarray
            return Z__ @ X__
        elif 1<=X__.ndim<=2 and Z__.ndim==1:   # 若Z__为1维ndarray，且X__为1维或2维ndarray
            return X__ @ Z__
        elif X__.ndim==2 and Z__.ndim==2:      # 若X__、Z__均为2维ndarray
            K__ = zeros([len(X__), len(Z__)])  # 核函数值矩阵
            for n, x_ in enumerate(X__):
                K__[n] = Z__ @ x_
            return K__
        else:
            raise ValueError('输入量X__、Z__应为ndarray，其属性ndim应为1或2')

class PolyKernel:
    """
    多项式核函数（Polynomial kernel function）
    核函数值 K(x_, z_) = (γ * x_ @ z_ + r)**d
    其中，d、γ、r为超参数
    """
    def __init__(self,
            d: int = 2,
            r: float = 1.,
            γ: float = 1.,
            ):
        assert type(d)==int and d>=1, '多项式核函数的指数d应为正整数，K(x_, z_) = (γ * x_ @ z_ + r)**d'
        assert γ>0, '多项式核函数的超参数γ应为正数，K(x_, z_) = (γ * x_ @ z_ + r)**d'
        self.d = d  # 指数d
        self.r = r  # 参数r
        self.γ = γ  # 参数γ
        self.linearKernel = LinearKernel()  # 实例化线性核函数

    def __call__(self, X__: ndarray, Z__: ndarray) -> ndarray:
        K__ = self.linearKernel(X__, Z__)    # 线性核函数的输出
        K__ = (self.γ*K__ + self.r)**self.d  # 多项式核函数的输出
        return K__

class RBFKernel:
    """
    高斯核函数（Gaussian kernel function），也称径向基函数（Radial Basis Function）
    核函数值 K(x_, z_) = exp(-γ * sum((x_ - z_)**2))
    其中，γ为超参数
    """
    def __init__(self, γ: float = 1.):
        assert γ>0, 'RBF核函数的超参数γ应为正数'
        self.γ = γ  # 参数γ

    def __call__(self, X__: ndarray, Z__: ndarray) -> ndarray:
        if ((X__.ndim==1 and Z__.ndim==2) or
            (X__.ndim==2 and Z__.ndim==1)):  # 若X__、Z__其中之一为1维ndarray，另一个2维ndarray
            return exp(-self.γ * ((X__ - Z__)**2).sum(axis=1))
        elif X__.ndim==1 and Z__.ndim==1:    # 若X__、Z__均为1维ndarray
            return exp(-self.γ * ((X__ - Z__)**2).sum())
        elif X__.ndim==2 and Z__.ndim==2:    # 若X__、Z__均为2维ndarray
            D2__ = (X__**2).sum(axis=1, keepdims=True) + (Z__**2).sum(axis=1) - 2*X__ @ Z__.T  # 距离平方矩阵
            return exp(-self.γ * D2__)
        else:
            raise ValueError('输入量X__、Z__应为ndarray，其属性ndim应为1或2')
