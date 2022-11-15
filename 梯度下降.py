from typing import Sequence
from numpy import ndarray, maximum, zeros_like

class GradientDescent:
    """梯度下降法"""
    def __init__(self,
             θ__: ndarray,          # ndarray类型的向量或矩阵：待优化的参数
             LR: float = 0.01,      # 全局学习率
             method: str = 'Adam',  # 学习率调整策略，可选'SGD'/'AdaGrad'/'RMSprop'/'AdaDelta'/'Momentum'/'Adam'/'Nesterov'/'AdaMax'/'Nadam'
             decayRates_: Sequence[float] = (.9, .999),  # 两个衰减率
             ):
        assert type(θ__)==ndarray, '待优化的参数θ__的数据类型应为ndarray'
        assert LR>0, '全局学习率LR应为正数'
        assert len(decayRates_)==2, '衰减率参数decayRates_长度应为2'
        assert 0<=decayRates_[0]<1, '首项衰减率的取值范围为[0, 1)'
        assert 0<=decayRates_[1]<1, '次项衰减率的取值范围为[0, 1)'
        self.θ__ = θ__                  # ndarray类型的向量或矩阵：待优化的参数
        self.LR = LR                    # 全局学习率
        self.β1, self.β2 = decayRates_  # 衰减率参数（第一衰减率、第二衰减率）
        self.Δθ__ = zeros_like(θ__)     # 向量或矩阵：参数更新的增量
        self.method = method.lower()    # 学习率调整策略，可选'SGD'/'AdaGrad'/'RMSprop'/'AdaDelta'/'Momentum'/'Adam'/'Nesterov'/'AdaMax'/'Nadam'
        self.m__ = zeros_like(θ__)      # 调整学习率所用到的累计参数
        self.n__ = zeros_like(θ__)      # 调整学习率所用到的累计参数
        self.t = 0                      # 更新的次数
        """选取学习率调整策略"""
        if self.method=='adam':
            self.optmizer = self.Adam
        elif self.method=='adamax':
            self.optmizer = self.AdaMax
        elif self.method=='nadam':
            self.optmizer = self.Nadam
        elif self.method=='adagrad':
            self.optmizer = self.AdaGrad
        elif self.method=='rmsprop':
            self.optmizer = self.RMSprop
        elif self.method=='adadelta':
            self.optmizer = self.AdaDelta
        elif self.method=='momentum':
            self.optmizer = self.Momentum
        elif self.method=='nesterov':
            self.optmizer = self.Nesterov
        elif self.method=='sgd':
            self.optmizer = self.SGD
        else:
            raise ValueError(f"未定义学习率调整策略'{method}'")

    def update(self, grad__: ndarray) -> ndarray:
        """输入参数的梯度（向量或矩阵），更新参数"""
        self.t += 1            # 更新的次数
        self.optmizer(grad__)  # 得到参数更新的增量self.Δθ__
        self.θ__ += self.Δθ__  # 更新参数
        return self.θ__

    def Adam(self, grad__):
        """
        Adam学习率调整策略
        参考2015 Diederik P. Kingma和Jimmy Lei Ba的论文： Adam: A method for stochastic optimization
        Algorithm 1: Adam, our proposed algorithm for stochastic optimization
        """
        t = self.t     # 读取：更新的次数
        β1 = self.β1   # 读取：衰减率
        β2 = self.β2   # 读取：衰减率
        self.m__[:] = β1*self.m__ + (1 - β1)*grad__     # 更新“有偏一阶矩估计”（用于Momentum）
        self.n__[:] = β2*self.n__ + (1 - β2)*grad__**2  # 更新“有偏二阶原始矩估计”（用于RMSprop）
        self.Δθ__[:] = -self.LR*(1 - β2**t)**0.5/(1 - β1**t) * self.m__/(self.n__**0.5 + 1e-8)  # 参数更新的增量

    def AdaMax(self, grad__):
        """
        AdaMax学习率调整策略
        参考2015 Diederik P. Kingma和Jimmy Lei Ba的论文： Adam: A method for stochastic optimization
        Algorithm 2: AdaMax, a variant of Adam based on the infinity norm
        """
        t = self.t    # 读取：更新的次数
        β1 = self.β1  # 读取：衰减率
        β2 = self.β2  # 读取：衰减率
        self.m__[:] = β1*self.m__ + (1 - β1)*grad__      # 更新“有偏一阶矩估计”
        self.n__[:] = maximum(β2*self.n__, abs(grad__))  # 更新“指数加权无穷大范数”
        self.Δθ__[:] = -self.LR/(1 - β1**t)*self.m__/(self.n__ + 1e-7)  # 参数更新的增量

    def Nadam(self, grad__):
        """
        Nadam学习率调整策略
        参考2016 Timothy Dozat的论文：Incorporating Nesterov Momentum into Adam
        Algorithm 8 Nesterov-accelerated adaptive moment estimation
        """
        t = self.t    # 读取：更新的次数
        β1 = self.β1  # 读取：衰减率
        β2 = self.β2  # 读取：衰减率
        self.m__[:] = β1*self.m__ + (1 - β1)*grad__     # 更新“有偏一阶矩估计”
        self.n__[:] = β2*self.n__ + (1 - β2)*grad__**2  # 更新“有偏二阶原始矩估计”
        mHat__ = self.m__/(1 - β1**(t + 1))
        nHat__ = self.n__/(1 - β2**t)
        mHat__ = (1 - β1)*grad__/(1 - β1**t) + β1*mHat__
        self.Δθ__[:] = -self.LR*mHat__/(nHat__**0.5 + 1e-7)  # 参数更新的增量

    def AdaGrad(self, grad__):
        """
        AdaGrad学习率调整策略
        参考2016 Timothy Dozat的论文：Incorporating Nesterov Momentum into Adam
        Algorithm 4 AdaGrad
        """
        self.n__ += grad__**2  # 对梯度平方累计求和
        self.Δθ__[:] = -self.LR*grad__/(self.n__**0.5 + 1e-7)  # 参数更新的增量

    def RMSprop(self, grad__):
        """
        RMSprop学习率调整策略
        参考Ian Goodfellow《深度学习》8.5.2 RMSProp，算法8.5 RMSProp算法
        """
        β1 = self.β1  # 读取：衰减率
        self.m__[:] = β1*self.m__ + (1 - β1)*grad__**2         # 更新“梯度平方的指数衰减移动平均”
        self.Δθ__[:] = -self.LR*grad__/(self.m__ + 1e-6)**0.5  # 参数更新的增量

    def AdaDelta(self, grad__):
        """
        AdaDelta学习率调整策略
        参考2012 Matthew D. Zeiler的论文：Adadelta: An adaptive learning rate method
        Algorithm 1 Computing ADADELTA update at time t
        该策略不需要全局学习率
        """
        ρ = self.β1   # 读取：衰减率
        self.m__[:] = ρ*self.m__ + (1 - ρ)*grad__**2     # 更新“梯度平方的指数衰减移动平均”
        self.Δθ__[:] = -(self.n__ + 1e-7)**0.5/(self.m__ + 1e-7)**0.5*grad__  # 参数更新差值
        self.n__[:] = ρ*self.n__ + (1 - ρ)*self.Δθ__**2  # 更新“参数更新差值Δθ平方的指数衰减移动平均”

    def Momentum(self, grad__):
        """
        经典Momentum学习率调整策略
        参考Ian Goodfellow《深度学习》8.3.2 动量
        """
        self.Δθ__[:] = self.β1*self.Δθ__ - self.LR*grad__  # 参数更新的增量

    def Nesterov(self, grad__):
        """
        Nesterov’s accelerated gradient学习率调整策略
        参考Timothy Dozat的论文：Incorporating Nesterov Momentum into Adam
        Algorithm 7 NAG rewritten
        """
        μ = self.β1  # 读取：衰减率
        self.m__[:] = μ*self.m__ + grad__
        mHat__ = grad__ + μ*self.m__
        self.Δθ__[:] = -self.LR*mHat__  # 参数更新的增量

    def SGD(self, grad__):
        """标准梯度下降（Standard gradient descent）"""
        self.Δθ__[:] = -self.LR*grad__  # 参数更新的增量
