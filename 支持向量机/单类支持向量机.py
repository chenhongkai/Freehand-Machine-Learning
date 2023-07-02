from random import choice, random

from numpy import array, ndarray, where, isnan, intersect1d, ones
import matplotlib.pyplot as plt

from 核函数 import LinearKernel, RBFKernel, PolyKernel


class OneClassSupportVectorMachine:
    """单类支持向量机"""
    def __init__(self,
            ν: float = 0.1,          # 超参数：训练样本集当中异常样本所占比例的上限
            kernel: str = 'linear',  # 核函数：可选 线性核'linear'/高斯核'RBF'/多项式核'poly'
            maxIterations: int = 10000,  # 最大迭代次数
            tol: float = 1e-3,  # 迭代停止的收敛精度（指SMO求解算法当中最大违反KKT条件的程度）
            d: int = 2,         # 超参数：多项式核函数的指数
            γ: float = 1.,      # 超参数：高斯核函数、多项式核函数的参数
            r: float = 1.,      # 超参数：多项式核函数的参数
            ):
        assert 0<ν<1, '训练样本集当中异常样本所占比例的上限ν的取值范围应为(0, 1)'
        assert type(maxIterations)==int and maxIterations>0, '最大迭代次数maxIterations应为正整数'
        assert tol>0, '收敛精度tol应大于0'
        assert type(d)==int and d>=1, '多项式核函数的指数d应为不小于1的正整数'
        assert γ>0, '高斯核函数、多项式核函数的参数γ应大于0'
        self.ν = ν       # 超参数：训练样本集当中异常样本所占比例的上限
        self.kernel = kernel.lower()        # 核函数：可选 线性核'linear'/高斯核'RBF'/多项式核'poly'
        self.maxIterations = maxIterations  # 最大迭代次数
        self.tol = tol   # 迭代停止的收敛精度（指SMO求解算法当中最大违反KKT条件的程度）
        self.d = d       # 超参数：多项式核函数的指数
        self.γ = γ       # 超参数：高斯核函数、多项式核函数的参数
        self.r = r       # 超参数：多项式核函数的参数
        self.M = None    # 输入特征向量的维数
        self.w_ = None   # M维向量：权重向量
        self.ρ = None    # 偏离距离ρ，决策超平面：ω_ @ φ(x_) - ρ = 0
        self.α_ = None   # N维向量：所有N个训练样本的拉格朗日乘子
        self.supportVectors__ = None  # 矩阵：所有支持向量
        self.αSV_ = None              # 向量：所有支持向量对应的拉格朗日乘子α
        self.minimizedObjectiveValues_ = None  # 列表：历次迭代的最小化目标函数值，对于SMO求解算法，指对偶问题的最小化目标函数值
        """选择核函数"""
        if self.kernel=='linear':
            self.kernelFunction = LinearKernel()
            print('使用线性核函数')
        elif self.kernel=='poly':
            self.kernelFunction = PolyKernel(d=self.d, γ=self.γ, r=self.r)
            print('使用多项式核函数')
        elif self.kernel=='rbf':
            self.kernelFunction = RBFKernel(γ=self.γ)
            print('使用高斯核函数')
        else:
            raise ValueError(f"未定义核函数'{kernel}'")

    def fit(self, X__: ndarray):
        """训练：使用SMO（Sequential Minimal Optimization）算法求解对偶问题，最小化目标函数"""
        # 求解决策超平面： ω_ @ φ(x)_ - ρ = 0
        assert type(X__)==ndarray and X__.ndim==2, '输入训练样本矩阵X__应为2维ndarray'
        ν = self.ν             # 读取：训练样本集当中异常样本所占比例的上限
        N, self.M = X__.shape  # 训练样本数量、输入特征向量的维数
        C = 1/(ν*N)            # 惩罚参数
        K__ = self.kernelFunction(X__, X__)  # N×N矩阵：核函数矩阵
        α_ = ones(N)/N               # N维向量：初始化所有拉格朗日乘子α
        ρ = 0.                       # 初始化偏离距离
        self.minimizedObjectiveValues_ = []  # 列表：记录历次迭代的目标函数值
        minimizedObjectiveValue = 0.5*(α_ @ K__ @ α_)    # 初始化：目标函数值
        for t in range(1, self.maxIterations + 1):
            indexSV_ = where(α_>0)[0]                   # 数组索引：索引满足α>0的支持向量
            indexNonBound_ = where((0<α_) & (α_<C))[0]  # 数组索引：索引满足0<α<C的支持向量
            self.minimizedObjectiveValues_.append(minimizedObjectiveValue)  # 记录当前目标函数值
            """检验所有样本点是否满足KKT条件，并计算各自违反KKT条件的程度"""
            wφx_ = α_[indexSV_] @ K__[indexSV_, :]         # N维向量：高维空间中，权重向量w与样本向量φ(x)的内积
            violateKKT_ = abs(wφx_ - ρ)                    # N维向量：开始计算“违反KKT条件的程度”
            violateKKT_[(α_==0) & (wφx_>=ρ)] = 0.          # N维向量：KKT条件 α=0 ←→ wφx - ρ >= 0
            violateKKT_[(0<α_) & (α_<C) & (wφx_==ρ)] = 0.  # N维向量：KKT条件 0<α<C ←→ wφx - ρ = 0
            violateKKT_[(α_==C) & (wφx_<=ρ)] = 0.          # N维向量：KKT条件 α=C ←→ wφx - ρ <= 0
            if violateKKT_.max()<self.tol:
                print(f'第{t}次SMO迭代，最大违反KKT条件的程度达到收敛精度{self.tol}，停止迭代!')
                break
            """选择αi"""
            indexViolateKKT_ = where(violateKKT_>0)[0]                               # 数组索引：找出违反KKT条件的α
            indexNonBoundViolateKKT_ = intersect1d(indexViolateKKT_, indexNonBound_) # 数组索引：找出违反KKT条件的非边界α
            if random()<0.85:
                # 有较大的概率（85%）选取违KKT反条件程度最大的αi进行下一步优化，若有非边界α，首选非边界α
                if len(indexNonBoundViolateKKT_)>0:
                    # 若存在违反KKT条件的非边界α，则选取违反KKT条件程度最大的非边界α
                    i = indexNonBoundViolateKKT_[violateKKT_[indexNonBoundViolateKKT_].argmax()]
                else:
                    # 若不存在违反KKT条件的非边界α，则直接选取违反KKT条件程度最大的α
                    i = violateKKT_.argmax()
            else:
                # 保留较小的概率（15%）随机选取违反KKT条件的α
                i = choice(indexViolateKKT_)
            """选择αj"""
            j = choice(indexViolateKKT_)  # 随机选择另一个违反KKT条件的αj
            while (X__[i]==X__[j]).all():
                j = choice(range(N))      # 所选样本X__[i]、X__[j]完全相同，重新选择αj
            """优化"""
            print(f'第{t}次SMO迭代，选择i = {i}, j = {j}')
            ζ = α_[i] + α_[j]      # αi、αj旧值之和
            αiOld, αjOld = α_[i], α_[j]  # 记录αi、αj的旧值
            Kii = K__[i, i]        # 从核矩阵读取核函数值
            Kjj = K__[j, j]        # 从核矩阵读取核函数值
            Kij = K__[i, j]        # 从核矩阵读取核函数值
            η = Kii + Kjj - 2*Kij  # ||φ(xi)-φ(xj)||**2>=0
            L, H = max(0, ζ - C), min(C, ζ)  # 确定αj的下限L、上限H
            αj = αjOld + (wφx_[i] - wφx_[j])/η  # 未剪辑的αj
            if αj>H:    # 剪辑αj
                αj = H
            elif αj<L:
                αj = L
            else:
                pass
            αi = ζ - αj  # 未剪辑的αi
            if αi>C:     # 剪辑αi
                αi = C
            elif αi<0:
                αi = 0
            else:
                pass
            α_[j], α_[i] = αj, αi   # 更新αi、αj
            """更新优化目标函数值"""
            vi = wφx_[i] - αiOld*Kii - αjOld*Kij  # 系数vi
            vj = wφx_[j] - αiOld*Kij - αjOld*Kjj  # 系数vj
            minimizedObjectiveValue += 0.5*((αi**2 - αiOld**2)*Kii
                                    + (αj**2 - αjOld**2)*Kjj
                                    + 2*(αi*αj - αiOld*αjOld)*Kij
                                    + 2*vi*(αi - αiOld)
                                    + 2*vj*(αj - αjOld)
                                      )    # 更新目标函数值
            """更新偏离距离ρ"""
            if 0<α_[i]<C:
                ρ = wφx_[i] + (αi - αiOld)*Kii + (αj - αjOld)*Kij
            elif 0<α_[j]<C:
                ρ = wφx_[j] + (αi - αiOld)*Kij + (αj - αjOld)*Kjj
            else:
                ρi = wφx_[i] + (αi - αiOld)*Kii + (αj - αjOld)*Kij
                ρj = wφx_[j] + (αi - αiOld)*Kij + (αj - αjOld)*Kjj
                ρ = 0.5*(ρi + ρj)
            if isnan(ρ):
                raise ValueError('偏离距离ρ值为nan，排查错误！')
        else:
            print(f'达到最大迭代次数{self.maxIterations}!')

        """优化结束，计算偏离距离ρ"""
        indexSV_ = where(α_>0)[0]  # 数组索引：找出支持向量
        self.αSV_ = α_[indexSV_]   # 向量：支持向量对应的拉格朗日乘子α
        self.α_ = α_                                # N维向量：N个拉格朗日乘子
        self.supportVectors__ = X__[indexSV_]       # 矩阵：所有支持向量
        indexNonBound_ = where((0<α_) & (α_<C))[0]  # 数组索引：找出满足0<α<C的支持向量
        if len(indexNonBound_)>0:
            # 若存在满足0<α<C的支持向量，计算偏离距离ρ，取平均值
            ρ_ = [self.αSV_ @ K__[indexSV_, k]
                  for k in indexNonBound_]
            self.ρ = sum(ρ_)/len(ρ_)  # 取偏离距离ρ平均值
        else:
            self.ρ = ρ
            print('不存在满足0<α<C的α，取最后一次迭代得到的偏离距离ρ')
        """计算权重向量w_"""
        if self.kernel=='linear':
            # 若使用线性核函数，计算权重向量
            self.w_ = self.αSV_ @ self.supportVectors__
        return self

    def predict(self, X__: ndarray):
        """测试"""
        assert type(X__)==ndarray and X__.ndim==2, '输入测试样本矩阵X__应为2维ndarray'
        assert X__.shape[1]==self.M, f'输入测试样本维数应等于训练样本维数{self.M}'
        y_ = where(self.decisionFunction(X__)>=0, 1, -1)  # 正常 +1 / 异常 -1
        return y_

    def decisionFunction(self, X__) -> ndarray:
        """计算决策函数值"""
        assert type(X__)==ndarray and X__.ndim==2, '输入样本矩阵X__应为2维ndarray'
        assert X__.shape[1]==self.M, f'输入样本维数应等于训练样本维数{self.M}'
        f_ = self.kernelFunction(X__, self.supportVectors__) @ self.αSV_ - self.ρ
        return f_

    def abnormalRate(self, X__: ndarray)-> float:
        """计算测试异常率"""
        正常样本数 = sum(self.predict(X__)==1)
        测试样本总数 = len(X__)
        异常率 = (测试样本总数 - 正常样本数)/测试样本总数
        return 异常率

    def plot2D(self, X__):
        """作图：对于特征为2维的样本，查看决策超平面、支持向量"""
        if self.M!=2:
            print('当输入特征向量的维数为2，方可调用作图')
            return
        import numpy as np
        fig = plt.figure(figsize=[6, 6])
        ax = fig.add_subplot(111)
        ax.set_title(f'{self.kernel.capitalize().upper()} kernel function; $ν$ = {self.ν:g}')  # 图标题：核函数名称、超参数ν
        ax.plot([0], [0], 'ks', markersize=10)  # 原点
        ax.plot(X__[:, 0], X__[:, 1], 'ro')     # 样本点
        ax.plot(self.supportVectors__[:, 0],
                self.supportVectors__[:, 1],
                'ok',
                markersize=15,
                markerfacecolor='none')  # 支持向量点
        x0range, x1range = X__[:, 0].ptp(), X__[:, 1].ptp()
        x0_ = np.linspace(X__[:, 0].min() - 0.2*x0range, X__[:, 0].max() + 0.2*x0range, 100)  # x0方向的网格刻度
        x1_ = np.linspace(X__[:, 1].min() - 0.2*x1range, X__[:, 1].max() + 0.2*x1range, 100)  # x1方向的网格刻度
        grid__ = [[self.decisionFunction(array([[x0, x1]]))[0]
                   for x0 in x0_]
                   for x1 in x1_]   # 网格点上的决策函数值
        ax.contour(x0_, x1_, grid__,
                   levels=[0],
                   linestyles='--',
                   colors='k')    # 决策函数值等于0的等高线
        ax.legend(['Origin', 'Samples', 'Support vectors'])
        ax.set_xlabel('$x_{0}$')
        ax.set_ylabel('$x_{1}$')

    def plotMinimizedObjectiveFunctionValues(self):
        """作图：训练迭代的最小化目标函数值"""
        import numpy as np
        values_ = np.array(self.minimizedObjectiveValues_)
        fig = plt.figure(figsize=[6, 6])
        ax = fig.add_subplot(111)
        ax.plot(range(1, len(values_) + 1), values_, 'r-')                 # 历次迭代的最小化目标函数值
        ax.plot(range(1, len(values_)), values_[:-1] - values_[1:], 'b-')  # 目标函数值相比上一次迭代的减小量
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Minimized objective function')
        ax.legend(['Objective function value', 'Decrement of objective function value'])


if __name__=='__main__':
    # 设置随机种子
    from numpy.random import seed; seed(0)
    from random import seed; seed(0)

    # 使用sklearn生成样本集
    from sklearn.datasets import make_blobs
    X__, _ = make_blobs(n_features=2,       # 特征数量
                        n_samples=100,      # 样本数量
                        centers=1,          # 类别数
                        cluster_std=[0.5],  # 各类别的样本分布标准差
                        )

    # 将样本拆分成训练集、测试集
    from sklearn.model_selection import train_test_split
    Xtrain__, Xtest__ = train_test_split(X__, random_state=0)

    # 实例化单类支持向量机模型
    model = OneClassSupportVectorMachine(
        ν=0.1,              # 超参数：训练样本集当中异常样本所占比例的上限
        kernel='linear',      # 核函数：可选 线性核'linear'/高斯核'RBF'/多项式核'poly'
        maxIterations=20000,  # 最大迭代次数
        tol=1e-4,  # 迭代停止的收敛精度（指SMO求解算法当中最大违反KKT条件的程度）
        d=2,       # 超参数：多项式核函数的指数
        γ=1.,      # 超参数：高斯核函数、多项式核函数的参数
        r=1.,      # 超参数：多项式核函数的参数
        )
    model.fit(Xtrain__)     # 训练
    model.plotMinimizedObjectiveFunctionValues()  # 作图：历次迭代的最小化目标函数值
    model.plot2D(Xtrain__)  # 作图：对于特征为2维的样本，查看决策超平面、支持向量
    if model.kernel=='linear':
        print(f'权重向量w_ = {model.w_}')
    print(f'偏离距离ρ = {model.ρ}')
    print(f'训练集异常率：{model.abnormalRate(Xtrain__):.3f}')
    print(f'测试集异常率：{model.abnormalRate(Xtest__):.3f}')

    # 对比sklearn的单类支持向量机
    print('\n使用相同的超参数、训练集和测试集，对比sklearn的单类支持向量机')
    from sklearn.svm import OneClassSVM
    import numpy as np
    modelSK = OneClassSVM(
        nu=model.ν,                    # 超参数：训练样本集当中异常样本所占比例的上限
        kernel=model.kernel,           # 核函数
        max_iter=model.maxIterations,  # 最大迭代次数
        tol=model.tol,   # 收敛精度
        degree=model.d,  # 超参数: 多项式核函数的指数
        gamma=model.γ,   # 超参数: 高斯核函数参数/多项式核函数参数
        coef0=model.r,   # 超参数: 多项式核函数参数
        )
    modelSK.fit(Xtrain__)  # 训练
    if model.kernel=='linear':
        print(f'sklearn的权重向量w_ = {modelSK.coef_[0] * 1/(len(Xtrain__)*model.ν)}')  # 权重向量w_和偏离距离ρ同乘一个非零常数，仍表示同一个超平面
    print(f'sklearn的偏离距离ρ = {modelSK.offset_[0]* 1/(len(Xtrain__)*model.ν)}')
    print(f'sklearn的训练集异常率：{(modelSK.predict(Xtrain__)<0).sum() / len(Xtrain__):.3f}')
    print(f'sklearn的测试集异常率：{(modelSK.predict(Xtest__)<0).sum() / len(Xtest__):.3f}')
