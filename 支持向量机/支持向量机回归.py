from random import choice, uniform

from numpy import array, zeros, ones, ndarray, where, isnan, inf, intersect1d, block
import matplotlib.pyplot as plt

from 梯度下降 import GradientDescent
from 核函数 import LinearKernel, RBFKernel, PolyKernel


class SupportVectorMachineRegression:
    """支持向量回归"""
    def __init__(self,
            C: float = 1.,    # 超参数：惩罚参数
            ε: float = 1.,    # 超参数：间隔带宽度为2ε
            kernel: str = 'linear',      # 核函数：可选 线性核'linear'/高斯核'RBF'/多项式核'poly'
            solver: str = 'SMO',         # 求解算法：可选 序列最小优化算法'SMO'/梯度下降法'Pegasos'
            maxIterations: int = 10000,  # 最大迭代次数
            tol: float = 1e-3,  # 收敛精度（指SMO求解算法当中最大违反KKT条件的程度）
            LR: float = 1.,     # 超参数：全局学习率（用于Pegasos求解算法）
            d: int = 2,         # 超参数：多项式核函数的指数
            γ: float = 1.,      # 超参数：高斯核函数、多项式核函数的参数
            r: float = 1.,      # 超参数：多项式核函数的参数
            ):
        assert C>0, '惩罚参数C应大于0'
        assert ε>0, 'ε间隔带半宽度ε应大于0'
        assert type(maxIterations)==int and maxIterations>0, '最大迭代次数maxIterations应为正整数'
        assert tol>0, '收敛精度tol应大于0'
        assert LR>0, '全局学习率LR大于0'
        assert type(d)==int and d>=1, '多项式核函数的指数d应为正整数'
        assert γ>0, '高斯核函数、多项式核函数的参数γ应为正数'
        self.C = C  # 超参数：惩罚参数
        self.ε = ε  # 超参数：间隔带宽度为2ε
        self.kernel = kernel.lower()        # 核函数：可选 线性核'linear'/高斯核'RBF'/多项式核'poly'
        self.solver = solver.lower()        # 求解算法：可选 序列最小优化算法'SMO'/梯度下降法'Pegasos'
        self.maxIterations = maxIterations  # 最大迭代次数
        self.tol = tol       # 收敛精度（指SMO求解算法当中最大违反KKT条件的程度）
        self.LR = LR         # 超参数：全局学习率（用于Pegasos求解算法）
        self.d = d           # 超参数：多项式核函数的指数
        self.γ = γ           # 超参数：高斯核函数、多项式核函数的参数
        self.r = r           # 超参数：多项式核函数的参数
        self.M = None        # 输入特征向量的维数
        self.w_ = None       # M维向量：权重向量
        self.b = None        # 偏置
        self.α_ = None       # N维向量：N个训练样本的拉格朗日乘子
        self.αhat_ = None    # N维向量：N个训练样本的拉格朗日乘子
        self.ΔαSV_ = None    # N维向量：所有支持向量对应的αhat-α值
        self.supportVectors__ = None  # 矩阵：所有支持向量
        self.αSV_ = None     # 向量：所有支持向量对应的拉格朗日乘子α
        self.ySV_ = None     # 向量：所有支持向量对应的标签
        self.minimizedObjectiveValues_ = None  # 列表：历次迭代的最小化目标函数值（对于Pegasos算法，指损失函数值；对于SMO求解算法，指对偶问题的最小化目标函数值）
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
        """选择求解算法"""
        if self.solver=='smo':
            print('使用SMO算法求解')
        elif self.solver=='pegasos':
            assert self.kernel=='linear', "在使用线性核函数'linear'的前提下，方可调用Pegasos（梯度下降）求解算法"
            print('使用Pegasos算法求解')
        else:
            raise ValueError(f"未定义求解算法'{self.solver}'")

    def fit(self, X__: ndarray, y_: ndarray):
        """训练：使用SMO算法或Pegasos算法进行训练"""
        assert type(X__)==ndarray and X__.ndim==2, '输入训练样本矩阵X__应为2维ndarray'
        assert type(y_)==ndarray  and y_.ndim==1, '输入训练标签y_应为1维ndarray'
        assert len(X__)==len(y_), '输入训练样本数量应等于标签数量'
        self.M = X__.shape[1]   # 输入特征向量的维数
        if self.solver=='smo':  # 使用SMO算法训练
            self.SMO(X__, y_)
        elif self.solver=='pegasos':  # 使用Pegasos算法训练
            self.Pegasos(X__, y_)
        return self

    def Pegasos(self, X__: ndarray, y_: ndarray):
        """使用Pegasos（Primal estimated sub-gradient solver）算法，即梯度下降法，求解原始优化问题，最小化损失函数"""
        C = self.C        # 读取：惩罚参数
        ε = self.ε        # 读取：ε-间隔带半宽度
        N, M = X__.shape  # 训练样本数量N、输入特征向量的维数M
        w_ = ones(M)      # M维向量：初始化权重向量
        b = array([0.])   # 初始化偏置（由于偏置b需要载入梯度下降优化器，故先定义为1维ndarray，优化结束后再将偏置b提取为浮点数float）
        optimizer_for_w_ = GradientDescent(w_, method='Adam', LR=self.LR)  # 实例化w_的梯度下降优化器，代入全局学习率LR，选择Adam学习率调整策略
        optimizer_for_b  = GradientDescent(b,  method='Adam', LR=self.LR)  # 实例化b的梯度下降优化器，代入全局学习率LR，选择Adam学习率调整策略
        minLoss = inf  # 初始化最小损失函数值
        self.minimizedObjectiveValues_ = losses_ = []  # 列表：记录每一次迭代的损失函数值
        ξ_ = zeros(N)  # N维向量：初始化N个松弛变量
        for t in range(1, self.maxIterations+1):
            Δ_ = X__ @ w_ + b - y_   # N维向量：N个残差；残差 = 预测值 - 真实值
            ξ_[Δ_>ε] = Δ_[Δ_>ε] - ε  # N维向量：N个松弛变量
            ξ_[(-ε<=Δ_) & (Δ_<=ε)] = 0.
            ξ_[Δ_<-ε] = -Δ_[Δ_<-ε] - ε
            loss = 0.5*w_ @ w_ + C*sum(ξ_)  # 损失函数值
            gradw_ = w_ + C*(X__[Δ_>ε].sum(axis=0) - X__[Δ_<-ε].sum(axis=0))  # M维向量：损失函数对权重向量w_的梯度
            gradb = C*((Δ_>ε).sum() - (Δ_<-ε).sum())                          # 损失函数对偏置b的梯度
            losses_.append(loss)                      # 记录损失函数值
            if loss<minLoss:
                minLoss = loss         # 记录历史最小损失函数ε
                wOptimal_ = w_.copy()  # 记录历史最优权重向量w_
                bOptimal = b.copy()    # 记录历史最优偏置b
            print(f'第{t}次迭代，损失函数值{loss:.5g}')
            # 梯度下降，更新权重向量w_和偏置b
            optimizer_for_w_.update(gradw_)  # 代入梯度至优化器，更新权重向量w_
            optimizer_for_b.update(gradb)  # 代入梯度至优化器，更新偏置b
        else:
            print(f'达到最大迭代次数{self.maxIterations}')
        
        print(f'\t最后一次迭代的损失函数值{loss}')
        print(f'\t历次迭代的最小损失函数值{minLoss}')
        self.w_ = wOptimal_.copy()       # 以历史最优权重向量作为训练结果
        self.b = bOptimal.copy().item()  # 以历史最优偏置作为训练结果

    def SMO(self, X__: ndarray, y_: ndarray):
        """使用SMO（Sequential Minimal Optimization）算法求解对偶优化问题，最小化目标函数"""
        C = self.C      # 读取：惩罚参数
        ε = self.ε      # 读取：ε间隔带半宽度
        N = len(X__)    # 训练样本数量
        c_ = block([y_ - ε, -(y_ + ε)])      # 2N维向量：引入向量c_
        z_ = block([ones(N), -ones(N)])      # 2N维向量：引入向量z_
        K__ = self.kernelFunction(X__, X__)  # N×N矩阵：核函数矩阵
        K__ = block([[K__, K__],
                     [K__, K__]])  # 2N×2N矩阵：扩展的核函数矩阵
        λ_ = zeros(2*N)  # 2N维向量：初始化2N个拉格朗日乘子，即αhat_和α_
        b = 0            # 初始化偏置
        self.minimizedObjectiveValues_ = []  # 列表：记录历次迭代的目标函数值
        minimizedObjectiveValue = 0.5*λ_ @ (z_*K__*z_.reshape(-1, 1)) @ λ_ - λ_ @ c_   # 初始化：目标函数值
        for t in range(1, self.maxIterations + 1):
            indexSV_ = where(λ_>0)[0]                   # 数组索引：满足λ>0的支持向量
            indexNonBound_ = where((0<λ_) & (λ_<C))[0]  # 数组索引：满足0<λ<C的支持向量
            self.minimizedObjectiveValues_.append(minimizedObjectiveValue)  # 记录当前目标函数值
            """检验所有2N个λ是否满足KKT条件，并计算各λ违反KKT条件的程度"""
            g_ = (λ_[indexSV_]*z_[indexSV_]) @ K__[indexSV_, :] + b  # 2N维向量：g(xi)值，i=1~N
            zg_ = z_*g_                                   # 2N维向量：zi*g(xi), i = 1~N
            violateKKT_ = abs(c_ - zg_)                   # 2N维向量：开始计算“违反KKT条件的程度”
            violateKKT_[(λ_==0) & (zg_>=c_)] = 0.         # 2N维向量：KKT条件 λ=0 ←→ z*g(x)≥c
            violateKKT_[(0<λ_) & (λ_<C) & (zg_==c_)] = 0. # 2N维向量：KKT条件 0<λ<C ←→ z*g(x)=c
            violateKKT_[(λ_==C) & (zg_<=c_)] = 0.         # 2N维向量：KKT条件 λ=C ←→ z*g(x)≤c
            if violateKKT_.max()<self.tol:
                print(f'第{t}次SMO迭代，最大违反KKT条件程度达到收敛精度{self.tol}，停止迭代!')
                break
            """选择λi"""
            indexViolateKKT_ = where(violateKKT_>0)[0]    # 数组索引：找出违反KKT条件的λ
            indexNonBoundViolateKKT_ = intersect1d(indexViolateKKT_, indexNonBound_)  # 数组索引：找出违反KKT条件的非边界λ
            if uniform(0, 1)<0.85:
                # 有较大的概率（85%）选取违反KKT条件程度最大的λ进行下一步优化，若有非边界λ，首选非边界λ
                if len(indexNonBoundViolateKKT_)>0:
                    # 若存在违反KKT条件的非边界λ，则选取违反KKT条件程度最大的非边界λ
                    i = indexNonBoundViolateKKT_[violateKKT_[indexNonBoundViolateKKT_].argmax()]
                else:
                    # 若不存在违反KKT条件的非边界λ，则直接选取违反KKT条件程度最大的λ
                    i = violateKKT_.argmax()
            else:
                # 保留较小的概率（15%）随机选取违反KKT条件的λ
                i = choice(indexViolateKKT_)
            """选择λj"""
            j = choice(indexViolateKKT_)  # 随机选择另一个违反KKT条件的λ
            ii = i-N if i>N-1 else i      # 因为第i个λ和第i+N个λ对应的是同一个样本
            jj = j-N if j>N-1 else j      # 因为第j个λ和第j+N个λ对应的是同一个样本
            while (X__[ii]==X__[jj]).all():
                j = choice(range(N))      # 所选样本X__[i]、X__[j]完全相同，重新选择λj
                jj = j-N if j>N-1 else j
            """优化λi、λj"""
            print(f'第{t}次SMO迭代，选择i = {i}, j = {j}')
            λiOld, λjOld = λ_[i], λ_[j]   # 记录λi, λj的旧值
            if z_[i]!=z_[j]:              # 确定λj的下限L、上限H
                L, H = max(0, λjOld - λiOld), min(C, C + λjOld - λiOld)
            else:
                L, H = max(0, λjOld + λiOld - C), min(C, λjOld + λiOld)
            Kii = K__[i, i]  # 从核矩阵读取核函数值
            Kjj = K__[j, j]  # 从核矩阵读取核函数值
            Kij = K__[i, j]  # 从核矩阵读取核函数值
            Ei = g_[i] - c_[i]*z_[i]   # 误差值Ei
            Ej = g_[j] - c_[j]*z_[j]   # 误差值Ej
            η = Kii + Kjj - 2*Kij      # ||φ(xi)-φ(xj)||**2 >= 0
            λj = λjOld + z_[j]*(Ei - Ej)/η  # 未剪辑的λj
            if λj>H:     # 剪辑λj
                λj = H
            elif λj<L:
                λj = L
            else:
                pass
            λi = λiOld + z_[i]*z_[j]*(λjOld - λj) # 未剪辑的λi
            if λi>C:     # 剪辑λi
                λi = C
            elif λi<0:
                λi = 0
            else:
                pass
            λ_[i], λ_[j] = λi, λj  # 更新λi、λj
            """更新目标函数值"""
            vi = g_[i] - λiOld*z_[i]*Kii - λjOld*z_[j]*Kij - b  # 记号vi
            vj = g_[j] - λiOld*z_[i]*Kij - λjOld*z_[j]*Kjj - b  # 记号vj
            minimizedObjectiveValue += (0.5*(λi**2 - λiOld**2)*Kii
                                     + 0.5*(λj**2 - λjOld**2)*Kjj
                                     + (λi*λj - λiOld*λjOld)*Kij*z_[i]*z_[j]
                                     + vi*z_[i]*(λi - λiOld)
                                     + vj*z_[j]*(λj - λjOld)
                                      ) - ((λi - λiOld)*c_[i] + (λj - λjOld)*c_[j])  # 更新目标函数值
            """更新偏置b"""
            if 0<λi<C:
                b =  -Ei - z_[i]*Kii*(λ_[i] - λiOld) - z_[j]*Kij*(λ_[j] - λjOld) + b
            elif 0<λj<C:
                b =  -Ej - z_[i]*Kij*(λ_[i] - λiOld) - z_[j]*Kjj*(λ_[j] - λjOld) + b
            else:
                bi = -Ei - z_[i]*Kii*(λ_[i] - λiOld) - z_[j]*Kij*(λ_[j] - λjOld) + b
                bj = -Ej - z_[i]*Kij*(λ_[i] - λiOld) - z_[j]*Kjj*(λ_[j] - λjOld) + b
                b = (bi + bj)/2
            if isnan(b):
                raise ValueError('偏置b值为nan，排查错误！')
        else:
            print(f'达到最大迭代次数{self.maxIterations}，停止迭代!')
        """优化结束，计算偏置b"""
        self.αhat_ = λ_[:N]                                    # N维向量：N个拉格朗日乘子αhat
        self.α_ = λ_[N:]                                       # N维向量：N个拉格朗日乘子α
        indexSV_ = where(self.αhat_!=self.α_)[0]               # 数值索引：所有支持向量
        self.ΔαSV_ = self.αhat_[indexSV_] - self.α_[indexSV_]  # 向量：支持向量对应的αhat-α值
        self.supportVectors__ = X__[indexSV_]                  # 矩阵：所有支持向量
        self.ySV_ = y_[indexSV_]                               # 向量：所有支持向量对应的标签
        indexNonBound_ = where((0<self.α_) & (self.α_<C))[0]   # 数组索引：满足0<α<C的α
        if len(indexNonBound_)>0:
            # 若存在满足0<α<C的α，计算偏置b，取平均值
            b_ = [y_[k] + ε - self.ΔαSV_ @ K__[k][indexSV_] for k in indexNonBound_]
            self.b = sum(b_)/len(b_)  # 取偏置b平均值
        else:
            self.b = b
            print('不存在满足0<α<C的α，取最后一次迭代得到的偏置b')
        """计算权重向量w_"""
        if self.kernel=='linear':
            # 若使用线性核函数，计算权重向量
            self.w_ = self.ΔαSV_ @ self.supportVectors__

    def predict(self, X__: ndarray) -> ndarray:
        """测试"""
        assert type(X__)==ndarray and X__.ndim==2, '输入测试样本矩阵X__应为2维ndarray'
        assert X__.shape[1]==self.M, f'输入测试样本维数应等于训练样本维数{self.M}'
        if self.solver=='smo':
            # 若使用SMO算法训练
            y_ = self.ΔαSV_ @ self.kernelFunction(self.supportVectors__, X__) + self.b
        elif self.solver=='pegasos':
            # 若使用Pegasos算法训练
            y_ = X__ @ self.w_ + self.b
        return y_

    def MAE(self, X__: ndarray, y_: ndarray) -> float:
        """计算回归测试的平均绝对误差MAE"""
        return abs(self.predict(X__) - y_).mean()

    def plot1D(self, X__, y_):
        """作图：对于特征为1维的样本，查看间隔带、支持向量"""
        if self.M!=1:
            print('当输入特征向量的维数为1，方可调用作图')
            return
        import numpy as np
        fig = plt.figure(figsize=[6, 6])
        ax = fig.add_subplot(111)
        ax.set_title(f'{self.kernel.capitalize()} kernel function; $C$ = {self.C:g}; $ε$ = {self.ε:g};\n'
                     f'Solved by {self.solver.upper()} algorithm')  # 图标题：核函数名称、惩罚参数C、间隔带半宽度ε、求解算法
        xMin, xMax = X__.min(), X__.max()       # 输入量下界、上界
        x_ = np.linspace(xMin, xMax, 200)       # 输入量
        yPredict_ = self.predict(x_.reshape(-1, 1))  # 回归值
        h = []  # 图句柄
        h += ax.plot(x_, yPredict_, '-b')  # 回归值
        h += ax.plot(X__, y_, 'or')        # 真实值
        h += ax.plot(x_, yPredict_ + self.ε, '--k')  # 间隔带上边界
        ax.plot(x_, yPredict_ - self.ε, '--k')       # 间隔带下边界
        if self.solver=='smo':
            # 若使用SMO算法求解，则标示支持向量点（使用Pegasos算法不能辨识支持向量）
            h += ax.plot(self.supportVectors__, self.ySV_, 'ok',
                    markersize=15,
                    markerfacecolor='none')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.legend(h,
                  ['Regression value', 'True value', 'Margin'] + (['Support vectors'] if self.solver=='smo' else [])
                  )

    def plotMinimizedObjectiveFunctionValues(self):
        """作图：训练迭代的最小化目标函数值"""
        import numpy as np
        values_ = np.array(self.minimizedObjectiveValues_)
        fig = plt.figure(figsize=[6, 6])
        ax = fig.add_subplot(111)
        ax.plot(range(1, len(values_) + 1), values_, 'r-')                 # 历次迭代的最小化目标函数值
        ax.plot(range(1, len(values_)), values_[:-1] - values_[1:], 'b-')  # 目标函数值相比上一次迭代的减小量
        ax.set_xlabel('Iteration')
        if self.solver=='pegasos':
            ax.legend(['Loss function value', 'Decrement of loss function value'])
            ax.set_ylabel('Loss function')
        elif self.solver=='smo':
            ax.legend(['Objective function value', 'Decrement of objective function value'])
            ax.set_ylabel('Minimized objective function')


if __name__=='__main__':
    # 设置随机种子
    from numpy.random import seed; seed(0)
    from random import seed; seed(0)

    # 生成回归样本集
    import numpy as np
    x_ = np.arange(0, 10, 0.1)
    X__ = x_.reshape(-1, 1)        # 矩阵：样本集
    y_ = np.sin(2*x_) + 0.1*x_**2  # 向量：样本标签

    # 将样本拆分成训练集、测试集
    from sklearn.model_selection import train_test_split
    Xtrain__, Xtest__, ytrain_, ytest_ = train_test_split(X__, y_, random_state=0)

    # 实例化支持向量机回归模型
    model = SupportVectorMachineRegression(
        C=1.,    # 超参数：惩罚参数
        ε=1.,    # 超参数：间隔带宽度为2ε，ε值不能太大
        kernel='linear',      # 核函数：可选 线性核'linear'/高斯核'RBF'/多项式核'poly'
        solver='SMO',         # 求解算法：可选 序列最小优化算法'SMO'/梯度下降法'Pegasos'
        maxIterations=50000,  # 最大迭代次数
        tol=1e-4,  # 收敛精度（指SMO求解算法当中最大违反KKT条件的程度）
        LR=0.01,   # 超参数：全局学习率（用于Pegasos求解算法）
        d=2,       # 超参数：多项式核函数的指数
        γ=1.,      # 超参数：高斯核函数、多项式核函数的参数
        r=1.,      # 超参数：多项式核函数的参数
        )
    model.fit(Xtrain__, ytrain_)  # 训练
    model.plot1D(Xtrain__, ytrain_)               # 作图：查看间隔带、支持向量
    model.plotMinimizedObjectiveFunctionValues()  # 作图：历次迭代的最小化目标函数值
    if model.kernel=='linear':
        print(f'权重向量w_ = {model.w_}')
    print(f'偏置b = {model.b}')
    print(f'训练集平均绝对误差：{model.MAE(Xtrain__, ytrain_):.3f}')
    print(f'测试集平均绝对误差：{model.MAE(Xtest__, ytest_):.3f}')

    # 对比sklearn的支持向量机回归
    print('\n使用同样的超参数、训练集和测试集，对比sklearn的支持向量机回归')
    from sklearn.svm import SVR
    modelSK = SVR(
        C=model.C,            # 超参数: 惩罚参数
        epsilon=model.ε,      # 超参数：间隔带宽度为2ε
        kernel=model.kernel,  # 核函数
        max_iter=model.maxIterations,  # 最大迭代次数
        tol=model.tol,        # 收敛精度
        degree=model.d,       # 超参数：多项式核函数的指数
        gamma=model.γ,        # 超参数：高斯核函数参数/多项式核函数参数
        coef0=model.r,        # 超参数：多项式核函数参数
        )
    modelSK.fit(Xtrain__, ytrain_)
    if modelSK.kernel=='linear':
        print(f'sklearn的权重向量w_ = {modelSK.coef_[0]}')
    print(f'sklearn的偏置b = {modelSK.intercept_[0]}')
    print(f'sklearn的训练集平均绝对误差：{abs(modelSK.predict(Xtrain__) - ytrain_).mean():.3f}')
    print(f'sklearn的测试集平均绝对误差：{abs(modelSK.predict(Xtest__) - ytest_).mean():.3f}')
