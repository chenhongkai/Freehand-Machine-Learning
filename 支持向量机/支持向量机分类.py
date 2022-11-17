from random import choice, random

from numpy import array, zeros, ones, ndarray, where, isnan, inf, intersect1d, maximum
import matplotlib.pyplot as plt

from 梯度下降 import GradientDescent
from 核函数 import LinearKernel, RBFKernel, PolyKernel


class SupprotVectorMachineClassification:
    """支持向量机分类"""
    def __init__(self,
            C: float = 1.,               # 超参数：惩罚参数
            kernel: str = 'linear',      # 核函数：可选 线性核'linear'/高斯核'rbf'/多项式核'poly'
            solver: str = 'SMO',         # 求解算法：可选 序列最小优化算法'SMO'/梯度下降法'Pegasos'
            maxIterations: int = 50000,  # 最大迭代次数
            tol: float = 1e-3,  # 迭代停止的收敛精度（指SMO求解算法当中最大违反KKT条件的程度）
            LR: float = 0.1,    # 超参数：全局学习率（用于Pegasos求解算法）
            d: int = 2,         # 超参数：多项式核函数的指数
            γ: float = 1.,      # 超参数：高斯核函数、多项式核函数的参数
            r: float = 1.,      # 超参数：多项式核函数的参数
            ):
        assert C>0, '惩罚参数C应大于0'
        assert type(maxIterations)==int and maxIterations>0, '最大迭代次数maxIterations应为正整数'
        assert tol>0, '收敛精度tol应大于0'
        assert LR>0, '全局学习率LR应大于0'
        assert type(d)==int and d>=1, '多项式核函数的指数d应为正整数'
        assert γ>0, '高斯核函数、多项式核函数的参数γ应大于0'
        self.C = C                    # 惩罚参数
        self.kernel = kernel.lower()  # 核函数：可选 线性核'linear'/高斯核'rbf'/多项式核'poly'
        self.solver = solver.lower()  # 求解算法：可选 序列最小优化'SMO'/梯度下降'Pegasos'
        self.maxIterations = maxIterations  # 最大迭代次数
        self.tol = tol                # 迭代停止的收敛精度（指SMO求解算法当中最大违反KKT条件的程度）
        self.LR = LR    # 超参数：全局学习率（用于Pegasos求解算法）
        self.d = d      # 超参数：多项式核函数的指数
        self.γ = γ      # 超参数：高斯核函数、多项式核函数的参数
        self.r = r      # 超参数：多项式核函数的参数
        self.M = None   # 输入特征向量的维数
        self.w_ = None  # M维向量：权重向量
        self.b = None   # 偏置
        self.α_ = None  # N维向量：所有N个训练样本的拉格朗日乘子
        self.supportVectors__ = None # 矩阵：所有支持向量
        self.αSV_ = None     # 向量：所有支持向量对应的拉格朗日乘子α
        self.ySV_ = None     # 向量：所有支持向量对应的标签
        self.losses_ = None  # 列表：每次迭代的损失函数值（对于SMO求解算法，指对偶问题的最小化目标函数值）
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
        assert set(y_)=={-1, 1},  '输入训练标签取值应为-1或+1'
        self.M = X__.shape[1]    # 输入特征向量的维数
        if self.solver=='smo':   # 使用SMO算法训练
            self.SMO(X__, y_)
        elif self.solver=='pegasos':  # 使用Pegasos算法训练
            self.Pegasos(X__, y_)
        return self

    def Pegasos(self, X__, y_):
        """使用Pegasos（Primal estimated sub-gradient solver）算法，即梯度下降法，求解原始优化问题，最小化损失函数"""
        C = self.C        # 读取：惩罚参数
        N, M = X__.shape  # 训练样本数量N、输入特征向量的维数M
        w_ = ones(M)      # M维向量：初始化权重向量
        b = array([0.])   # 初始化偏置（由于偏置b需要载入梯度下降优化器，故先定义为1维ndarray，优化结束后再将偏置b提取为浮点数float）
        optimizer_for_w_ = GradientDescent(w_, method='Adam', LR=self.LR)  # 实例化w_的梯度下降优化器，代入全局学习率LR，选择Adam学习率调整策略
        optimizer_for_b  = GradientDescent(b, method='Adam', LR=self.LR)   # 实例化b的梯度下降优化器，代入全局学习率LR，选择Adam学习率调整策略
        minLoss = inf     # 初始化最小损失函数值
        self.losses_ = losses_ = []  # 列表：记录每一次迭代的损失函数值
        for t in range(1, self.maxIterations + 1):
            ξ_ = maximum(1 - y_*(X__ @ w_ + b), 0)  # N维向量：N个松弛变量
            I_ = ξ_>0                        # N维向量：N个0/1指示量
            loss = 0.5*w_ @ w_ + C*ξ_.sum()  # 损失函数值
            gradw_ = w_ - C*(I_*y_ @ X__)    # M维向量：损失函数对权重向量w_的梯度
            gradb = -C*(I_ @ y_)             # 损失函数对偏置b的梯度
            losses_.append(loss)             # 记录损失函数值
            if loss<minLoss:
                minLoss = loss         # 记录历史最小损失函数
                wOptimal_ = w_.copy()  # 记录历史最优权重向量w_
                bOptimal = b.copy()    # 记录历史最优偏置b
            print(f'第{t}次迭代，损失函数值{loss:.5g}')
            # 梯度下降，更新权重向量w_和偏置b
            optimizer_for_w_.update(gradw_)  # 代入梯度至优化器，更新权重向量w_
            optimizer_for_b.update(gradb)    # 代入梯度至优化器，更新偏置b
        else:
            print(f'达到最大迭代次数{self.maxIterations}')

        print(f'\t最后一次迭代的损失函数值{loss}')
        print(f'\t历次迭代的最小损失函数值{minLoss}')
        self.w_ = wOptimal_.copy()      # 以历史最优权重向量作为训练结果
        self.b = bOptimal.copy().item() # 以历史最优偏置作为训练结果

    def SMO(self, X__, y_):
        """使用SMO（Sequential Minimal Optimization）算法求解对偶问题，最小化目标函数"""
        C = self.C      # 读取：惩罚参数
        N = len(X__)    # 训练样本数量
        K__ = self.kernelFunction(X__, X__)  # N×N矩阵：核函数矩阵
        α_ = zeros(N)   # N维向量：初始化N个拉格朗日乘子
        b = 0.          # 初始化偏置
        self.losses_ = losses_ = []  # 列表：记录历次迭代的目标函数值
        loss = 0.5*α_ @ (y_*K__*y_.reshape(-1, 1)) @ α_ - α_.sum()  # 初始化优化的目标函数值
        for t in range(1, self.maxIterations + 1):
            indexSV_ = where(α_>0)[0]                   # 数组索引：满足α>0的支持向量
            indexNonBound_ = where((0<α_) & (α_<C))[0]  # 数组索引：满足0<α<C的支持向量，满足0<α<C的α称为非边界α（Non-bound）
            losses_.append(loss)                        # 记录当前目标函数值
            """检验所有N个样本是否满足KKT条件，并计算各样本违反KKT条件的程度"""
            g_ = (α_[indexSV_]*y_[indexSV_]) @ K__[indexSV_, :] + b  # N维向量：g(xi)值，i=1~N，李航《统计学习方法（第二版）》7.4节
            E_ = g_ - y_                                       # N维向量：E值，Ei=g(xi)-yi，i=1~N，李航《统计学习方法（第二版）》7.4节
            yg_ = y_*g_                                        # N维向量：yi*g(xi)，i=1~N
            violateKKT_ = abs(1 - yg_)                         # N维向量：开始计算“违反KKT条件的程度”
            violateKKT_[(α_==0) & (yg_>=1)] = 0.               # N维向量：KKT条件 αi=0   ←→ yi*g(xi)≥1，李航《统计学习方法（第二版）》7.111式
            violateKKT_[(0<α_) & (α_<C) & (yg_==1)] = 0.       # N维向量：KKT条件 0<αi<C ←→ yi*g(xi)=1，李航《统计学习方法（第二版）》7.112式
            violateKKT_[(α_==C) & (yg_<=1)] = 0.               # N维向量：KKT条件 αi=C   ←→ yi*g(xi)≤1，李航《统计学习方法（第二版）》7.113式
            indexViolateKKT_ = where(violateKKT_>self.tol)[0]  # 数组索引：找出违反KKT条件的α
            if len(indexViolateKKT_)==0:
                # 若不存在违反KKT条件的α，停止迭代
                print(f'第{t}次迭代，最大违反KKT条件的程度达到收敛精度{self.tol}，停止迭代!')
                break
            """选择αi"""
            indexNonBoundViolateKKT_ = intersect1d(indexViolateKKT_, indexNonBound_)  # 数组索引：找出违反KKT条件的非边界α
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
            # j = E_.argmin() if E_[i]>0 else E_.argmax()  # 经过试验发现：若按照“使|Ei-Ej|最大”这样的规则来选择j，则往往所需迭代次数极大，甚至陷入死循环。优化效率还不如随机选择j。
            while (X__[i]==X__[j]).all():
                j = choice(range(N))  # 所选样本X__[i]、X__[j]完全相同，重新选择αj
            """优化αi、αj"""
            print(f'第{t}次SMO迭代，所选i = {i}, j = {j}')
            αiOld, αjOld = α_[i], α_[j]  # 记录αi、αj的旧值
            if y_[i]!=y_[j]:             # 确定αj的下限L、上限H
                L, H = max(0, αjOld - αiOld), min(C, C + αjOld - αiOld)
            else:
                L, H = max(0, αjOld + αiOld - C), min(C, αjOld + αiOld)
            Kii = K__[i, i]        # 从核矩阵读取核函数值K(X__[i], X__[i])
            Kjj = K__[j, j]        # 从核矩阵读取核函数值K(X__[j], X__[j])
            Kij = K__[i, j]        # 从核矩阵读取核函数值K(X__[i], X__[j])
            η = Kii + Kjj - 2*Kij  # ||φ(xi)-φ(xj)||**2 >= 0
            αj = αjOld + y_[j]*(E_[i] - E_[j])/η  # 未剪辑的αj
            if αj>H:    # 剪辑αj
                αj = H
            elif αj<L:
                αj = L
            else:
                pass
            αi = αiOld + y_[i]*y_[j]*(αjOld - αj)  # 未剪辑的αi
            if αi>C:    # 剪辑αi
                αi = C
            elif αi<0:
                αi = 0
            else:
                pass
            α_[j], α_[i] = αj, αi   # 更新αi、αj
            """更新目标函数值"""
            vi = g_[i] - αiOld*y_[i]*Kii - αjOld*y_[j]*Kij - b  # 记号vi，李航《统计学习方法（第2版）》定理7.6的证明
            vj = g_[j] - αiOld*y_[i]*Kij - αjOld*y_[j]*Kjj - b  # 记号vj，李航《统计学习方法（第2版）》定理7.6的证明
            loss += (0.5*(αi**2 - αiOld**2)*Kii
                   + 0.5*(αj**2 - αjOld**2)*Kjj
                   + (αi*αj - αiOld*αjOld)*Kij*y_[i]*y_[j]
                   + vi*y_[i]*(αi - αiOld)
                   + vj*y_[j]*(αj - αjOld)
                    ) - (αi - αiOld + αj - αjOld)  # 更新目标函数值
            """更新偏置b"""
            if 0<α_[i]<C:
                b = -E_[i] - y_[i]*Kii*(α_[i] - αiOld) - y_[j]*Kij*(α_[j] - αjOld) + b  # 李航《统计学习方法（第2版）》式7.115
            elif 0<α_[j]<C:
                b = -E_[j] - y_[i]*Kij*(α_[i] - αiOld) - y_[j]*Kjj*(α_[j] - αjOld) + b
            else:
                bi = -E_[i] - y_[i]*Kii*(α_[i] - αiOld) - y_[j]*Kij*(α_[j] - αjOld) + b
                bj = -E_[j] - y_[i]*Kij*(α_[i] - αiOld) - y_[j]*Kjj*(α_[j] - αjOld) + b
                b = (bi + bj)/2
            if isnan(b):
                raise ValueError('偏置b值为nan，排查错误！')
        else:
            print(f'达到最大迭代次数{self.maxIterations}!')

        """优化结束，计算偏置b"""
        indexSV_ = where(α_>0)[0]              # 数组索引：支持向量
        self.αSV_ = α_[indexSV_]               # 向量：支持向量对应的拉格朗日乘子α
        self.ySV_ = y_[indexSV_]               # 向量：支持向量对应的标签
        self.supportVectors__ = X__[indexSV_]  # 矩阵：所有支持向量
        self.α_ = α_                           # N维向量：N个拉格朗日乘子
        indexNonBound_ = where((0<α_) & (α_<C))[0]  # 数组索引：满足0<α<C的支持向量
        if len(indexNonBound_)>0:
            # 若存在满足0<α<C的支持向量，计算偏置b，取平均值
            b_ = [(y_[k] - (self.αSV_*self.ySV_) @ K__[k, indexSV_]) for k in indexNonBound_]
            self.b = sum(b_)/len(b_)  # 取偏置b平均值
        else:
            self.b = b  # 取最后一次迭代的偏置b
        """计算权重向量w_"""
        if self.kernel=='linear':
            # 若使用线性核函数，计算权重向量
            self.w_ = (self.αSV_*self.ySV_) @ self.supportVectors__

    def predict(self, X__: ndarray) -> ndarray:
        """测试"""
        assert type(X__)==ndarray and X__.ndim==2, '输入测试样本矩阵X__应为2维ndarray'
        assert X__.shape[1]==self.M, f'输入测试样本维数应等于训练样本维数{self.M}'
        y_ = where(self.decisionFunction(X__)>=0, 1, -1)  # 判定类别
        return y_

    def decisionFunction(self, X__: ndarray) -> ndarray:
        """计算决策函数值 f = w_ @ x_ + b"""
        assert X__.ndim==2, '输入样本矩阵X__应为2维ndarray'
        assert X__.shape[1]==self.M, f'输入测试样本维数应等于训练样本维数{self.M}'
        if self.solver=='smo':
            # 若使用SMO算法进行训练
            f_ = (self.αSV_*self.ySV_) @ self.kernelFunction(self.supportVectors__, X__)  + self.b
        elif self.solver=='pegasos':
            # 若使用Pegasos算法进行训练
            f_ = X__ @ self.w_ + self.b
        return f_

    def accuracy(self, X__: ndarray, y_: ndarray) -> float:
        """计算测试正确率"""
        测试样本正确个数 = sum(self.predict(X__)==y_)
        测试样本总数 = len(y_)
        return 测试样本正确个数/测试样本总数

    def plot2D(self, X__: ndarray, y_: ndarray):
        """作图：对于特征为2维的样本，查看分类边界、Margin间隔、支持向量"""
        if self.M!=2:
            print('当输入特征向量的维数为2，方可调用作图')
            return
        import numpy as np
        fig = plt.figure(figsize=[6, 6])
        ax = fig.add_subplot(111)
        ax.set_title(f'{self.kernel.capitalize()} kernel function; $C$ = {self.C:g};\n'
                     f'Solved by {self.solver.upper()} algorithm')  # 图标题：核函数名称、惩罚参数C
        ax.plot(X__[y_==1][:, 0], X__[y_==1][:, 1], 'or')    # 正类点（红）
        ax.plot(X__[y_==-1][:, 0], X__[y_==-1][:, 1], 'xb')  # 反类点（蓝）
        if self.solver=='smo':
            # 若使用SMO算法求解，则标示支持向量点（使用Pegasos算法不能辨识支持向量）
            ax.plot(self.supportVectors__[:, 0], self.supportVectors__[:, 1], 'ok',
                    markersize=15,
                    markerfacecolor='none')
        x0range, x1range = X__[:, 0].ptp(), X__[:, 1].ptp()  # x0、x1两个特征的极差
        x0_ = np.linspace(X__[:, 0].min() - 0.2*x0range, X__[:, 0].max() + 0.2*x0range, 100)  # x0方向的网格刻度
        x1_ = np.linspace(X__[:, 1].min() - 0.2*x1range, X__[:, 1].max() + 0.2*x1range, 100)  # x1方向的网格刻度
        grid__ = [[self.decisionFunction(np.array([[x0, x1]]))[0]
                   for x0 in x0_]
                   for x1 in x1_]  # 网格点上的决策函数值
        ax.contour(x0_, x1_, grid__,
                   levels=[-1, 0, 1],
                   linestyles=['--', '-', '--'],
                   colors=['b', 'k', 'r'])  # 决策函数值等于-1、0、+1的三根等高线
        ax.set_xlabel('$x_{0}$')
        ax.set_ylabel('$x_{1}$')

    def plotLoss(self):
        """作图：训练迭代的损失函数值（SMO算法为最小化的目标函数值）"""
        import numpy as np
        losses_ = np.array(self.losses_)
        fig = plt.figure(figsize=[6, 6])
        ax = fig.add_subplot(111)
        ax.plot(range(1, len(losses_) + 1), losses_, 'r-')                 # 历次迭代的损失函数值
        ax.plot(range(1, len(losses_)), losses_[:-1] - losses_[1:], 'b-')  # 损失函数值相比上一次迭代的减小量
        ax.legend(['Loss', 'Loss decrement'])
        ax.set_ylabel('Loss function')
        ax.set_xlabel('Iteration')

if __name__=='__main__':
    # 设置随机种子
    from numpy.random import seed; seed(0)
    from random import seed; seed(0)

    # 使用sklearn生成二分类样本集
    from sklearn.datasets import make_blobs
    X__, y_ = make_blobs(n_features=2,    # 特征数量
                         n_samples=200,   # 样本数量
                         centers=2,       # 类别数
                         cluster_std=[1., 1.],  # 各类别的样本分布标准差
                         random_state=0)  # 随机种子
    y_[y_==0] = -1  # 将标签值0改成-1

    # 将样本拆分成训练集、测试集
    from sklearn.model_selection import train_test_split
    Xtrain__, Xtest__, ytrain_, ytest_ = train_test_split(X__, y_)

    # 实例化支持向量机分类模型
    model = SupprotVectorMachineClassification(
        C=1.,             # 超参数：惩罚参数
        kernel='linear',  # 核函数：可选 线性核'linear'/高斯核'RBF'/多项式核'poly'
        solver='SMO',     # 求解算法：可选 序列最小优化算法'SMO'/梯度下降法'Pegasos'
        maxIterations=50000,  # 最大迭代次数
        tol=1e-4,  # 迭代停止的收敛精度（指SMO求解算法当中最大违反KKT条件的程度）
        LR=0.01,   # 超参数：全局学习率（用于Pegasos求解算法）
        d=2,       # 超参数：多项式核函数的指数
        γ=1.,      # 超参数：高斯核函数、多项式核函数的参数
        r=1.,      # 超参数：多项式核函数的参数
        )
    model.fit(Xtrain__, ytrain_)     # 训练
    model.plotLoss()                 # 作图：历次迭代的损失函数值（SMO算法为最小化的目标函数值）
    model.plot2D(Xtrain__, ytrain_)  # 作图：查看分类边界、Margin间隔、支持向量
    if model.kernel=='linear':
        print(f'权重向量w_ = {model.w_}')
    print(f'偏置b = {model.b}')
    print(f'训练集正确率：{model.accuracy(Xtrain__, ytrain_):.3f}')
    print(f'测试集正确率：{model.accuracy(Xtest__, ytest_):.3f}')

    # 对比sklearn的支持向量机分类
    print('\n使用同样的超参数、训练集和测试集，对比sklearn的支持向量机分类')
    from sklearn.svm import SVC
    modelSK = SVC(
        C=model.C,            # 超参数：惩罚参数
        kernel=model.kernel,  # 核函数
        max_iter=model.maxIterations, # 最大迭代次数
        tol=model.tol,        # 收敛精度
        degree=model.d,       # 超参数: 多项式核函数的指数
        gamma=model.γ,        # 超参数: 高斯核函数参数/多项式核函数参数
        coef0=model.r,        # 超参数: 多项式核函数参数
        )
    modelSK.fit(Xtrain__, ytrain_)  # 训练
    if model.kernel=='linear':
        print(f'sklearn的权重向量w_ = {modelSK.coef_[0]}')
    print(f'sklearn的偏置b = {modelSK.intercept_[0]}')
    print(f'sklearn的训练集正确率：{modelSK.score(Xtrain__, ytrain_):.3f}')
    print(f'sklearn的测试集正确率：{modelSK.score(Xtest__, ytest_):.3f}')
