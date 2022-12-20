from numpy import ndarray, zeros, unique


class NaiveBayes:
    """朴素贝叶斯"""
    def __init__(self,
                 isSmoothing: bool = True,  # 是否使用拉普拉斯平滑
                 ):
        assert type(isSmoothing)==bool, 'isSmoothing应为布尔值'
        self.isSmoothing = isSmoothing  # 是否使用拉普拉斯平滑（Laplacian smoothing）
        if self.isSmoothing:
            print('使用拉普拉斯平滑')
        else:
            print('不使用平滑')
        self.M = None         # 输入特征向量的维数
        self.K = None         # 类别数
        self.classes_ = None  # K维向量：K个类别标签
        self.Py_ = None       # 字典：存储K个类别的先验概率
        self.values__ = None  # 字典：样本x_的各个离散值特征可能的取值
        self.Pxy___ = None    # 字典：存储所有条件概率；高斯分布的均值、方差、标准差
        self.indexContinuousFeatures_ = None  # 数组索引：连续值特征的序号

    def fit(self, X__: ndarray, y_: ndarray,
            indexContinuousFeatures_=(),  # 连续值特征的序号
            ):
        """训练"""
        assert type(X__)==ndarray and X__.ndim==2, '输入训练样本矩阵X__应为2维ndarray'
        assert type(y_)==ndarray  and y_.ndim==1, '输入训练标签y_应为1维ndarray'
        assert len(X__)==len(y_), '输入训练样本数量应等于标签数量'
        self.indexContinuousFeatures_ = tuple(indexContinuousFeatures_)  # 数组索引：连续值特征的序号
        if self.indexContinuousFeatures_:
            print(f'给定第{self.indexContinuousFeatures_}特征为连续值特征')
        N, self.M = X__.shape          # 训练样本数量、输入特征向量的维数
        self.classes_ = unique(y_)     # K维向量：K个类别标签
        self.K = len(self.classes_)    # 类别数
        self.values__ = values__ = {}  # 字典：训练样本的各个离散特征的所有可能取值
        self.Py_ = {}                  # 字典：存储K个类别的先验概率
        self.Pxy___ = Pxy___ = {}      # 字典：存储所有条件概率；高斯分布的均值、方差、标准差
        # Pxy___[c1ass][m][value] 表示“在c1ass类别下，第m个特征取值为value的条件概率”
        # Pxy___[c1ass][m]['μ'] 表示“在c1ass类别下，第m个特征取值所服从高斯分布的为均值”
        for k, c1ass in enumerate(self.classes_):
            """遍历所有K个类别"""
            # 计算先验概率
            Nk = (y_==c1ass).sum()  # 第k类别的样本数量
            if self.isSmoothing:
                self.Py_[c1ass] = (Nk + 1)/(N + self.K)  # 使用平滑的先验概率
            else:
                self.Py_[c1ass] = Nk/N  # 不使用平滑的先验概率

            # 计算条件概率
            Pxy___[c1ass] = {}  # 存储c1ass类别的条件概率
            for m in range(self.M):
                """遍历第k类别的所有M个特征"""
                Pxy___[c1ass][m] = {}  # 存储c1ass类别的第m特征的条件概率或高斯分布参数

                if m in self.indexContinuousFeatures_:
                    # 对于连续值特征，计算均值μ、方差σ2、标准差σ
                    xm_ = X__[y_==c1ass, m].astype(float)    # 提取第k类别第m特征的数据
                    Pxy___[c1ass][m]['μ'] = xm_.mean()       # 第k类别第m特征取值的均值
                    Pxy___[c1ass][m]['σ2'] = σ2 = xm_.var()  # 第k类别第m特征取值的方差
                    Pxy___[c1ass][m]['σ'] = σ2**0.5          # 第k类别第m特征取值的标准差
                else:
                    # 对于离散型特征，计算条件概率
                    values__[m] = unique(X__[:, m])  # 第m特征所有可能的取值
                    L = len(values__[m])             # 第m特征可能取值的数量
                    for l, value in enumerate(values__[m]):
                        """遍历第k类别的第m特征的所有可能的取值，计算条件概率"""
                        Nxy = ((y_==c1ass) & (X__[:, m]==value)).sum()   # 满足“属于第k类别且第m特征取值为value”的样本数量
                        if self.isSmoothing:
                            Pxy___[c1ass][m][value] = (Nxy + 1)/(Nk + L)  # 使用平滑的条件概率
                        else:
                            Pxy___[c1ass][m][value] = Nxy/Nk  # 不使用平滑的条件概率
        return self

    def predict(self, X__: ndarray) -> ndarray:
        """测试"""
        assert type(X__)==ndarray and X__.ndim==2, '输入测试样本矩阵X__应为2维ndarray'
        assert X__.shape[1]==self.M, f'输入测试样本维数应等于训练样本维数{self.M}'
        PyPxy__ = zeros([len(X__), self.K])  # N×K矩阵：初始化先验概率与条件概率之积
        for n in range(len(X__)):
            """遍历所有测试样本"""
            for k, c1ass in enumerate(self.classes_):
                """遍历所有K个类别，计算测试样本属于各类别的后验概率（先验概率与条件概率之积）"""
                PyPxy__[n, k] = self.Py_[c1ass]  # 赋先验概率
                for m in range(self.M):
                    """遍历第k类别所有特征，索引对应的条件概率"""
                    if m in self.indexContinuousFeatures_:
                        # 若第m特征为连续值特征
                        μ = self.Pxy___[c1ass][m]['μ']    # 第m特征的均值
                        σ = self.Pxy___[c1ass][m]['σ']    # 第m特征的标准差
                        σ2 = self.Pxy___[c1ass][m]['σ2']  # 第m特征的方差
                        xm = float(X__[n, m])             # 第m特征的取值
                        π = 3.141592653589793             # 圆周率
                        e = 2.718281828459045             # 自然对数的底数
                        p = 1/((2*π)**0.5*σ)*e**(-(xm - μ)**2/(2*σ2))  # 高斯概率密度
                        PyPxy__[n, k] *= p  # 先验概率与条件概率密度之积
                    else:
                        # 若第m特征为离散值特征
                        value = X__[n, m]  # 读取：第n样本的第m特征的取值
                        assert value in self.values__[m], f"训练样本集当中第{m}特征无此取值'{value}'"
                        PyPxy__[n, k] *= self.Pxy___[c1ass][m][value]  # 先验概率与条件概率之积
        y_ = self.classes_[PyPxy__.argmax(axis=1)]  # 根据最大化类别的后验概率，归类
        return y_

    def accuracy(self, X__: ndarray, y_: ndarray) -> float:
        """预测正确率"""
        测试样本正确数 = sum(self.predict(X__)==y_)
        测试样本总数 = len(y_)
        return 测试样本正确数/测试样本总数

if __name__=='__main__':
    import numpy as np
    # 设置随机种子
    from numpy.random import seed; seed(0)
    from random import seed; seed(0)

    print('使用天气数据集进行试验...\n')
    天气数据集 = np.array([
    #    风向   湿度 紫外线指数  温度  类别标签
        ['东', '干燥',  '强',    33,  '晴天'],
        ['南', '潮湿',  '强',    31,  '晴天'],
        ['南', '潮湿',  '弱',    24,  '阴天'],
        ['西', '干燥',  '强',    27,  '阴天'],
        ['西', '潮湿',  '强',    30,  '阴天'],
        ['南', '适中',  '弱',    22,  '雨天'],
        ['北', '潮湿',  '弱',    20,  '雨天'],
        ])

    print('使用仅包含离散值特征（风向、湿度、紫外线指数）的训练样本集')
    Xtrain__ = 天气数据集[:, :3]            # 训练样本特征向量矩阵
    ytrain_  = 天气数据集[:, -1]            # 训练样本标签
    model = NaiveBayes(isSmoothing=True)   # 实例化朴素贝叶斯模型，使用平滑
    model.fit(Xtrain__, ytrain_)           # 训练
    print(f'对训练样本预测的正确率：{model.accuracy(Xtrain__, ytrain_):.3f}')

    Xtest__ = np.array([['南', '干燥', '强'],
                        ['北', '干燥', '强']])  # 测试样本集
    ytest_ = model.predict(Xtest__)            # 测试样本集的预测结果
    for x_, y in zip(Xtest__, ytest_):
        print(f'预测新样本 {x_} 的类别为 "{y}"')  # 显示预测结果

    print('\n---------------------------------------------------\n')
    print('使用包含离散值特征和连续值特征（风向、湿度、紫外线指数、温度）的训练样本集')
    Xtrain__ = 天气数据集[:, :4]           # 训练样本特征向量矩阵
    ytrain_  = 天气数据集[:, -1]           # 训练样本标签
    indexContinuousFeatures_ = (3,)       # 连续值特征的序号
    model = NaiveBayes(isSmoothing=True)  # 实例化朴素贝叶斯模型，使用平滑
    model.fit(Xtrain__, ytrain_, indexContinuousFeatures_=indexContinuousFeatures_)  # 训练
    print(f'对训练样本预测的正确率：{model.accuracy(Xtrain__, ytrain_):.3f}')

    Xtest__ = np.array([['南', '干燥', '强', 34],
                        ['北', '干燥', '强', 26]])  # 测试样本集
    ytest_ = model.predict(Xtest__)                # 测试样本集的预测结果
    for x_, y in zip(Xtest__, ytest_):
        print(f'预测新样本 {x_} 的类别为 "{y}"')  # 显示预测结果
