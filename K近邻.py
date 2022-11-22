from numpy import ndarray, inf, array, zeros, unique


class KNearestNeighbors:
    """K近邻分类、回归算法"""
    def __init__(self,
            K: int = 3,          # 超参数：近邻数K
            mode: str = '回归',  # 模式：可选 '分类'/'回归'
            ):
        assert type(K)==int and K>0, '近邻数K应为正整数'
        assert mode in {'分类', '回归'}, "模式mode应为'分类' 或 '回归'"
        self.K = K            # 超参数：近邻数K
        self.mode = mode      # 模式：可选 '分类'/'回归'
        self.M = None         # 输入特征向量的维数
        self.KDtree = None    # 字典：KD树（k-dimensional tree）
        self.classes_ = None  # 向量：'分类'模式，存储所有类别

    def fit(self, X__: ndarray, y_: ndarray):
        """训练：根据训练样本集创建KD树"""
        assert type(X__)==ndarray and X__.ndim==2, '输入训练样本矩阵X__应为2维ndarray'
        assert type(y_)==ndarray  and y_.ndim==1, '输入训练样本标签y_应为1维ndarray'
        assert len(X__)==len(y_), '输入训练样本数量应等于标签数量'
        print(f'训练K近邻{self.mode}器，近邻数K = {self.K}')
        self.M = X__.shape[1]   # 输入特征向量的维数
        if self.mode=='分类':
            self.classes_ = unique(y_)  # 向量：存储所有类别
        self.KDtree = self.createKDtree(X__, y_)  # 构建KD树
        return self

    def createKDtree(self,
            X__: ndarray,    # 矩阵：样本的特征向量
            y_: ndarray,     # 向量：样本的标签
            depth: int = 0,  # 当前树深
            ):
        """递归地构建KD树"""
        N = len(X__)  # 样本数量
        if N==0:
            # 若无输入样本，则pass，直接返回None
            pass
        elif N==1:
            # 若仅输入1个样本，返回结点（字典）
            m = depth % self.M  # 选择第m特征进行切分
            node = {   # 创建结点
                'xSplit_' : X__[0].copy(),  # 切分点的样本
                'ySplit' : y_[0].copy(),    # 切分点的标签
                'splittingDimension' : m,   # 切分点的特征索引
                'left' : None,  # 再无左子结点
                'right': None,  # 再无右子结点
                }
            return node
        else:
            # 若输入至少2个样本，返回结点（字典）
            m = depth % self.M                # 选择第m特征进行切分
            Xm_ = X__[:, m].copy()            # N维向量：提取第m特征的数据
            indexSort_ = Xm_.argsort()        # N维向量：将Xm_从小到大排列的索引
            indexMedian = indexSort_[N//2]    # Xm_的中位数的索引
            xSplit_ = X__[indexMedian].copy() # 切分点的样本（Xm_中位数对应的样本）
            ySplit = y_[indexMedian].copy()   # 切分点的标签（Xm_中位数对应的样本标签）
            depth += 1                        # 树深+1
            indexLeft_ = indexSort_[:(N//2)]     # 数组索引：划分为左结点的样本
            indexRight_ = indexSort_[(N//2)+1:]  # 数组索引：划分为右结点的样本
            node = {  # 创建结点
                'xSplit_' : xSplit_,       # 切分点的样本
                'ySplit'  : ySplit,        # 切分点的标签
                'splittingDimension' : m,  # 切分点的特征索引
                'left'  : self.createKDtree(X__[indexLeft_],  y_[indexLeft_],  depth),  # 创建左结点KD树
                'right' : self.createKDtree(X__[indexRight_], y_[indexRight_], depth),  # 创建右结点KD树
                }
            return node

    def search(self,
               node: dict,        # 字典或None：当前结点
               x_: ndarray,       # M维向量：单个样本
               dictionary: dict,  # 一个存储K个近邻的信息的字典
               ) -> None:
        """
        递归地从KD树找到K个近邻

        字典dictionary的结构:
        dictionary = {'nearest X__' : [x1_, x2_, ..., xK_],  # 列表：存储K个距离样本x_最近的训练样本
                      'nearest y_' :  [y1, y2, ..., yK],     # 列表：存储K个距离样本x_最近的训练样本的标签
                      'distances_' : [d1, d2, ..., dK],      # 列表：存储样本x_与K个近邻训练样本的距离值
                      'maxDistance' : dMax,                  # 数值：存储dictionary['distances_']的最大值
                       }
        """
        if node is None:
            # 若当前结点无训练样本，则返回
            return
        m = node['splittingDimension']  # 当前结点以第m特征进行切分
        xSplit_ = node['xSplit_']       # 当前结点的切分点样本
        ySplit = node['ySplit']         # 当前结点的切分点样本标签
        if x_[m]<xSplit_[m]:
            # 若测试样本x_的第m特征值小于当前切分点样本的第m特征值，则测试样本x_离左子树更近
            nearerNode = node['left']
            furtherNode = node['right']
        else:
            # 否则，测试样本x_离右子树更近
            nearerNode = node['right']
            furtherNode = node['left']

        # 搜索距离测试样本x_较近的子树nearerNode
        self.search(nearerNode, x_, dictionary)      # 搜索子树nearerNode
        distance = (((xSplit_ - x_)**2).sum())**0.5  # 测试样本x_与切分点xSplit_的距离
        if len(dictionary['nearest y_'])<self.K:
            # 若dictionary所存储的近邻尚少于K个，则添加近邻
            dictionary['nearest X__'].append(xSplit_)
            dictionary['nearest y_'].append(ySplit)
            dictionary['distances_'].append(distance)
            dictionary['maxDistance'] = max(dictionary['distances_'])
        else:
            # 若dictionary所存储的近邻已达到K个，则检查是否可替代某个近邻
            if distance<dictionary['maxDistance']:
                # 若当前结点的切分点xSplit_与测试样本x_的距离小于dictionary的最大距离，则替换掉最远距离的近邻
                indexMax = dictionary['distances_'].index(dictionary['maxDistance'])  # 索引最远距离的近邻
                dictionary['nearest X__'][indexMax] = xSplit_     # 替换成当前结点的切分点样本
                dictionary['nearest y_'][indexMax] = ySplit       # 替换成当前结点的切分点样本标签
                dictionary['distances_'][indexMax] = distance     # 替换距离值
                dictionary['maxDistance'] = max(dictionary['distances_'])  # 更新最大距离

        # 判断是否搜索较远的子树furtherNode
        # 检查furtherNode的区域是否可能存在更近的样本
        # 即检查“furtherNode的区域”是否与“以x_为球心、以x_与‘当前近邻点’之间的距离为半径的超球体”相交
        if abs(x_[m] - xSplit_[m])<=dictionary['maxDistance']:
            # 若相交，则搜索距离测试样本x_较远的子树furtherNode
            self.search(furtherNode, x_, dictionary)
        return

    def predict(self, X__: ndarray) -> ndarray:
        """测试"""
        assert type(X__)==ndarray and X__.ndim==2, '输入测试样本矩阵X__应为2维ndarray'
        assert X__.shape[1]==self.M, f'输入测试样本维数应等于训练样本维数{self.M}'
        y_ = []  # 所有预测值
        for x_ in X__:
            # 遍历所有测试样本x_，作出预测
            dictionary = {'nearest X__' : [],
                          'nearest y_'  : [],
                          'distances_'  : [],
                          'maxDistance' : inf}
            self.search(self.KDtree, x_, dictionary)  # 找到测试样本x_的K个近邻，存储在dictionary
            if self.mode=='分类':
                y_.append(self.majority(dictionary['nearest y_'])) # 投票，取多数类别作为预测类别
            elif self.mode=='回归':
                y_.append(sum(dictionary['nearest y_']) / self.K)  # 取K个近邻的标签的平均值作为回归值
        return array(y_)

    def majority(self, y_):
        """计算“众数”：输入标签集y_，输出占多数的类别标签（众数）"""
        N_ = zeros(len(self.classes_))  # 向量：初始属于各个类别的样本数量
        for k, c1ass in enumerate(self.classes_):
            # 遍历所有类别，计算各类别的样本数量
            N_[k] = (y_==c1ass).sum()
        majorityClass = self.classes_[N_.argmax()]  # 占最多数的类别（众数）
        return majorityClass

    def MAE(self, X__: ndarray, y_: ndarray) -> float:
        """计算回归测试的平均绝对误差MAE"""
        assert self.mode=='回归', '回归模式方可计算平均绝对误差MAE'
        return abs(self.predict(X__) - y_).mean()

    def accuracy(self, X__: ndarray, y_: ndarray) -> float:
        """计算分类测试的正确率"""
        assert self.mode=='分类', '分类模式方可计算测试正确率'
        测试正确样本数 = sum(self.predict(X__)==y_)
        测试样本总数 = len(y_)
        return 测试正确样本数/测试样本总数

if __name__=='__main__':
    # 设置随机种子
    from numpy.random import seed; seed(0)
    from random import seed; seed(0)

    # 导入分类数据集
    print('使用鸢尾花分类数据集进行分类测试...')
    from sklearn.datasets import load_iris
    X__, y_ = load_iris()['data'], load_iris()['target']  # 提取样本、标签
    # 将数据拆分成训练集、测试集
    from sklearn.model_selection import train_test_split
    Xtrain__, Xtest__, ytrain_, ytest_ = train_test_split(X__, y_)
    # 实例化K近邻分类模型
    model = KNearestNeighbors(
        K=15,  # 超参数：近邻数K
        mode='分类')
    model.fit(Xtrain__, ytrain_)  # 训练
    print(f'训练集正确率 {model.accuracy(Xtrain__, ytrain_):.4f}')
    print(f'测试集正确率 {model.accuracy(Xtest__, ytest_):.4f}')

    # 对比sklearn的K近邻分类
    print('\n使用同样的超参数设定值、训练集和测试集，对比sklearn的K近邻分类')
    from sklearn.neighbors import KNeighborsClassifier  # 导入sklearn的K近邻分类器
    modelSK = KNeighborsClassifier(n_neighbors=model.K)
    modelSK.fit(Xtrain__, ytrain_)
    print(f'sklearn的K近邻分类：训练集正确率 {modelSK.score(Xtrain__, ytrain_):.4f}')
    print(f'sklearn的K近邻分类：测试集正确率 {modelSK.score(Xtest__, ytest_):.4f}')

    print('='*60)

    # 导入回归数据集
    print('\n使用波士顿房价回归数据集进行回归测试...')
    from sklearn.datasets import load_boston
    X__, y_ = load_boston()['data'], load_boston()['target']  # 提取样本、标签
    # 将数据拆分成训练集、测试集
    Xtrain__, Xtest__, ytrain_, ytest_ = train_test_split(X__, y_)
    # 实例化K近邻回归模型
    model = KNearestNeighbors(
        K=9,   # 超参数：近邻数K
        mode='回归',
        )
    model.fit(Xtrain__, ytrain_)  # 训练
    print(f'训练集平均绝对误差 {model.MAE(Xtrain__, ytrain_)}')
    print(f'测试集平均绝对误差 {model.MAE(Xtest__, ytest_)}')

    # 对比sklearn的K近邻回归
    print('\n使用同样的超参数设定值、训练集和测试集，对比sklearn的K近邻回归')
    from sklearn.neighbors import KNeighborsRegressor   # 导入sklearn的K近邻回归器
    modelSK = KNeighborsRegressor(n_neighbors=model.K)
    modelSK.fit(Xtrain__, ytrain_)
    print(f'sklearn的K近邻回归：训练集平均绝对误差 {abs(modelSK.predict(Xtrain__) - ytrain_).mean()}')
    print(f'sklearn的K近邻回归：测试集平均绝对误差 {abs(modelSK.predict(Xtest__) - ytest_).mean()}')

    print(f'\n对于完整数据集，两个K近邻回归器的最大预测偏差：{abs(modelSK.predict(X__) - model.predict(X__)).max()}')