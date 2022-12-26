from typing import Optional
from numpy import ndarray, unique, array
from numpy.random import choice

from 决策树工具函数 import entropy, purity, majority, \
    splitDataset_forDiscreteFeature,\
    biPartitionDataset_forContinuousFeature

class ID3decisionTree:
    """ID3决策树分类"""
    def __init__(self,
            minSamplesLeaf: int = 1,            # 超参数：叶结点的最少样本数量
            maxDepth: int = 7,                  # 超参数：最大树深
            maxPruity: float = 1.,              # 超参数：叶结点的最大纯度
            maxFeatures: Optional[int] = None,  # 超参数：最大特征数
            α: float = 0.,                      # 超参数：代价复杂度剪枝的惩罚参数
            ):
        assert type(minSamplesLeaf)==int and minSamplesLeaf>0, '叶结点的最少样本数量minSamplesLeaf应为正整数'
        assert type(maxDepth)==int and maxDepth>0, '最大树深maxDepth应为正整数'
        assert 0<maxPruity<=1, '叶结点的最大纯度maxDepth的取值范围应为(0, 1]'
        assert (maxFeatures is None) or (type(maxFeatures)==int and maxFeatures>0), '最大特征数maxFeatures应为正整数，或不给定maxFeatures'
        assert α>=0, '代价复杂度剪枝的惩罚参数α应为非负数'
        self.minSamplesLeaf = minSamplesLeaf  # 超参数：叶结点的最少样本数量
        self.maxDepth = maxDepth              # 超参数：最大树深
        self.maxPruity = maxPruity            # 超参数：叶结点的最大纯度
        self.maxFeatures = maxFeatures        # 超参数：最大特征数
        self.α = α                            # 超参数：代价复杂度剪枝的惩罚参数
        self.M = None                         # 输入特征向量的维数
        self.tree = None                      # 字典：树
        self.indexContinuousFeatures_ = None  # 数组索引：连续值特征的序号

    def fit(self, X__: ndarray, y_: ndarray,
            indexContinuousFeatures_=(),  # 数组索引：连续值特征的序号
            ):
        """训练：生成ID3决策树"""
        assert type(X__)==ndarray and X__.ndim==2, '输入训练样本矩阵X__应为2维ndarray'
        assert type(y_)==ndarray  and y_.ndim==1, '输入训练样本标签y_应为1维ndarray'
        assert len(X__)==len(y_), '输入训练样本数量应等于标签数量'
        self.M = X__.shape[1]  # 输入特征向量的维数
        self.indexContinuousFeatures_ = tuple(indexContinuousFeatures_)  # 数组索引：连续值特征的序号
        if self.indexContinuousFeatures_:
            print(f'给定第{self.indexContinuousFeatures_}特征为连续值特征')

        if self.maxFeatures is None:
            self.maxFeatures = self.M  # 若未指定最大特征数maxFeatures，则默认设置为特征总数M
        assert self.maxFeatures<=self.M, f'最大特征数maxFeatures不应大于输入特征向量的维数{self.M}，当前 maxFeatures = {self.maxFeatures}'

        indexFeatureCandidates_ = array(range(self.M))  # 数组索引：所有候选特征的序号
        self.tree = self.createTree(X__, y_, indexFeatureCandidates_=indexFeatureCandidates_)  # 生成树
        return self

    def createTree(self,
                   X__: ndarray,
                   y_: ndarray,
                   indexFeatureCandidates_: ndarray,  # 数组索引：所有候选的特征的序号
                   depth: int = 0,                    # 当前树深
                   ) -> dict:
        """递归地生成ID3分类树"""
        '''
        结点的结构，字典：

            {
            'class label' : 结点的类别标签,
            'number of samples' : 结点的样本数量,
            'entropy' : 结点的熵值,
            'purity': 结点的纯度,
            'child nodes' : {'划分特征的取值1' : 子结点,
                             '划分特征的取值2' : 子结点,
                             ...
                            }
            'index of splitting feature' : 划分特征的序号,
            'partition point' : 划分点,
            }
        
        
        注：
        当结点的划分特征为离散值特征时，无'partition point'键值对；
        当结点的划分特征为离散值特征时，'划分特征的取值1'、'划分特征的取值2'、'划分特征的取值3'、……等，分别为划分特征的各个离散取值；
        当结点的划分特征为连续值特征时，'划分特征的取值1'和'划分特征的取值2'分别为'+'和'-'，分别指示划分特征取值大于、不大于划分点的子结点；
        当结点为叶结点时，无'child nodes'、'index of splitting feature'、'partition point'等键值对。
        '''

        N = len(X__)    # 结点样本数量
        p = purity(y_)  # 结点纯度
        """创建结点"""
        node = {'class label': majority(y_),  # 记录：结点类别
                'number of samples': N,       # 记录：结点样本数量
                'entropy': entropy(y_),       # 记录：结点熵值
                'purity': p,                  # 记录：结点纯度
                }
        if (len(set(y_))==1 or           # 若该结点是纯的（所有类别标签都相同），则返回叶结点
            len(indexFeatureCandidates_)==0 or  # 若所有特征已被用于划分，则返回叶结点
            depth>=self.maxDepth or      # 若达到最大树深，则返回叶结点
            N<=self.minSamplesLeaf or    # 若达到叶结点的最少样本数量，则返回叶结点
            p>=self.maxPruity or         # 若达到叶结点的最大纯度，则返回叶结点
            len(unique(X__, axis=0))==1  # 若所有样本特征向量都相同，则返回叶结点
           ):
            return node
        """选择最优的划分特征"""
        result = self.chooseBestFeature(X__, y_, indexFeatureCandidates_)
        if type(result)==tuple:
            # 若result是一个元组，则选择了一个连续值特征序号和对应的最优划分点
            mBest, tBest = result[0], result[1]
        else:
            # 若result是一个数，则选择了一个离散值特征序号
            mBest = result
        """创建子结点"""
        node['index of splitting feature'] = mBest  # 记录：最优划分特征的序号
        depth += 1                                  # 树深+1
        if mBest in self.indexContinuousFeatures_:
            # 若第mBest特征是连续值特征，则按最优划分点tBest，二分样本集
            node['partition point'] = tBest  # 记录：最优划分点
            node['child nodes'] = {}         # 初始化子结点
            Xhigher__, Xlower__, yHigher_, yLower_ = \
                biPartitionDataset_forContinuousFeature(X__=X__, y_=y_, m=mBest, t=tBest)  # 二分样本集
            node['child nodes']['+'] = \
                self.createTree(Xhigher__, yHigher_, indexFeatureCandidates_, depth)  # 创建第mBest特征取值大于tBest的子结点
            node['child nodes']['-'] = \
                self.createTree(Xlower__,  yLower_,  indexFeatureCandidates_, depth)  # 创建第mBest特征取值不大于tBest的子结点
        else:
            # 若第mBest特征是离散值特征，则按第mBest特征的所有离散取值划分样本集
            indexFeatureCandidates_ = indexFeatureCandidates_[indexFeatureCandidates_!=mBest]  # 剔除已被用于划分的离散值特征
            values_ = unique(X__[:, mBest])  # 第mBest特征的所有取值
            node['child nodes'] = {}         # 初始化子结点
            for value in values_:
                # 遍历最优划分特征的所有取值，分别创建子结点
                Xsubset__, ySubset_ = \
                    splitDataset_forDiscreteFeature(X__=X__, y_=y_, m=mBest, value=value)  # 划分出子样本集
                node['child nodes'][value] = \
                    self.createTree(Xsubset__, ySubset_, indexFeatureCandidates_, depth)  # 创建第mBest特征取值为value的子结点
        return node

    def chooseBestFeature(self, X__: ndarray, y_: ndarray,
                          indexFeatureCandidates_: ndarray):
        """
        选择最优的划分特征：
        输入样本集X__、标签y_，输出最优划分特征的序号mBest，对于连续值特征，还输出最优划分点tBest
        """
        N = len(X__)               # 样本数量
        baseEntropy = entropy(y_)  # 样本集的信息熵
        for m in indexFeatureCandidates_.copy():
            if len(set(X__[:, m]))==1:
                # 若第m特征只有1个取值，则剔除该特征，不能选该特征进行划分
                indexFeatureCandidates_ = indexFeatureCandidates_[indexFeatureCandidates_!=m]
        Nfeatures = min(self.maxFeatures, len(indexFeatureCandidates_))                   # 特征子集的特征数量
        indexFeaturesChosen_ = choice(indexFeatureCandidates_, Nfeatures, replace=False)  # 数组索引：随机选取的特征子集的序号
        informationGains_ = {}         # 初始化字典：{ 选取的特征的序号: 按该特征划分所获得的信息增益}
        indexBestPartitionPoint_ = {}  # 初始化字典：{ 选取的连续值特征的序号: 该连续值特征的最优划分点}
        for m in indexFeaturesChosen_:
            # 遍历所有选取的特征，计算按各特征划分所获得的信息增益
            values_ = unique(X__[:, m])  # 第m特征所有的取值（剔除重复值）
            if m in self.indexContinuousFeatures_:
                # 若第m特征为连续值特征
                values_ = values_.astype(float)      # 将第m特征的取值转化为浮点型
                values_.sort()                       # 升序排列第m特征的取值
                t_ = (values_[:-1] + values_[1:])/2  # 划分点序列
                informationGains_[m] = -1e100  # 初始化：按第m特征划分所获得的信息增益
                for t in t_:
                    # 遍历第m特征的所有划分点，找到最优划分点
                    Xhigher__, Xlower__, yHigher_, yLower_ = \
                        biPartitionDataset_forContinuousFeature(X__=X__, y_=y_, m=m, t=t)  # 按划分点t划分成2个子样本集
                    pHigher = len(yHigher_)/N  # 第m特征取值大于t的子样本集所占比例
                    pLower  = len(yLower_)/N   # 第m特征取值不大于t的子样本集所占比例
                    newEntropy = pHigher*entropy(yHigher_) + pLower*entropy(yLower_)  # 2个子样本集的熵值
                    informationGain = baseEntropy - newEntropy  # 信息增益
                    if  informationGain > informationGains_[m]:
                        indexBestPartitionPoint_[m] = t         # 记录：最优划分点
                        informationGains_[m] = informationGain  # 记录：最大信息增益
            else:
                # 若第m特征为离散值特征
                newEntropy = 0.  # 初始化：按第m特征划分得到的多个子样本集的熵值
                for value in values_:
                    # 遍历第m特征所有的取值
                    Xsubset__, ySubset_ = \
                        splitDataset_forDiscreteFeature(X__=X__, y_=y_, m=m, value=value)  # 划分出子样本集
                    p = len(ySubset_)/N                # 第m特征取值为value的样本所占比例
                    newEntropy += p*entropy(ySubset_)  # 多个子样本集的熵值
                informationGains_[m] = baseEntropy - newEntropy  # 按第m特征划分所获得的信息增益

        """选择最优的特征"""
        maxInformationGain = -1e100  # 初始化最大信息增益
        for m in informationGains_:
            # 遍历所有选取特征的信息增益，找出信息增益最大的特征
            if informationGains_[m]>maxInformationGain:
                maxInformationGain = informationGains_[m]  # 最大信息增益
                mBest = m   # ID3算法选择信息增益最大的特征作为最优的划分特征

        if mBest in self.indexContinuousFeatures_:
            # 若最优的划分特征是连续值特征，则返回特征序号和最优划分点
            tBest = indexBestPartitionPoint_[mBest]  # 最优划分点
            return mBest, tBest
        else:
            # 若最优的划分特征是离散值特征，则仅返回特征序号
            return mBest

    def classify(self, node, x_):
        """递归地确定样本x_的类别：输入1个样本x_，输出预测类别"""
        if 'child nodes' not in node:
            # 若当前结点不含'child nodes'键，则该结点为叶结点，返回叶结点的类别标签
            y = node['class label']
        else:
            # 若当前结点是内部结点，则继续进入下级子结点
            m = node['index of splitting feature']  # 当前结点以第m特征进行划分
            if m in self.indexContinuousFeatures_:
                # 若第m特征是连续值特征
                t = node['partition point']  # 读取划分点
                if x_[m].astype(float)>t:
                    node = node['child nodes']['+']  # 进入第m特征取值大于划分点t的子结点
                else:
                    node = node['child nodes']['-']  # 进入第m特征取值不大于划分点t的子结点
            else:
                # 若第m特征是离散值特征
                for value in node['child nodes'].keys():
                    # 遍历当前结点第m特征的取值
                    if x_[m]==value:
                        node = node['child nodes'][value]  # 进入第m特征取值为value的子结点
                        break
                else:
                    # 若当前结点无x_[m]这个取值的子结点，返回当前节点的类别标签
                    y = node['class label']
                    return y

            y = self.classify(node, x_)  # 进入子结点
        return y

    def pruning(self):
        """剪枝"""
        print(f'执行代价复杂度剪枝，惩罚参数：α = {self.α}')
        print(f'剪枝前，ID3决策树的叶结点总数：{self.countLeafNumber(self.tree)}')
        self.tree = self.costComplexityPruning(self.tree)  # 剪枝
        print(f'剪枝后，ID3决策树的叶结点总数：{self.countLeafNumber(self.tree)}')
        return self

    def costComplexityPruning(self, node):
        """递归地执行代价复杂度剪枝"""
        for key in node['child nodes']:
            # 对所输入的内部结点，遍历其各个子结点
            if 'child nodes' in node['child nodes'][key]:
                # 若子结点含有键'child nodes'，则该结点是内部结点，可尝试剪枝
                node['child nodes'][key] = self.costComplexityPruning(node['child nodes'][key])

        # 判断当前结点node的所有子结点是否为叶结点
        nodeIsLeaf_ = [('child nodes' not in node['child nodes'][key])
                       for key in node['child nodes']]  # 若当前结点node的某个子结点不含有下一级子结点，则该子结点为叶结点

        if all(nodeIsLeaf_):
            # 若当前结点node的所有子结点都是叶结点，尝试剪枝
            lossBeforePruning = 0.  # 初始化：剪枝前的损失函数值
            for key in node['child nodes']:
                # 遍历所有叶结点
                entropy = node['child nodes'][key]['entropy']      # 叶结点的熵值
                N = node['child nodes'][key]['number of samples']  # 叶结点的样本数量
                lossBeforePruning += entropy*N            # 剪枝前的损失函数值
            lossBeforePruning += self.α*len(nodeIsLeaf_)  # 剪枝前的损失函数值
            lossAfterPruning = node['number of samples']*node['entropy'] + self.α   # 剪枝后的损失函数值
            if lossAfterPruning <= lossBeforePruning:
                # 若剪枝后的损失函数值小于或等于剪枝前的损失函数值，则剪枝
                del node['child nodes']  # 删除键'child nodes'，内部结点变成叶结点
                del node['index of splitting feature']
                if 'partition point' in node:
                    del node['partition point']
        return node

    def countLeafNumber(self, node: dict) -> int:
        """计算结点（树）的叶结点数"""
        if 'child nodes' not in node:
            # 若结点node无'child nodes'键，则其为叶结点
            number = 1
        else:
            # 若结点node有'child nodes'键，则其为内部结点
            number = 0
            for key in node['child nodes']:
                # 遍历内部结点的子结点
                number += self.countLeafNumber(node['child nodes'][key])
        return number

    def predict(self, X__: ndarray) -> ndarray:
        """测试"""
        assert type(X__)==ndarray and X__.ndim==2, '输入测试样本矩阵X__应为2维ndarray'
        assert X__.shape[1]==self.M, f'输入测试样本特征向量维数应等于训练样本特征向量维数{self.M}'
        y_ = [self.classify(self.tree, x_) for x_ in X__]  # 对样本逐个进行分类
        y_ = array(y_)
        return y_

    def accuracy(self, X__: ndarray, y_: ndarray) -> float:
        """计算测试正确率"""
        测试正确样本数 = sum(self.predict(X__)==y_)
        总样本数 = len(y_)
        return 测试正确样本数/总样本数

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
        ['北', '适中',  '强',    35,  '晴天'],
        ['西', '适中',  '强',    26,  '晴天'],

        ['南', '潮湿',  '弱',    24,  '阴天'],
        ['西', '干燥',  '强',    27,  '阴天'],
        ['西', '潮湿',  '强',    30,  '阴天'],
        ['东', '适中',  '弱',    22,  '阴天'],

        ['南', '适中',  '弱',    22,  '雨天'],
        ['北', '潮湿',  '弱',    20,  '雨天'],
        ['东', '适中',  '弱',    29,  '雨天'],
        ['西', '潮湿',  '弱',    24,  '雨天'],
        ])

    Xtrain__ = 天气数据集[:, :-1]     # 样本特征向量矩阵
    ytrain_ = 天气数据集[:, -1]       # 样本标签
    indexContinuousFeatures_ = (3,)  # 连续值特征的序号

    # 实例化ID3决策树模型
    model = ID3decisionTree(
            minSamplesLeaf=1,  # 超参数：叶结点的最少样本数量
            maxDepth=7,        # 超参数：最大树深
            maxPruity=1.,      # 超参数：叶结点的最大纯度
            # maxFeatures=4,     # 超参数：最大特征数
            α=1.5,             # 超参数：代价复杂度剪枝的惩罚参数
            )
    model.fit(Xtrain__, ytrain_,
              indexContinuousFeatures_=indexContinuousFeatures_,  # 连续值特征的序号
              )      # 训练
    model.pruning()  # 剪枝
    print(f'训练样本集的测试正确率：{model.accuracy(Xtrain__, ytrain_):.4f}\n')

    Xtest__ = np.array([['东', '潮湿', '弱', 20],
                        ['南', '干燥', '强', 34],
                        ['北', '干燥', '强', 26],
                        ['南', '干燥', '弱', 20]])  # 测试样本集
    ytest_ = model.predict(Xtest__)                # 测试样本集的预测结果
    for x_, y in zip(Xtest__, ytest_):
        print(f'预测新样本 {x_} 的类别为 "{y}"')  # 显示预测结果
