from numpy import ndarray, array, zeros, unique, log2

def biPartitionDataset_forDiscreteFeature(
        X__: ndarray,  # 样本特征向量矩阵
        y_: ndarray,   # 样本标签
        m: int,        # 特征序号
        value,         # 第m特征的某个取值
        ):
    """
    对离散值特征，二分样本集:
    将原始样本集X__、标签集y_，按照第m特征的取值是否为value，划分为2个子集
    """
    logicYes_ = X__[:, m]==value  # 逻辑索引：第m特征的取值是value的样本
    logicNo_ = ~logicYes_         # 逻辑索引：第m特征的取值非value的样本
    # 提取第m特征取值是value的样本
    Xyes__ = X__[logicYes_]
    yYes_  = y_[logicYes_]
    # 提取第m特征取值非value的样本
    Xno__ = X__[logicNo_]
    yNo_  = y_[logicNo_]
    return Xyes__, Xno__, yYes_, yNo_

def biPartitionDataset_forContinuousFeature(
        X__: ndarray,  # 样本特征向量矩阵
        y_: ndarray,   # 样本标签
        m: int,        # 特征序号
        t: float,      # 第m特征的划分点
        ):
    """
    对连续值特征，二分样本集：
    将原始样本集X__、标签集y_，按照第m特征的取值是否大于t，划分为2个子集
    """
    logicHigher_ = X__[:, m].astype(float)>t  # 逻辑索引：第m特征的取值大于t的样本
    logicLower_ = ~logicHigher_               # 逻辑索引：第m特征的取值不大于t的样本
    # 提取第m特征的取值大于t的样本
    Xhigher__ = X__[logicHigher_]
    yHigher_ = y_[logicHigher_]
    # 提取第m特征的取值不大于t的样本
    Xlower__ = X__[logicLower_]
    yLower_ = y_[logicLower_]
    return Xhigher__, Xlower__, yHigher_, yLower_

def splitDataset_forDiscreteFeature(
        X__: ndarray,  # 样本特征向量矩阵
        y_: ndarray,   # 样本标签
        m: int,        # 特征序号
        value,         # 第m特征的某个取值
        ):
    """
    对离散值特征，划分出子样本集：
    提取原始样本集X__、标签集y_当中第m特征取值是value的子样本集
    """
    values_ = X__[:, m]  # 提取第m特征的取值
    # 提取第m特征的取值是value的样本
    Xsubset__ = X__[values_==value]
    ySubset_ = y_[values_==value]
    return Xsubset__, ySubset_

def entropy(y_: ndarray) -> float:
    """输入类别标签集y_，输出信息熵"""
    N = len(y_)            # 样本数量
    classes_ = unique(y_)  # 所有类别标签
    K = len(classes_)      # 类别数
    N_ = zeros(K)          # 各类别的样本数量
    for k, c1ass in enumerate(classes_):
        # 遍历K个类别，计算各类别的样本数量
        N_[k] = sum(y_==c1ass)  # 第k类别的样本数量
    p_ = N_/N  # 所有类别的样本数量占总样本数的比例
    s = -log2(p_).dot(p_)  # 信息熵
    return s

def gini(y_: ndarray) -> float:
    """输入类别标签集y_，输出基尼指数gini"""
    N = len(y_)            # 样本数量
    classes_ = unique(y_)  # 所有类别标签
    K = len(classes_)      # 类别数
    N_ = zeros(K)          # 各类别的样本数量
    for k, c1ass in enumerate(classes_):
        # 遍历K个类别，计算各类别的样本数量
        N_[k] = sum(y_==c1ass)  # 第k类别的样本数量
    p_ = N_/N         # K维向量：K个类别的样本在样本集所占比例
    gini = 1 - sum(p_**2)
    return gini

def purity(y_: ndarray) -> float:
    """输入标签集y_，输出纯度"""
    N = len(y_)            # 样本数量
    classes_ = unique(y_)  # 所有类别标签
    K = len(classes_)      # 类别数
    N_ = zeros(K)          # 各类别的样本数量
    for k, c1ass in enumerate(classes_):
        # 遍历K个类别，计算各类别的样本数量
        N_[k] = sum(y_==c1ass)  # 第k类别的样本数量
    p_ = N_/N  # 所有类别的样本在总体样本集所占比例
    return p_.max()

def majority(y_):
    """输入标签集y_，输出占多数的类别（众数）"""
    y_ = array(y_)         # 转化为ndarray
    assert y_.ndim==1, '输入标签集y_应可转化为1维ndarray'
    classes_ = unique(y_)  # 所有类别标签
    K = len(classes_)      # 类别数
    N_ = zeros(K)          # K个类别的样本数量
    for k, c1ass in enumerate(classes_):
        # 遍历K个类别，计算各类别的样本数量
        N_[k] = sum(y_==c1ass)  # 第k类别的样本数量
    majorityClass = classes_[N_.argmax()]  # 占多数的类别
    return majorityClass
