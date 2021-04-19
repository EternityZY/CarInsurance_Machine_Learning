#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 13:54
# @Author  : GXl
# @File    : 3.3.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import operator
import numpy as np
from copy import copy
from numpy import ndarray, log, apply_along_axis
import math
"""
Parameters:
    dataSet - 数据集
Returns:
    shannonEnt - 经验熵(香农熵)
"""


# 函数说明:计算给定数据集的经验熵(香农熵)
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)  # 返回数据集的行数
    labelCounts = {}  # 保存每个标签(Label)出现次数的字典
    for featVec in dataSet:  # 对每组特征向量进行统计
        currentLabel = featVec[-1]  # 提取标签(Label)信息
        if currentLabel not in labelCounts.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # Label计数
    shannonEnt = 0.0  # 经验熵(香农熵)
    for key in labelCounts:  # 计算香农熵
        prob = float(labelCounts[key]) / numEntires  # 选择该标签(Label)的概率
        shannonEnt -= prob * math.log(prob, 2)  # 利用公式计算
    return shannonEnt  # 返回经验熵(香农熵)


"""
Parameters:
    无
Returns:
    dataSet - 数据集
    labels - 特征标签
"""


# 函数说明:创建测试数据集
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
    return dataSet, labels  # 返回数据集和分类属性


"""
Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
Returns:
    无
"""


# 函数说明:按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []  # 创建返回的数据集列表
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet  # 返回划分后的数据集


"""
Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值
"""


# 函数说明:选择最优特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    bestInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征的索引值
    for i in range(numFeatures):  # 遍历所有特征
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建set集合{},元素不可重复
        newEntropy = 0.0  # 经验条件熵
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy  # 信息增益
        # print("第%d个特征的增益为%.3f" % (i, infoGain))   #打印每个特征的信息增益
        if (infoGain > bestInfoGain):  # 计算信息增益
            bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature  # 返回信息增益最大的特征的索引值


"""
Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素(类标签)
"""


# 函数说明:统计classList中出现此处最多的元素(类标签)
def majorityCnt(classList):
    classCount = {}
    for vote in classList:  # 统计classList中每个元素出现的次数
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 根据字典的值降序排序
    return sortedClassCount[0][0]  # 返回classList中出现次数最多的元素


"""
Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树
"""


# 函数说明:创建决策树
def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]  # 取分类标签(是否放贷:yes or no)
    if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
    bestFeatLabel = labels[bestFeat]  # 最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
    del (labels[bestFeat])  # 删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]  # 得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)  # 去掉重复的属性值
    for value in uniqueVals:  # 遍历特征，创建决策树。
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree


"""
Parameters:
    myTree - 决策树
Returns:
    numLeafs - 决策树的叶子结点的数目
"""


# 函数说明:获取决策树叶子结点的数目
def getNumLeafs(myTree):
    numLeafs = 0  # 初始化叶子
    # python3中myTree.keys()返回的是dict_keys,不在是list,
    # 所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]  # 获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


"""
Parameters:
    myTree - 决策树
Returns:
    maxDepth - 决策树的层数
"""


# 函数说明:获取决策树的层数
def getTreeDepth(myTree):
    maxDepth = 0  # 初始化决策树深度
    # python3中myTree.keys()返回的是dict_keys,不在是list,
    # 所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]  # 获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth  # 更新层数
    return maxDepth


"""
Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
Returns:
    无
"""


# 函数说明:绘制结点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=16)  # 设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  # 绘制结点
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


"""
Parameters:
    cntrPt、parentPt - 用于计算标注位置
    txtString - 标注的内容
Returns:
    无
"""


# 函数说明:标注有向边属性值
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注位置
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=45)


"""
Parameters:
    myTree - 决策树(字典)
    parentPt - 标注的内容
    nodeTxt - 结点名
Returns:
    无
"""


# 函数说明:绘制决策树
def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
    numLeafs = getNumLeafs(myTree)  # 获取决策树叶结点数目，
    # 决定了树的宽度
    depth = getTreeDepth(myTree)  # 获取决策树层数
    firstStr = next(iter(myTree))  # 下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制结点
    secondDict = myTree[firstStr]  # 下一个字典，
    # 也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，
            # 如果不是字典，
            # 代表此结点为叶子结点
            plotTree(secondDict[key], cntrPt, str(key))  # 不是叶结点，
            # 递归调用继续绘制
        else:  # 如果是叶结点，绘制叶结点，
            # 并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


"""
Parameters:
    inTree - 决策树(字典)
Returns:
    无
"""


# 函数说明:创建绘制面板
def createPlot(inTree, max_depth=4):
    size = (200, 100)
    plt.figure(figsize=size)
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0  # x偏移
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    # plt.show()  # 显示绘制结果
    plt.savefig('DecisionTree/max_depth={:d}.png'.format(max_depth))





class Node:

    attr_names = ("prob", "left", "right", "feature", "split", "gain")

    def __init__(self, prob=None, left=None, right=None, feature=None, split=None, gain=None):
        self.prob = prob
        self.left = left
        self.right = right
        self.feature = feature
        self.split = split
        self.gain = gain

    def __str__(self):
        ret = []
        for attr_name in self.attr_names:
            attr = getattr(self, attr_name)
            # Describe the attribute of Node.
            if attr is None:
                continue
            if isinstance(attr, Node):
                des = "%s: Node object." % attr_name
            else:
                des = "%s: %s" % (attr_name, attr)
            ret.append(des)

        return "\n".join(ret) + "\n"

    def copy(self, node):

        for attr_name in self.attr_names:
            attr = getattr(node, attr_name)
            setattr(self, attr_name, attr)


class MyDecisionTree:

    def __init__(self):
        self.root = Node()
        self.depth = 1
        self._rules = None
        self.label = ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
                        'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium',
                        'Policy_Sales_Channel', 'Vintage', 'age_bin_cat', 'old_damaged']
        self.dict_tree = {}

    def __str__(self):
        ret = []
        for i, rule in enumerate(self._rules):
            literals, prob = rule

            ret.append("Rule %d: " % i + ' | '.join(
                literals) + ' => y_hat %.4f' % prob)


        return "\n".join(ret)

    @staticmethod
    def _expr2literal(expr: list) -> str:

        feature, operation, split = expr
        operation = ">=" if operation == 1 else "<"

        return "Feature%d %s %.4f" % (feature, operation, split)

    def priorTravel(self, node):
        # 决策树先序遍历，获得字典形式，方便可视化

        if node == None:
            return None
        if node.left and node.right:

            featureStr = self.label[node.feature]
            splitStr1 = '< {:.4f}'.format(node.split)
            splitStr2 = '>= {:.4f}'.format(node.split)
            tempTree = {featureStr:{}}
            tempTree[featureStr][splitStr1] = self.priorTravel(node.left)
            tempTree[featureStr][splitStr2] = self.priorTravel(node.right)

            return tempTree
        else :
            # leafStr = 'y_hat {:%.4f}'.format(node.prob)
            # leafStr = 'y_hat'
            # tempTree = {}
            # tempTree[leafStr] = '{:.4f}'.format(node.prob)
            leafStr = 'P={:.2f}'.format(node.prob)

            return leafStr


    def get_rules(self):

        # Breadth-First Search.
        que = [[self.root, []]]
        self._rules = []

        while que:
            node, exprs = que.pop(0)

            # Generate a rule when the current node is leaf node.
            if not(node.left or node.right):
                # Convert expression to text.
                literals = list(map(self._expr2literal, exprs))
                self._rules.append([literals, node.prob])

            # Expand when the current node has left child.
            if node.left:
                rule_left = copy(exprs)
                rule_left.append([node.feature, -1, node.split])
                que.append([node.left, rule_left])

            # Expand when the current node has right child.
            if node.right:
                rule_right = copy(exprs)
                rule_right.append([node.feature, 1, node.split])
                que.append([node.right, rule_right])

    @staticmethod
    def _get_info(prob_pos: float) -> float:
        """Calculate entropy of label.
        Probability:
        P(X=x_i) = p_i, i <- [1, n]
        Entropy:
        H(p) = -Sum(p_i * log(p_i)), i <- [1, n]
        Take binary classifaction for exmaple,
        Likelihood function, yi <- y, and p is a constant:
        Likelihood = Product(p^yi * (1-p)^(1-yi))
        Take the logarithm of both sides of this equation:
        L = Sum(yi * logp + (1-yi) * log(1-p))
        L / m = Sum(yi/m * logp + (1-yi) / m * log(1-p))
        L / m = Sum(p * logp + (1-p) * log(1-p))
        L / m = H(p)
        So Maximizing the Entropy equals to Maximizing the likelihood.
        """

        # According to L'Hospital's rule, 0 * log2(0) equals to zero.
        if prob_pos in (0, 1):
            entropy = 0
        else:
            prob_neg = 1 - prob_pos
            entropy = -prob_pos * log(prob_pos) - prob_neg * log(prob_neg)

        return entropy

    def _get_cond_info(self, ratio_left: float, prob_left: float, prob_right: float) -> float:
        """Calculate conditonal info of x, y
        Suppose there are k cases, conditional Probability:
        P(A = A_i), i <- [1, k]
        Entropy:
        CondInfo(X, y) = -Sum(p_i * H(y | A = A_i)), i <- [1, k]
        """

        info_left = self._get_info(prob_left)
        info_right = self._get_info(prob_right)
        return ratio_left * info_left + (1 - ratio_left) * info_right

    def _get_split_gain(self, col: ndarray, label: ndarray, split: float, info: float) -> Node:
        """Calculate the information gain of label when col is splitted into two pieces.
        Info gain:
        Gain(X, y) = Info(y) - CondInfo(X, y)
        """

        # Split label.
        left = col < split

        # Calculate ratio.
        ratio_left = left.sum() / len(col)

        # Calculate conditional information.
        prob_left = label[left].mean()
        prob_right = label[col >= split].mean()
        info_cond = self._get_cond_info(ratio_left, prob_left, prob_right)

        # Create nodes to store result.
        node = Node(split=split)
        node.gain = info - info_cond
        node.left = Node(prob_left)
        node.right = Node(prob_right)

        return node

    def _choose_split(self, col: ndarray, label: ndarray) -> Node:

        # Feature cannot be splitted if there's only one unique element.
        node = Node()
        unique = set(col)
        if len(unique) == 1:
            return node

        # In case of empty split.
        unique.remove(min(unique))

        # Calculate info
        info = self._get_info(label.mean())

        # Get split point which has max info gain.
        ite = map(lambda x: self._get_split_gain(col, label, x, info), unique)
        node = max(ite, key=lambda x: x.gain)

        return node

    def _choose_feature(self, data: ndarray, label: ndarray) -> Node:

        # Compare the mse of each feature and choose best one.
        _ite = map(lambda x: (self._choose_split(data[:, x], label), x),
                   range(data.shape[1]))
        ite = filter(lambda x: x[0].split is not None, _ite)

        # Return None if no feature can be splitted.
        node, feature = max(
            ite, key=lambda x: x[0].gain, default=(Node(), None))
        node.feature = feature

        return node

    def fit(self, data: ndarray, label: ndarray, max_depth=4, min_samples_split=2):
        """Build a decision tree classifier.
        Note:
            At least there's one column in data has more than 2 unique elements,
            and label cannot be all the same value.
        Arguments:
            data {ndarray} -- Training data.
            label {ndarray} -- Target values.
        Keyword Arguments:
            max_depth {int} -- The maximum depth of the tree. (default: {4})
            min_samples_split {int} -- The minimum number of samples required
            to split an internal node. (default: {2})
        """

        # Initialize with depth, node, indexes.
        self.root.prob = label.mean()
        que = [(self.depth + 1, self.root, data, label)]

        # Breadth-First Search.
        while que:
            depth, node, _data, _label = que.pop(0)

            # Terminate loop if tree depth is more than max_depth.
            if depth > max_depth:
                depth -= 1
                break

            # Stop split when number of node samples is less than
            # min_samples_split or Node is 100% pure.
            if len(_label) < min_samples_split or all(_label == label[0]):
                continue

            # Stop split if no feature has more than 2 unique elements.
            _node = self._choose_feature(_data, _label)
            if _node.split is None:
                continue

            # Copy the attributes of _node to node.
            node.copy(_node)

            # Put children of current node in que.
            idx_left = (_data[:, node.feature] < node.split)
            idx_right = (_data[:, node.feature] >= node.split)
            que.append(
                (depth + 1, node.left, _data[idx_left], _label[idx_left]))
            que.append(
                (depth + 1, node.right, _data[idx_right], _label[idx_right]))

        # Update tree depth and rules.
        self.depth = depth
        self.get_rules()

        self.dict_tree = self.priorTravel(self.root)
        print(self.dict_tree)
        createPlot(self.dict_tree, max_depth=max_depth)


    def predict_one_prob(self, row: ndarray) -> float:

        node = self.root
        while node.left and node.right:
            if row[node.feature] < node.split:
                node = node.left
            else:
                node = node.right

        return node.prob

    def predict_proba(self, data: ndarray) -> ndarray:

        return apply_along_axis(self.predict_one_prob, axis=1, arr=data)

    def predict(self, data: ndarray, threshold=0.5):

        prob = self.predict_proba(data)
        return (prob >= threshold).astype(int)



























if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    print(myTree)
    createPlot(myTree)
