### 总结

这次实验中，学习并实践使用了SVM进行训练与预测，并与前面学习的KNN与贝叶斯方法以及逻辑回归进行了性能比较。

#### 一、make_circle()的分类
| 分类方法 | 正确率 |
| :-------------: | :-------: |
|KNN |100%|
|NaiveBayes| 99.4% |
|Logistic| 50.4% |
|SVM |100%|

这里通过make_circle数据集，将四种分类方法进行了可视化，由于逻辑回归的特性，其正确率在50%左右；

其他分类方法均能达到接近100%的分类正确率。


#### 二、三个常用图像数据集的分类性能比较

| 分类方法\数据集 | 17flowers | Digits | Face images |
| :-------------: | :-------: | :----: | :---------: |
|       KNN       |  20.22%   | 98.89% |   100.0%    |
|   NaiveBayes    |  29.78%   | 82.22% |   100.0%    |
|LogisticRegression|  29.04%  | 97.22% |   100.0%    |
|SVM              |  36.96%  | 99.44% |   100.0%    |

在分类准确率方面，SVM相较其他三种分类方法全方面胜出，这还是在使用默认参数未进行调参的情况下。

接下来是时间消耗：

| 分类方法\数据集 | 17flowers | Digits | Face images |
| :-------------: | :-------: | :----: | :---------: |
|       KNN       |  0.031s \| 0.812s  | 0.001s \| 0.015s |  0.001s \| 0.033s  |
|   NaiveBayes    |  0.283s \| 1.45s  | 0.001s \| 0.001s |   0.059s \| 0.101s   |
|LogisticRegression|  14.9s \| 0.035s  | 0.096s \| 0.001s |   1.97s \| 0.004s   |
|SVM        |  36.4s \| 11.31s  | 0.035s \| 0.024s |   0.312s \| 0.102s   |

SVM的速度随数据集大小的提升，训练以及预测消耗的时间上升率要大于其他三个分类方法（非线性上升）。

#### 二、cifar10的三个批次性能比较

各批次准确率如下：

| 分类方法\数据集 | batch_1   | batch_2| batch_3     |
| :-------------: | :-------: | :----: | :---------: |
|       KNN       |  28.97%   | 28.88% |   30.28%    |
|   NaiveBayes    |  29.29%   | 29.85% |   29.58%    |
|LogisticRegression|  37.17%  | 37.44% |   36.73%    |
|	SVM	|  47.16%  | 47.59% | 47.08% |

时间方面，SVM的预测消耗和训练消耗均达到了百秒级别(100s~300s之间)，远大于其他三种分类方法，但另一方面其准确率又是最高的。

（其他时间参考上节内容）
