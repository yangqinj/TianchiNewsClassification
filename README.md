#新闻文本分类

阿里天池新闻分类新人赛。

## 数据分析

标签总共有14个，每条新闻平均1200个词语。


## fasttext

随机抽样2万个样本，保留全部词语，词汇表共7k+个词语，全部语料有18M词语。使用5折交叉训练评估模型效果，线程数使用默认值（代码中默认为cpu个数-1，即23个线程），训练时间花费了大约10分钟，cpu使用率大概200%。学习率最终降低到0，loss最终平均达到0.1左右，acc和f1都达到了0.84~0.88左右。标签准确率最差的也有0.75，最好的有0.95。

但是论文中使用20个线程训练数据集，只花费了几秒钟的时间，与实际测试效果差别太大，因此尝试了不同的线程数，结果如下：（后来发现是机器的配置信息选择的2核，所以只有最多只能跑2个线程）

| 线程数 | 训练时间(m) | CPU使用率(%) | 每秒每个线程处理单词数 |
| ------ | ----------- | ------------ | ---------------------- |
| 23     | 10+         | ~200         | 2w+                    |
| 20     | 10          | ~200         | 5w+                    |
| 15     | 4           | ~200         | 14w+                   |
| 10     | 3+          | ~200         | 22w+                   |
| 5      | 3           | ~200         | 45w+                   |
| 2      | 2           | ~200         | 170w+                  |
| 1      | 3           | ~100         | 2258350                |



完整的20w数据集训练使用了2个线程，总共166M词语，花费时间大概20分钟。词语维度为200，初始lr为1.0，lr更新频率为200。预测结果提交后F1值为`0.9107`。

前期数据分析的时候，发现`3750, 900, 648`这三个字符很可能是标点符号，因此增加了预处理环节去除这三个字符。重新训练后，提交结果F1值为`0.9110`。观察到学习率在接近完成的时候仍然有0.03左右，且最终的loss值达到了0.08，因此尝试将epoch提高到30，lr更新频率提高到230。最终，loss值降低到大概0.07，但是测试集上提交后F1值降低到`0.9084`。



## 词向量训练

因为新闻文本被匿名处理为字符编号，因此需要自己训练词向量。词向量的采用了word2vec向量，维度为200。训练语料使用了预处理后的训练集和测试集。其他参数都使用了默认值，整个训练过程花费了半个小时。

## TextCNN

初始学习率设置为0.001，train loss虽然总体在下降但是十分震荡，而且最多下降到0.3左右就无法再减少，而dev loss则一直非常平稳而且很早趋于平滑。

| ![image-20201123204620183](/Users/yangqj/Documents/Documents/知识就是力量/OnlineCourse/jingdong_nlp/Note_md/figs/image-20201123204620183-6141108.png) | ![image-20201123204659754](/Users/yangqj/Documents/Documents/知识就是力量/OnlineCourse/jingdong_nlp/Note_md/figs/image-20201123204659754-6141119.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |



根据train loss的变化情况，尝试使用Step学习率衰减，每经过100个step则将学习率减少为原来的0.95。可以看到，train loss的震荡程度略微缓解，但是总体依然震荡；dev loss的收敛时间大大减少。这说明使用学习率衰减对训练有所改善，但是改善程度不高。

| ![image-20201123210221654](/Users/yangqj/Documents/Documents/知识就是力量/OnlineCourse/jingdong_nlp/Note_md/figs/image-20201123210221654-6141149.png) | ![image-20201123210240127](/Users/yangqj/Documents/Documents/知识就是力量/OnlineCourse/jingdong_nlp/Note_md/figs/image-20201123210240127-6141157.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

尝试降低学习的初始学习率为0.0001，发现train loss整体变大，但是下降的趋势以及震荡的趋势是相似的。dev loss收敛的时间变得很久。这说明学习率偏小，学习率衰减使得学习率又不断变得更小，因此学习率设置为0.001是比较合适的。

| ![image-20201123204914426](/Users/yangqj/Documents/Documents/知识就是力量/OnlineCourse/jingdong_nlp/Note_md/figs/image-20201123204914426-6141174.png) | ![image-20201123204932018](/Users/yangqj/Documents/Documents/知识就是力量/OnlineCourse/jingdong_nlp/Note_md/figs/image-20201123204932018-6141180.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |



考虑到新闻文本的长度，尝试将batch size从64增加到128，使得每次训练有更多的数据和特征。可以观察到增加batch size之后train loss和dev loss的最低值都有所改善，而且train loss的震荡有所缓解。

| ![image-20201123211401808](/Users/yangqj/Documents/Documents/知识就是力量/OnlineCourse/jingdong_nlp/Note_md/figs/image-20201123211401808-6141200.png) | ![image-20201123211420391](/Users/yangqj/Documents/Documents/知识就是力量/OnlineCourse/jingdong_nlp/Note_md/figs/image-20201123211420391-6141205.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

继续增加batch size到256，可以观察到有很大的提升：train loss的震荡问题有极大改善，而且loss总体下降了很大，最低大概了0.2。

| ![image-20201123214243156](/Users/yangqj/Documents/Documents/知识就是力量/OnlineCourse/jingdong_nlp/Note_md/figs/image-20201123214243156-6141219.png) | ![image-20201123214259279](/Users/yangqj/Documents/Documents/知识就是力量/OnlineCourse/jingdong_nlp/Note_md/figs/image-20201123214259279-6141225.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

考虑到新闻涉及到的类别中“房产”楼盘名称、“教育”学校名称、“股票”名称等可能存在五个字的词语，因此大小为5的卷积核。从实验结果中可以看到，不管是训练集还是验证集都略有所改善。考虑到训练数据集太大，而改善的程度比较小，因此全量训练时并没有使用5-gram。

| ![image-20201123220420462](/Users/yangqj/Documents/Documents/知识就是力量/OnlineCourse/jingdong_nlp/Note_md/figs/image-20201123220420462-6141252.png) | ![image-20201123220436948](/Users/yangqj/Documents/Documents/知识就是力量/OnlineCourse/jingdong_nlp/Note_md/figs/image-20201123220436948-6141259.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |



使用全量数据集进行训练，花费了2个小时。训练过程的loss值如下：

| ![image-20201124233359733](/Users/yangqj/Documents/Workspace/AliTianchi/新闻分类/NewsClassification/figs/image-20201124233359733.png) | ![image-20201124233416804](/Users/yangqj/Documents/Workspace/AliTianchi/新闻分类/NewsClassification/figs/image-20201124233416804.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20201124233604485](/Users/yangqj/Documents/Workspace/AliTianchi/新闻分类/NewsClassification/figs/image-20201124233604485.png) | ![image-20201124233617545](/Users/yangqj/Documents/Workspace/AliTianchi/新闻分类/NewsClassification/figs/image-20201124233617545.png) |

可以看到训练时候的loss值还是比较震荡额，而且总体来说下界还是只到达了0.3左右，不过最低值达到了0.2；相比于2w的数据量loss和F1值都有所下降。验证集上的最终最终结果有所提升，验证集上的F1值甚至将近达到了0.94。提交结果F1值提升到了0.9234。根据当前的测试结果来看，loss无法继续提升的原因应该是模型的结果太简单了，导致无法完全学习长文本中的特征。

## BiLSTM

采用2层双向BiLSTM模型，后接全连接层，使用softmax完成分类。初始学习率设置为0.01，没有使用学习率衰减，batch_size设置为128，dropout为0.3。可以看到此时出现了非常明显的过拟合现象，而且训练集上的初始化loss特别大，验证集的loss相比其他模型也偏大。为了解决过拟合问题，尝试增加dropout为0.4后，过拟合现象并没有太大改善。

| ![image-20201130215217192](/Users/yangqj/Documents/Workspace/AliTianchi/新闻分类/NewsClassification/figs/image-20201130215217192.png) | ![image-20201130215238207](/Users/yangqj/Documents/Workspace/AliTianchi/新闻分类/NewsClassification/figs/image-20201130215238207.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20201130225634375](/Users/yangqj/Documents/Workspace/AliTianchi/新闻分类/NewsClassification/figs/image-20201130225634375.png) | ![image-20201130225649526](/Users/yangqj/Documents/Workspace/AliTianchi/新闻分类/NewsClassification/figs/image-20201130225649526.png) |



尝试简化模型结构从而减少参数，将BiLSTM修改为1层。可以观察到模型的学习能力大大下降，在验证集的loss增加了好多。

| ![image-20201130234212914](/Users/yangqj/Documents/Workspace/AliTianchi/新闻分类/NewsClassification/figs/image-20201130234212914.png) | ![image-20201130234231465](/Users/yangqj/Documents/Workspace/AliTianchi/新闻分类/NewsClassification/figs/image-20201130234231465.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |



