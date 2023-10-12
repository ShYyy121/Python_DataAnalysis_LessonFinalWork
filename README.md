# Python数据分析课程大作业-kaggle泰坦尼克号幸存者预测
## 问题简介、目标
RMS泰坦尼克号的沉没是人尽皆知的悲剧，1912年4月15日，在她的处女航中，泰坦尼克号在与冰山相撞后沉没，在2224名乘客和机组人员中造成1502人死亡。造成海难失事的原因之一是乘客和机组人员没有准备足够的救生艇。尽管成为幸存者需要一定的运气因素，但有些人比其他人更容易生存，比如女人，孩子和上流社会。
此报告将会综合分析乘客的各类特征对幸存与否的影响，通过机器学习建立并选择最优模型，对test数据集中乘客的生存进行预测，预测哪些乘客可以幸免于难，生成csv文件，最后提交结果。

## 问题分析流程设计
构建思维导图，方便富有逻辑、清晰直观地分析和解决问题。总体分析问题、解决问题的步骤和方案如图1所示。
图1：Titanic问题分析解决方案思维导图
![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/46bcd2a0-5ca7-4e5c-8775-2b9f06e64d35)

## 问题分析实现步骤
4.1 数据集初步探索
1. 数据集获取
注册kaggle账号，并从Titanic比赛中下载数据。数据下载页面的链接如下：
https://www.kaggle.com/c/titanic/data 
共下载了gender_submission.csv、train.csv、test.csv三个文件。
2.数据集基本情况分析

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/e7bff48c-ebe4-476c-b172-cb24eea4f335)

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/a765bd8a-e070-46ce-afab-ba8991ccf484)

图2、图3：数据集的基本情况
由图2和图3可知数据集初步可分为11个特征和1个标签，且各特征值的总和、均值、标准差、最值、上下四分位点均在图中显示。查看变量类型以及数量，分析得：
①full数据集共1309行；
②浮点型变量有3个，整型变量4个，字符型5个；
③Survived列为标签，1代表获救，0代表遇难；
④Age\Cabin\Embarked\Fare\Name\Parch\PassengerId\Pclass\Sex\SibSip\Ticket共11列为特征；
⑤Age\Cabin\Embarked\Fare数据有缺失。
再对数据的缺失情况进行分析：
①年龄（Age）里面数据总数是1046条，缺失了1309-1046=263，缺失率263/1309=20%，缺失较多；
②船票价格（Fare）里面数据总数是1308条，缺失了1条数据；
③登船港口（Embarked）里面数据总数是1307，只缺失了2条数据，缺失较少；
④船舱号（Cabin）里面数据总数是295，缺失了1309-295=1014，缺失率=1014/1309=77.5%，缺失较大。

4.2 数据预处理
4.2.1缺失特征数据处理
前面对缺失数据的分析，可以让我们数据预处理时针对不同的数据做出不同的处理，最终使得数据集中不再用空值。处理的方法如下：
①如果是数值类型，用平均值取代；
②如果是分类数据，用最常见的类别取代（出现次数最多的类别，众数）；
③使用模型预测缺失值，例如：K-NN。
所以，我们对缺失的数据分别做出以下处理：
①平均值填充Age；②平均值填充Fare；

     ![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/985572d8-07d3-42c3-9639-299021dac8ab)

图4：登船港口（Embarked）的数据基本情况
③数填充Embarked，从结果来看，S类别最常见。将缺失值填充为最频繁出现的值。另外，通过资料可知出发地点：S=英国-南安普顿Southampton，途径地点1：C=法国-瑟堡市Cherbourg，途径地点2：Q=爱尔兰-昆士敦Queenstown。
因为船舱号（Cabin）的缺失数据较多，且通过查看Cabin的数据情况，发现Cabin有非常多不同的船舱，如果用众数填充会影响数据的准确性，故填充为U，表示未知（Uknow）。最后查看处理前后的数据情况。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/0d747f4a-f76f-4d65-b6bd-68072e92b874)

图5：数据处理前后对比柱状图

4.2.2特征提取并转化
完成上述工作后，要对特征进行提取并进行转化。首先对Sex和Embarked和Pclass三个特征进行哑变量处理。
对于Sex，将性别的值映射为数值，男（male）对应数值1，女（female）对应数值0。
对于有直接类别的登船港口（Embarked），使用get_dummies方法进行one-hot编码，产生虚拟变量（dummy variables），列名前缀是Embarked。因为已经使用登船港口（Embarked）进行了one-hot编码产生了它的虚拟变量（dummy variables），所以这里把登船港口（Embarked）删掉。
对于有有直接类别的客舱等级（Pclass），处理方法与登船港口（Embarked）相同。处理完之后删客舱等级（Pclass）这一列。
另外，通过观察和参考网上的相关资料，发现在乘客名字（Name）中，有一个非常显著的特点，即乘客头衔每个名字当中都包含了具体的称谓或是头衔，将这部分信息提取出来后可以作为非常有用一个新变量，可以帮助我们进行预测。例如：例如：Braund, Mr. Owen Harris、Heikkinen, Miss. Laina、Oliva y Ocana, Dona. Fermina、Peter, Master. Michael J。所以这里需要定义一个函数，即从姓名中获取头衔。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/3ca6be1d-24b6-431b-b04e-0cf886fe3bcf)

图6：定义的从姓名中获取头衔的方法
之后，根据乘客的称谓或头衔的数据信息，定义定义以下几种头衔类别：Officer政府官、Royalty王室（皇室）、Mr已婚男士、Mrs已婚妇女、Miss年轻未婚女子、Master有技能的人。同上面的操作。进行one-hot编码后删除名字（Name）这一列。
接下来对客舱号（Cabin）进行处理，客舱号的类别值是首字母，根据首字母对客舱号进行one-hot编码，之后删除客舱号（Cabin）这一列。
另外，我们还可以建立家庭人数和家庭类别，家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己，因为乘客自己也是家庭成员的一个，所以家庭人数最后要加1。再定义家庭类别：①小家庭Family_Single：家庭人数=1；②中等家庭Family_Small: 2<=家庭人数<=4；
③大家庭Family_Large: 家庭人数>=5。同样地对家庭类别进行one-hot编码，因为原数据中并没有家庭类别这一项，所以进行编码后要将家庭类别添加到数据集中。
4.2.3特征值优化
接下来我们需要查看各特征值和标签Survived间的相关指数，再选取相关性高的特征进行降维。这里使用corr方法计算完各个特征的相关系数后，为了方便对比，可以进行降序排列后，再通过折线图显示，可以更为直观。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/efbce8e5-211b-4c45-a105-61e6f137ea4f)
![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/d46c719c-8282-4733-b6d5-6bd24794fb4f)


图7、8特征值和标签的相关指数计算结果和折线图
在绘制折现图时，存在一些问题，一是因为折线图没有x轴的坐标是特征的名称，没有具体的数值作为对应，所以画出来的直线图只能是一条直线。虽然可以看出y轴的特征值下降是非线性的，但无法通过折线图很好地体现出来。另外，因为折线图的数据是通过sort_values方法降序排列得到的，无法直接像数据分析实验一种那样直接读取.npz文件的数据，在这里尝试了将数据保存为.csv文件再按行读取到程序中，将数据存储到列表中，再绘制折线图。此处的问题明显是因为对python数据读取、matplotlib等掌握不够熟练，以后需要多加使用和练习。
相关系数的绝对值越大，说明该特征和Survived间的相关性越高（正负表示正相关和负相关），所以根据特征值的相关系数的情况，选取titleDf（头衔）、pclassDf（客舱等级）、familyDf（家庭大小）、Fare（船票价格）、cabinDf（船舱号）、embarkedDf（登船港）、Sex（性别）作为特征。如图所示。选择完特征后使用describe方法查看数据集的情况。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/ad8b273d-9f1f-42e8-a071-e6efbe9c0b5d)

图9根据相关系数选择特征
4.3 构建分析模型及预测

1. 模型构建
学习是通过训练train数据集来拟合回归方程，预测是用学习过程中拟合出的回归方程，运用与test数据集中求出预测值。本次实验原始提供了三个数据文件train.csv、test.csv和gender_submission.csv。其中train.csv是训练数据集，用于模型训练，test.csv是测试数据集，用于模型评估。gender_submission.csv则是一个提交文件的示例。我们知道原始数据集有总共有891条数据，从特征集合full_X中提取原始数据集提取前891行数据时，我们要减去1，因为行号是从0开始的。代码如图所示。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/78c26a0b-8ffd-49c9-8760-d0c4d74dd2b7)

图10从特征集合中提取数据用于模型构建
随后，利用sklearn的model_selection提供的train_test_split函数对数据集进行拆分。source_X所要划分的样本特征集，source_y所要划分的样本结果,，train_size就是样本占比，如果是整数的话即为样本的数量。随后数据数据集的大小。如下图所示。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/7f4c0550-a78f-4a3c-8519-11ac764599aa)

图11建立模型用的训练数据集和测试数据集

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/9a0d2cfb-3260-4d94-be2b-254c421a133f)


图12数据集的大小情况
2. 模型验证
因为之前所学习的相关评价模型的指标，例如评价回归模型的平均绝对误差、均方误差、中值绝对误差、可解释方差和R方值，评价聚类模型的ARI评价法、AMI评价法、V-measures评分、FMI评价法、轮廓系数评价法和Calinski-Harabasz指数等，评价分类模型的精确率、召回率、F1值、Cohen’s Kappa系数和ROC曲线等，有需要有预测数值和真实值进行对比，而本题的数据无法满足需要，在跟老师讨论过后，决定采用横向验证的方法，即使用多个模型或算法，对比不同模型和算法的评分，最终确定的一个最优的模型或算法。
横向验证共选取了LogisticRegression（逻辑回归模型）、RandomForestClassifier（随机森林分类模型）、SVC（支持向量机分类模型）、LinearSVC（支持向量机线性分类模型）、GradientBoostingClassifier（梯度提升分类模型）、KneighborsClassifier（KNN临近算法模型）和GaussianNB（朴素贝叶斯模型）共七种模型。分别对模型进行训练后，再用model.score方法对其进行打分。最后比较每个模型的分数。七段代码非常相似，这里仅截取一段作为示例。scores是储存每个模型评分的列表，方便后面绘制图表进行比较。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/d39fd163-ad3c-415b-b0f6-6212464dc314)


图13对模型进行训练和打分
3. 模型评价
运行几遍程序后，发现每次对模型的评价分数是不固定的，会有上下浮动，所以决定对单个模型多次进行评价，获得多个评价分数，最终对每个模型的评价分数均值进行比较。代码如下。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/7be95c31-6162-4322-96b5-10eb698c0eb8)

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/33e89306-ccd3-4dd2-b71d-73df52544e1c)


图14、15对模型进行多次评价并求取得分平均值
另外，在运行过程中，LinearSVC（支持向量机线性分类模型）会出现“Liblinear failed to converge, increase”警告，意为Liblinear无法收敛，增加了迭代次数。通过增加max_iter（迭代次数）等方法仍无法解决，但此警告对模型评分并没有明显影响，为了保证程序运行结果的显示，导入warnings包后用warnings.filterwarnings("ignore")语句忽略此警告。
进行评价的次数在for循环开始处可以修改，本次实验共进行了150次评价，最终每个模型的平均得分如图所示。可以看出LogisticRegression（逻辑回归模型）、RandomForestClassifier（随机森林分类模型）和GradientBoostingClassifier（梯度提升分类模型）得分较高



![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/e7137e64-616b-4970-b3de-3d9cb8846366)

图16 每个模型的平均得分

4. 预测
根据模型评价的结果，选取平均分较高的LogisticRegression（逻辑回归模型）、RandomForestClassifier（随机森林分类模型）和GradientBoostingClassifier（梯度提升分类模型）。同时，在预测前，参考了网上查阅的相关模型优化算法，获得了RandomForestClassifier（随机森林分类模型）优化的参数，原本还想继续获得LogisticRegression（逻辑回归模型）的参数，但因为模型优化方面了解还不够全面，最终没有成功。模型优化在机器学习和深度学习等方面非常重要，仍然需要后续的学习和实践，这里不再详细赘述，相关算法因为运行时间较长，在代码中作为注释。
另外，在进行预测工作时，又接触到了XGBoost（eXtremeGradientBoosting极端梯度提升模型），其中y_hat表示的是为1的概率，即生还的概率，也同样作为选择的模型之一。XGBoost模型程序在运行后会出现一些警告，可以忽略。
之后输入参数后进行预测。因为生成的预测值是浮点数，范围是0.0到1,0，但是Kaggle网站要求提交的结果是整型，范围是0到1，所以要对数据类型进行转换。之后保存预测的数据文件并提交。另外，在网上查阅资料时还接触到了模型融合，也可以在以后进行学习，在图19中又查阅补充对数据处理、特征的提取和优化方面重新进行了修改和优化，利用GridSearchCV等方法重新优化数据的预处理，最后利用KNN模型预测结果。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/a06189b2-1380-4c7d-be09-d5504f699cc7)

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/34f8f5cd-a944-49bd-acfc-6ae7b7699794)

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/b3a339f8-f0e8-4329-9c8c-dede5fcbdd19)

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/fb29f0d4-97c7-4627-a047-89e73a0ef879)


图17 、18、19生成预测结果的代码

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/86d0374e-3aa9-41fc-a926-e78f7730e6eb)

图20为补充对数据预处理进行优化的代码
## 问题分析结果
（多次运行程序得到预测结果并提交，发现分数存在波动，此处为目前得到的最好成绩）

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/92f4aefb-8362-495e-a6a0-a0362249895f)

GradientBoostingClassifier（梯度提升分类模型）

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/1604c9da-0fd6-427b-a3f1-6db3b6d2917f)

eXtremeGradientBoosting（极端梯度提升模型）

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/2d761a9a-aad3-40b7-92b3-cc9056aff438)

LogisticRegression（逻辑回归模型）

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/9f91b788-cbf3-495b-acf3-309cb84e87a4)

RandomForestClassifier（随机森林分类模型）

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonFinalWork/assets/145829122/b050fc0e-2df6-4d65-951f-e3e16bb11016)

重新优化后K-Nearest Neighbor（最邻近算法模型）
可以看出五种模型的得分都不是很高，主要问题还是出在对模型的调参优化、对数据集的处理方案不够、对特征的提取和优化等方面，模型的调参优化可以很好地提升模型的性能和预测的准确度，另外还可以采用模型融合的方式提高预测的准确度。同时，实验的前半部分对数据的预处理过程，以及对特征值提取、转化、优化都存在不足和可改进之处。
还有，模型构建、评估和验证方面也存在一些不足，因为对交叉验证的使用不熟练，以及受到实验数据的限制.上述几点也是本实验可以改进和提升的地方，需要在今后不断学习与实践。
