# -*- coding: utf-8 -*-

'''
1.更改程序第226行的模型评价次数（for循环次数）可以减少程序运行时间
2.XGBoost模型在程序运行后会出现一些警告（eval-error、train-error），可以忽略
3.优化算法因为运行时间较长，在程序中作为注释
4.程序运行后显示的统计图存在部分内容重叠，可以到image文件夹中查看
'''

import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

plt.rcParams['font.sans-serif'] = ['SimHei']#设置中文显示
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings("ignore")#忽略警告

#导入数据
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')
print ('训练数据集:',train.shape,'测试数据集:',test.shape)
rowNum_train=train.shape[0]
rowNum_test=test.shape[0]
print('训练数据集行数：',rowNum_train,
     ',测试数据集行数：',rowNum_test,)

#对所有特征实现独热编码



#合并数据集，方便同时对两个数据集进行清洗
full = train.append( test , ignore_index = True )
print ('合并后的数据集:',full.shape)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print (full.describe())
print("\n=====================================================\n")
print(full.info())
print("\n=====================================================\n")


#对缺失数据进行处理
full['Age']=full['Age'].fillna( full['Age'].mean() )
full['Fare'] = full['Fare'].fillna( full['Fare'].mean() )
print(full['Embarked'].head())
print(full['Embarked'].value_counts())
full['Embarked'] = full['Embarked'].fillna( 'S' )
full['Cabin'] = full['Cabin'].fillna( 'U' )
print("处理后：")
print(full.info())
print("\n=====================================================\n")
#用柱状图对比处理前后的数据
p1=plt.figure(figsize=(8,7))
a1=p1.add_subplot(1,1,1)
p = range(4)
data1=[1046,1038,1307,295]
data2=[1309,1309,1309,295]
data2=[1309,1309,1309,295]
plt.bar(x=p,height=data1,width=0.3,color='orange',label='处理前')
plt.bar(x=[i+0.3 for i in p],height=data2,width=0.3,color='blue',label='处理后')
plt.bar(x=[i+0.6 for i in p],height=data2,width=0.3,color='pink',label='理想行数')
plt.ylabel('行数')
plt.ylim(0,1500)
label1=['Age','Fare','Embarked','Cabin']
plt.xticks(range(4),label1)
plt.legend()
plt.title('数据处理前后对比图')
plt.savefig('./image/数据处理前后对比图.png')
plt.show()


#数据分类处理
#Sex
sex_mapDict={'male':1,
            'female':0}
full['Sex']=full['Sex'].map(sex_mapDict)
#Embarke
embarkedDf = pd.DataFrame()
embarkedDf = pd.get_dummies( full['Embarked'] , prefix='Embarked' )
full = pd.concat([full,embarkedDf],axis=1)
full.drop('Embarked',axis=1,inplace=True)
#Pclass
pclassDf = pd.DataFrame() 
pclassDf = pd.get_dummies( full['Pclass'] , prefix='Pclass' )
full = pd.concat([full,pclassDf],axis=1)
full.drop('Pclass',axis=1,inplace=True)

#Nmae
def getTitle(name):
    str1=name.split( ',' )[1] #Mr. Owen Harris
    str2=str1.split( '.' )[0]#Mr
    #strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    str3=str2.strip()
    return str3
titleDf = pd.DataFrame()#存放提取后的特征
titleDf['Title'] = full['Name'].map(getTitle)#map函数：对Series每个数据应用自定义的函数计算
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }

#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = titleDf['Title'].map(title_mapDict)
#使用get_dummies进行one-hot编码
titleDf = pd.get_dummies(titleDf['Title'])
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,titleDf],axis=1)
full.drop('Name',axis=1,inplace=True)#删掉姓名这一列

#Cabin
cabinDf = pd.DataFrame()
full[ 'Cabin' ] = full[ 'Cabin' ].map( lambda c : c[0] )
##使用get_dummies进行one-hot编码，列名前缀是Cabin
cabinDf = pd.get_dummies( full['Cabin'] , prefix = 'Cabin' )
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,cabinDf],axis=1)
full.drop('Cabin',axis=1,inplace=True)#删掉客舱号这一列

familyDf = pd.DataFrame()
familyDf[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1
#if 条件为真的时候返回if前面内容，否则返回0
familyDf[ 'Family_Single' ] = familyDf[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
familyDf[ 'Family_Small' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
familyDf[ 'Family_Large' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
full = pd.concat([full,familyDf],axis=1)
print(full.head())
print("\n=====================================================\n")


#相关性矩阵
corrDf = full.corr()
#查看各个特征与生成情况（Survived）的相关系数，ascending=False表示按降序排列
data=corrDf['Survived'].sort_values(ascending = False)
print(data)
data.to_csv( './temp/temp.csv' , index = True )#保存结果
filename='./temp/temp.csv'
with open(filename,'r')as file:
    #1.创建阅读器对象
    reader=csv.reader(file)
    #2.读取文件头信息
    header_row=next(reader)
    chara,hights=[],[]
    for row in reader:
        chara.append(row[0])
        hights.append((row[1]))
        
#绘制折线图
p2=plt.figure(figsize=(32,9))
y0=p2.add_subplot(1,1,1)
plt.plot(chara,hights,c='red')
plt.ylim(32,0)
plt.xlabel('特征')
plt.ylabel('相关系数')
plt.title('特征与标签的相关系数折线图')
plt.savefig('./image/特征与标签的相关系数折线图.png')
plt.show()

#特征选择
print("\n=====================================================\n")
full_X = pd.concat( [titleDf,#头衔
                     pclassDf,#客舱等级
                     familyDf,#家庭大小
                     full['Fare'],#船票价格
                     cabinDf,#船舱号
                     embarkedDf,#登船港口
                     full['Sex']#性别
                    ] , axis=1 )
print(full_X.describe())


#模型构建
sourceRow =891
source_X = full_X.loc[0:sourceRow-1,:]#原始数据集：特征
source_y = full.loc[0:sourceRow-1,'Survived']#原始数据集：标签   
pred_X = full_X.loc[sourceRow:,:]#预测数据集：特征
print("\n=====================================================\n")
print('原始数据集有多少行:',source_X.shape[0])#确保这里原始数据集取的是前891行的数据
print('预测数据集有多少行:',pred_X.shape[0])

#建立模型用的训练数据集和测试数据集
train_X, test_X, train_y, test_y = train_test_split(source_X ,
                                                    source_y,
                                                    train_size=0.8)
#输出数据集大小
print ('原始数据集特征：',source_X.shape, 
       '训练数据集特征：',train_X.shape ,
      '测试数据集特征：',test_X.shape)
print ('原始数据集标签：',source_y.shape, 
       '训练数据集标签：',train_y.shape ,
      '测试数据集标签：',test_y.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

scores1=[]
scores2=[]
scores3=[]
scores4=[]
scores5=[]
scores6=[]
scores7=[]

'''
如果需要缩短程序运行时间，请将150改为更小的数
'''
for i in range(10):
    
   train_X, test_X, train_y, test_y = train_test_split(source_X ,
                                                    source_y,
                                                    train_size=0.8)
   
   #逻辑回归
   model = LogisticRegression()
   model.fit( train_X , train_y )
   score=model.score(test_X , test_y )
   scores1.append(score)

   #随机森林分类
   model = RandomForestClassifier(n_estimators=100)
   model.fit( train_X , train_y )
   score=model.score(test_X , test_y )
   scores2.append(score)

   #支持向量机分类
   model = SVC()
   model.fit( train_X , train_y )
   score=model.score(test_X , test_y )
   scores3.append(score)

   #支持向量机线性分类
   model = LinearSVC(max_iter=2000)
   model.fit( train_X , train_y )
   score=model.score(test_X , test_y )
   scores4.append(score)

   #梯度提升分类
   model = GradientBoostingClassifier()
   model.fit( train_X , train_y )
   score=model.score(test_X , test_y )
   scores5.append(score)

   #KNN临近算法模型
   model = KNeighborsClassifier(n_neighbors = 3)
   model.fit( train_X , train_y )
   score=model.score(test_X , test_y )
   scores6.append(score)

   #朴素贝叶斯模型
   model = GaussianNB()
   model.fit( train_X , train_y )
   score=model.score(test_X , test_y )
   scores7.append(score)
   if i%10==0:
       print('评价次数：',i)

score_log=sum(scores1) / len(scores1)
score_RFC=sum(scores2) / len(scores2)
score_SVC=sum(scores3) / len(scores3)
score_SVCLi=sum(scores4) / len(scores4)
score_GBC=sum(scores5) / len(scores5)
score_KNN=sum(scores6) / len(scores6)
score_BAY=sum(scores7) / len(scores7)

print('逻辑回归模型的最终评分为：',score_log)
print('随机森林分类模型的最终评分为：',score_RFC)
print('支持向量机分类模型的最终评分为：',score_SVC)
print('支持向量机线性分类模型的最终评分为：',score_SVCLi)
print('梯度提升分类模型的最终评分为：',score_GBC)
print('KNN临近算法模型的最终评分为：',score_KNN)
print('朴素贝叶斯模型的最终评分为：',score_BAY)


'''
#参考网络上的方法用网格搜索自动化选取最优参数，用网格搜索得到的最优参数是n_estimators = 46，max_depth = 6
#这个算法运行时间较长，故作为注释
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
pipe=Pipeline([('select',SelectKBest(k=20)), 
               ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])

param_test = {'classify__n_estimators':list(range(20,50,2)), 
              'classify__max_depth':list(range(3,60,3))}
gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10)
gsearch.fit(train_X,train_y)
print(gsearch.best_params_, gsearch.best_score_)
'''


model1 = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 46,
                                  max_depth = 6, 
                                  max_features = 'sqrt')
model1.fit( train_X , train_y )
pred_Y = model1.predict(pred_X)
pred_Y=pred_Y.astype(int)#乘客id
passenger_id = full.loc[sourceRow:,'PassengerId']#数据框：乘客id，预测生存情况的值
predDf = pd.DataFrame( 
    { 'PassengerId': passenger_id , 
     'Survived': pred_Y } )
predDf.to_csv( './Titanic_Predict_RandomForestClassifier_submission6852.csv' , index = False )#保存结果


model2 = LogisticRegression(C=0.9,max_iter=200,penalty='l2',solver='liblinear',tol=0.0001)
model2.fit( train_X , train_y )
pred_Y = model2.predict(pred_X)
pred_Y=pred_Y.astype(int)#乘客id
passenger_id = full.loc[sourceRow:,'PassengerId']#数据框：乘客id，预测生存情况的值
predDf = pd.DataFrame( 
    { 'PassengerId': passenger_id , 
     'Survived': pred_Y } )
predDf.to_csv( './Titanic_Predict_LogisticRegression_submission6852.csv' , index = False )#保存结果


model3 = GradientBoostingClassifier()
model3.fit( train_X , train_y )
pred_Y = model3.predict(pred_X)
pred_Y=pred_Y.astype(int)#乘客idpassenger_id = full.loc[sourceRow:,'PassengerId']#数据框：乘客id，预测生存情况的值
predDf = pd.DataFrame( 
    { 'PassengerId': passenger_id , 
     'Survived': pred_Y } )
predDf.to_csv( './Titanic_Predict_GradientBoostingClassifier_submission6852.csv' , index = False )#保存结果




data_train = xgb.DMatrix(train_X, label=train_y)
data_test = xgb.DMatrix(test_X, label=test_y)
watch_list = [(data_test, 'eval'), (data_train, 'train')]
param = {'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic'}
         # 'subsample': 1, 'alpha': 0, 'lambda': 0, 'min_child_weight': 1}
model4 = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)
pred_X = full_X.loc[rowNum_train:,:]
pred_X=xgb.DMatrix(pred_X)
y_hat = model4.predict(pred_X)
y_hat[y_hat > 0.5] = 1
y_hat[~(y_hat > 0.5)] = 0
y_hat=y_hat.astype(int)
passenger_id=passenger_id.astype(int)
passenger_id = full.loc[rowNum_train:,'PassengerId']#数据框：乘客id，预测生存情况的值
predDf = pd.DataFrame( 
    { 'PassengerId': passenger_id , 
     'Survived': y_hat} )
predDf.shape
predDf.head()
predDf.to_csv( './Titanic_Predict_eXtremeGradientBoosting_submission6852.csv' , index = False )#保存结果




train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
data_df = train_df.append(test_df)
data_df['Title'] = data_df['Name']

for name_string in data_df['Name']:
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)


mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
           'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data_df.replace({'Title': mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute


train_df['Age'] = data_df['Age'][:891]
test_df['Age'] = data_df['Age'][891:]


data_df.drop('Title', axis = 1, inplace = True)
data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']


train_df['Family_Size'] = data_df['Family_Size'][:891]
test_df['Family_Size'] = data_df['Family_Size'][891:]
data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])
data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                            'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:",
      data_df.loc[data_df['Family_Survival']!=0.5].shape[0])
for _, grp_df in data_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passenger with family/group survival information: "
      +str(data_df[data_df['Family_Survival']!=0.5].shape[0]))


train_df['Family_Survival'] = data_df['Family_Survival'][:891]
test_df['Family_Survival'] = data_df['Family_Survival'][891:]
data_df['Fare'].fillna(data_df['Fare'].median(), inplace = True)


data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)

label = LabelEncoder()
data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])

train_df['FareBin_Code'] = data_df['FareBin_Code'][:891]
test_df['FareBin_Code'] = data_df['FareBin_Code'][891:]

train_df.drop(['Fare'], 1, inplace=True)
test_df.drop(['Fare'], 1, inplace=True)
data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)

label = LabelEncoder()
data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])

train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:891]
test_df['AgeBin_Code'] = data_df['AgeBin_Code'][891:]

train_df.drop(['Age'], 1, inplace=True)
test_df.drop(['Age'], 1, inplace=True)
train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
test_df['Sex'].replace(['male','female'],[0,1],inplace=True)

train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
               'Embarked'], axis = 1, inplace = True)
test_df.drop(['Name','PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
              'Embarked'], axis = 1, inplace = True)
train_df.head(3)
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']
X_test = test_df.copy()
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)
n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size,
               'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True,
                cv=10, scoring = "roc_auc")
gd.fit(X, y)
print(gd.best_score_)
print(gd.best_estimator_)
gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(X)
knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
                           metric_params=None, n_jobs=1, n_neighbors=6, p=2,
                           weights='uniform')
knn.fit(X, y)
y_pred = knn.predict(X_test)
temp = pd.DataFrame(pd.read_csv("data/test.csv")['PassengerId'])
temp['Survived'] = y_pred
temp.to_csv('./Titanic_Predict_KNN_submission6852.csv', index = False)
