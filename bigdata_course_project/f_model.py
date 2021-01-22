import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz
from sklearn.model_selection import GridSearchCV
import sys


def preProcess(df, test_flag, cols_train):
    """
        数据预处理：性别映射、日期转化、缺失值处理、归一化
    """
    cols = list(df.columns.values)  # 列表头
    if 'id' in cols:
        del (df['id'])  # 删除id列
        cols.remove('id')
    # print(cols)       # 显示列表头
    # print(df.dtypes)  # 显示每一列的类型

    # 筛选重复值并删除
    # df.duplicated()
    df.drop_duplicates()

    # 缺失值处理：
    # 统计每一列缺失值的个数，如果该列属性缺失值个数大于70%且是对训练集预处理，则删除该列
    # 否则填充缺失值，填充值为该列数据的平均值
    if test_flag:   # 在测试集中删掉训练集中已经被删除的属性列
        for col in cols:
            if col not in cols_train:
                cols.remove(col)

    row_num = df.shape[0]  # 获取df数据的行数
    for col in cols:
        if (df[col].isnull().sum() > 0.7 * row_num) and (not test_flag):
            del (df[col])
        else:
            df_mean = df[col].mean()
            df[col] = df[col].fillna(np.around(df_mean, 0))  # 缺失值填充为整数
    cols = list(df.columns.values)  # 因为前面有缺省过多的列表被删掉了，所以在这里更新cols

    # 对年龄、身高、孕前体重、收缩压、舒张压进行离散化（前提是他们没有因为缺失值太多而被删掉）
    if '年龄' in cols:
        bins = [16, 25, 35, 45, 55, 100]
        df['年龄'] = pd.cut(df['年龄'], bins, labels=False)
    if '身高' in cols:
        bins = [130, 150, 160, 170, 180, 190, 300]
        df['身高'] = pd.cut(df['身高'], bins, labels=False)
    if '孕前体重' in cols:
        bins = [30, 40, 50, 60, 70, 80, 90, 100, 200]
        df['孕前体重'] = pd.cut(df['孕前体重'], bins, labels=False)
    if '收缩压' in cols:
        bins = [60, 80, 100, 120, 140, 160, 180, 300]
        df['收缩压'] = pd.cut(df['收缩压'], bins, labels=False)
    if '舒张压' in cols:
        bins = [40, 50, 60, 70, 80, 90, 100, 200]
        df['舒张压'] = pd.cut(df['舒张压'], bins, labels=False)

    # 对以下类型的数据进行归一
    normalization_cols = ['RBP4', '孕前BMI', '糖筛孕周', 'VAR00007', 'wbc', 'ALT', 'AST', 'Cr',
                          'BUN', 'CHO', 'TG','HDLC', 'LDLC', 'ApoA1', 'ApoB', 'Lpa', 'hsCRP']
    for col in normalization_cols:
        if col in cols:  # 该列没有因为缺失值太多而被删掉
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    # df.to_csv('f_preprocessed.csv', encoding='gbk')
    # print(df)         # 显示表格内容


class Model():
    def __init__(self, **kwargs):
        # 模型初始化
        # 基于信息熵
        self.model = DTC(criterion='entropy', max_depth=10)

    def train(self, X, y):
        # 模型构建,用GridSearchCV寻找本实验中决策树模型的最优参数和结果
        param_grid = {'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'],
                      'max_depth': [5, 30, 50, 60, 100, None],
                      'min_samples_leaf': [2, 3, 5, 10]}
        grid = GridSearchCV(DTC(), param_grid, cv=6)
        grid.fit(X, y)
        self.model = DTC(criterion=grid.best_params_['criterion'],
                         max_depth=grid.best_params_['max_depth'],
                         min_samples_leaf=grid.best_params_['min_samples_leaf'],
                         splitter=grid.best_params_['splitter'])
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        pred_y = self.model.predict(X)
        return pred_y


def resultToCsv(pred, test):
    """
        将label预测值与实际值输出到csv文件
    """
    df = pd.DataFrame(columns=['label实际值', 'label预测值'])
    df.index.name = '编号=原始数据id号-1'
    df['label实际值'] = test['label']
    df['label预测值'] = pred
    df.to_csv('f_result.csv', encoding='gbk')
    print('Check prediction of every one in f_result.csv')


if __name__ == '__main__':
    # 读取训练集数据 train_x,train_y 并训练模型（直接读取，不需要在命令行指定）
    df_train = pd.read_csv('f_train.csv', encoding='gbk')
    preProcess(df_train, 0, [])  # 数据预处理
    print('train data pre-process finished.\n--------------------------------')
    cols_train = list(df_train.columns.values)
    cols_train.remove('label')
    X = df_train[cols_train]
    y = df_train[['label']]
    X_train, X_test, y_train, y_test = train_test_split(X, y)  # 划分测试集与训练集

    # 从命令行获取并读取测试集数据并做相应的预处理得到test_X,test_Y
    df_test = pd.DataFrame
    if len(sys.argv) > 1:
        test_file_path = sys.argv[1]
        df_test = pd.read_csv(test_file_path, encoding='gbk')
        preProcess(df_test, 1, cols_train)
        cols_test = cols_train
        X_test = df_test[cols_test]
        y_test = df_test[['label']]
        X_train = df_train[cols_test]    # 将train的x,y更新为d_train.csv中的所有数据
        y_train = df_train[['label']]

    model = Model()
    print('model training starts...')
    model.train(X_train, y_train)
    print('model training finished.\n--------------------------------')
    # 调用 predict 接口对测试数据进行预测
    print('predicting the test data')
    pred_y = model.predict(X_test)
    print('predicting complete.\n--------------------------------')
    resultToCsv(pred_y, y_test)  # 预测结果与实际结果输出到csv文件
    # 计算损失函数
    print(classification_report(y_test, pred_y))
    # 最后需要输出预测的准确率或者均方误差等指标值

# scaler = StandardScaler()
# scaler.fit(X_test)
# X_test_Standard = scaler.transform(X_test)
# scaler.fit(X_train)
# X_train_Standard = scaler.transform(X_train)
