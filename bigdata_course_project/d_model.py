import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
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

    sex = '性别'
    date = '体检日期'

    # 筛选重复值并删除
    # df.duplicated()
    df.drop_duplicates()

    # 将性别映射成0 1
    sex_mapping = {'女': 0, '男': 1}
    df[sex] = df[sex].map(sex_mapping)
    df[sex] = df[sex].fillna(0)  # 有非男女的性别，上一步映射后为缺失值NaN，填充该缺失值为0

    # 将月份和日期转化为距离2021.01.01的天数
    df[date] = pd.to_datetime(df[date], format="%d/%m/%Y")
    df[date] = pd.to_datetime('01/01/2021', format="%d/%m/%Y") - df[date]

    # 将性别和体检日期转为整数类型
    # # 先将日期 xxx days转为str，然后去掉 days再转为int64
    # df[date] = df[date].astype('str').apply(lambda x: x[:-5]).astype('int64')
    df[date] = df[date].dt.days
    df[sex] = df[sex].astype('int64')
    # print(df.dtypes)  # 显示每一列的类型

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
            df[col] = df[col].fillna(df_mean)
    cols = list(df.columns.values)  # 因为前面有缺省过多的列表被删掉了，所以在这里更新cols

    # 对除血糖和性别（性别只有0 1，归不归了结果一样）外的所有数据归一
    for col in cols[1:-1]:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    # df.to_csv('d_preprocessed.csv', encoding='gbk')
    # print(df)         # 显示表格内容


class Model():
    def __init__(self, **kwargs):
        # 模型初始化
        # 构建模型，2个隐藏层，第一个隐藏层有100个神经元，第2隐藏层50个神经元，训练30000周期(直到收敛为止)
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=30000)

    def train(self, X, y):
        # 模型构建
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        pred_y = self.model.predict(X) / 100
        return np.around(pred_y, 3)


def resultToCsv(pred, test):
    """
        将血糖预测值与实际值输出到csv文件
    """
    df = pd.DataFrame(columns=['血糖实际值', '血糖预测值'])
    df.index.name = '编号=原始数据id号-1'
    df['血糖实际值'] = test['血糖']
    df['血糖预测值'] = pred
    df.to_csv('d_result.csv', encoding='gbk')
    print('Check prediction of every one in d_result.csv')


if __name__ == '__main__':
    # 读取训练集数据 train_x,train_y 并训练模型（直接读取，不需要在命令行指定）
    df_train = pd.read_csv('d_train.csv', encoding='gbk')
    preProcess(df_train, 0, [])  # 数据预处理
    print('train data pre-process finished.\n--------------------------------')
    cols_train = list(df_train.columns.values)
    cols_train.remove('血糖')
    X = df_train[cols_train]
    y = df_train[['血糖']]
    X_train, X_test, y_train, y_test = train_test_split(X, y)  # 划分测试集与训练集
    y_train = (100 * y_train).astype('int64')  # 将目标训练值（血糖值）*100后转为int64类型

    # 从命令行获取并读取测试集数据并做相应的预处理得到test_X,test_Y
    df_test = pd.DataFrame
    if len(sys.argv) > 1:
        test_file_path = sys.argv[1]
        df_test = pd.read_csv(test_file_path, encoding='gbk')
        preProcess(df_test, 1, cols_train)
        cols_test = cols_train
        X_test = df_test[cols_test]
        y_test = df_test[['血糖']]
        X_train = df_train[cols_test]    # 将train的x,y更新为d_train.csv中的所有数据
        y_train = df_train[['血糖']]
        y_train = (100 * y_train).astype('int64')  # 将目标训练值（血糖值）*100后转为int64类型

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
    print('MSE: ', mean_squared_error(y_test, pred_y))
    # 最后需要输出预测的准确率或者均方误差等指标值

# scaler = StandardScaler()
# scaler.fit(X_test)
# X_test_Standard = scaler.transform(X_test)
# scaler.fit(X_train)
# X_train_Standard = scaler.transform(X_train)
