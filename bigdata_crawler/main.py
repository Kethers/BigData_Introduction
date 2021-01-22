import requests
import jsonpath
import pandas as pd
import time
from fake_useragent import UserAgent

ua = UserAgent()
headers = {'User-Agent': ua.random}
valid_websites_num = 0


# 时间转换,原始时间例如"time":1497320494166，转换成例如'2017-06-13 10:21:34.166'年月日时分秒
def stampToTime(stamp):
    datatime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(str(stamp)[0:10])))
    datatime = datatime + '.' + str(stamp)[10:]
    return datatime


# 获取json数据
def get_json(url):
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            json_text = response.json()
            global valid_websites_num
            valid_websites_num += 1
            return json_text
    except Exception:
        print('此页有问题！')
        return None


# n获取评论时间，用户id，用户昵称，评论
def get_comments(url):
    data = []
    doc = get_json(url)  # 调用get_json()函数，获取json数据
    jobs = doc['hotComments']
    for job in jobs:
        dic = {}
        # 使用jsonpath获取需要提取的键值对数据，jsonpath相关内容见官网地址
        # 官网地址   https://pypi.org/project/jsonpath-ng/
        # 调用stampToTime()函数转换时间
        # dic['time'] = stampToTime(jsonpath.jsonpath(job, '$..time')[0])#时间
        # dic['userId'] = jsonpath.jsonpath(job['user'], '$..userId')[0]  # 用户ID
        # dic['nickname'] = jsonpath.jsonpath(job['user'], '$..nickname')[0]  # 用户名
        dic['content'] = jsonpath.jsonpath(job, '$..content')[0].replace('\r', '')  # 评论
        data.append(dic)
    jobs = doc['comments']
    for job in jobs:
        dic = {}
        dic['content'] = jsonpath.jsonpath(job, '$..content')[0].replace('\r', '')
        data.append(dic)
    return data


def main():
    # 评论信息用json数据表示，url时json数据的网址
    url_num = 482633119  # YouSeeBIGGIRL/T:T，歌曲id初始化

    while (valid_websites_num <= 10010):
        url = "http://music.163.com/api/v1/resource/comments/R_SO_4_%d?limit=100&offset=0" % url_num
        # old_valid_websites_num = valid_websites
        comments = get_comments(url)
        url_num += 1
        # comments的类型时list，每个元素都是一个字典，将其每个元素改成list，方便写入csv
        # print(type(comments))
        commentslist = []
        for single_comment in comments:
            singleuser = []
            # singleuser.append(single_comment['time'])
            # singleuser.append(single_comment['userId'])
            # singleuser.append(single_comment['nickname'])
            # 去除content中的\n符号
            singleuser.append(str(single_comment['content']).replace('\n', ''))
            commentslist.append(singleuser)
            file = open("./dataset0.txt", "a+", encoding='utf-8')
        for comment in commentslist:
            comment = str(comment)  # 转成字符串类型，方便切片去掉['评论内容']中首尾的括号和单引号
            # file.write(comment[2:len(comment)-2] + '\n')
            # print(comment[2:len(comment)-2] + '\n')
        print(valid_websites_num)
        # columnsName = ['时间', '用户ID', '用户名', '评论']
        # list没有to_csv的属性，也就是说list直接是转存不了为csv，
        # 为了解决这个问题，我们可以引入pandas模块，使用其DataFrame属性。
        # testdata=pd.DataFrame(columns=columnsName,data=commentslist)
        # 将列表testdata存为csv文件
        # testdata.to_csv('comments.csv',encoding='utf-8')


if __name__ == "__main__":
    main()
