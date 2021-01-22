import jieba

# 原始爬取的文件，一行就是一个用户的评论
origin_data_fo = open("./dataset0.txt", "r", encoding='utf-8')

# 读取停用词表
stopwords_fo = open("./stopwords.txt", "r", encoding='utf-8')
stopwords = stopwords_fo.read().split('\n')  # 以'\n'为标识符对读取到的字符串进行切片，即提取每一行的字符串而不要\n

# 分词输出到的文件
divided_data_fo = open("./divided_with_spaces.txt", "w", encoding='utf-8')

line_num = 0
for line in origin_data_fo.readlines():
        line_num += 1
        cut = list(jieba.cut_for_search(line))
        print("Reading Line %d" % line_num)
        print(line)
        # 精确模式
        # print('精确模式输出：')
        for word in cut:
            if not (word.strip() in stopwords) and (len(word.strip()) > 1):
                # print(' '.join(word), end='')
                divided_data_fo.write(word + ' ')  # 分词以空格隔开
        print('\n')

print("--------------Mission Complete!--------------")
