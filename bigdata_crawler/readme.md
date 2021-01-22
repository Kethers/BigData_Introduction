# 作业1文件说明

### main.py

调用网易云评论api链接，会返回一个json包，对包中内容进行字符串处理，得到该url的用户评论内容，url后面的编号为歌曲编号，使用循环对其+1可爬取共计10010个歌曲的前面十几个评论

### test_website.txt

代码所使用的使用的一些网页

### dataset0.txt

爬虫爬取到的评论

### words_divided.py

使用jieba库对爬取到的评论进行分词处理，方便后续hadoop实验中进行词频统计

### stopwords.txt

分词代码中要用到的停用词

### divided_with_spaces.txt

对dataset0.txt中的评论用jieba分词后得到的分词文件

### part-r-0000

hadoop实验中对divided_with_spaces.txt进行词频统计得到的统计结果

