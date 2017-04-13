import time
import re
from numpy import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

start_time = time.time()

file_object = open("199801_clear_utf8.txt")
article_num_list = [""]
article_content_list = [""]
article_content = ""
count = 0
article_num_pattern = re.compile(r"\d{8}-\d{2}-\d{3}")
article_paragraph_pattern = re.compile(r"\d{8}-\d{2}-\d{3}-\d{3}/m\s([\s\S]+)\Z")
delete_tag_pattern = re.compile(r"[，。、；（）/A-Za-z\n]")
delete_duplicate_space_pattern = re.compile(r"\s+")
for line in file_object:
    article_num = article_num_pattern.match(line)
    if article_num:
        article_paragraph = article_paragraph_pattern.findall(line)[0]
        article_paragraph = re.subn(delete_tag_pattern, "", article_paragraph)[0]
        article_paragraph = re.subn(delete_duplicate_space_pattern, " ", article_paragraph)[0]
        if article_num.group(0) == article_num_list[count]:
            article_content += article_paragraph
        else:
            article_content_list.append(article_content)
            count += 1
            article_num_list.append(article_num.group(0))
            article_content = article_paragraph
#最后一次还未拼接循环已结束
article_content_list.append(article_content)

del article_num_list[0]
del article_content_list[0:2]

finish_time = time.time()
cost_time = (finish_time - start_time)
print("Read file and build article array time cost: %ss" % cost_time)
start_time = time.time()

vectorizer = CountVectorizer()
transformer = TfidfTransformer()

X = vectorizer.fit_transform(article_content_list)

tfidf = transformer.fit_transform(X)

finish_time = time.time()
cost_time = (finish_time - start_time)
print("Compute tf-idf matrix time cost: %ss" % cost_time)
start_time = time.time()

#计算每篇文章与其他文章的相似度,保存为上三角矩阵
#此部分代码为自己完成,难免效率较低
[article_count, word_count] = shape(tfidf)
print("Article Count:", article_count, "Word Bag Count:", word_count)
vectorLen = []

for i in range(article_count):
    v = tfidf[i, :]
    vectorLen.append(float((v * v.T)[0, 0]))

res = zeros((article_count, article_count))
for i in range(article_count):
    v1 = tfidf[i, :]
    for j in range(i+1, article_count):
        v2 = tfidf[j, :]
        res[j, i] = (v1 * v2.T)[0, 0]/(vectorLen[i] * vectorLen[j])

finish_time = time.time()
cost_time = (finish_time - start_time)
print("Compute article cos similarity matrix time cost: %ss" % cost_time)
start_time = time.time()

savetxt("res2.txt", res)

finish_time = time.time()
cost_time = (finish_time - start_time)
print("Write result to file time cost: %ss" % cost_time)

maxNum = 0
for i in range(article_count):
    if res[maxNum, 0] < res[i, 0]:
        maxNum = i

print(article_num_list[maxNum])