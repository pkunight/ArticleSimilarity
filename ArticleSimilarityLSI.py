import time
import re
import math
import numpy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

start_time = time.time()

file_object = open("199801_clear_utf8.txt")
article_num_list = [""]
word_count = {}

#每个词出现过的文章数
word_article_count = {}
#每篇文章内的词数临时计数
article_word_count = {}

word_bag = {}
total_article_word_list = []
article_num_pattern = re.compile(r"\d{8}-\d{2}-\d{3}")
article_word_pattern = re.compile(r"\S*[^，。：；\"、（）—《》]/\S")
article_count = 0
for line in file_object:
    article_num = article_num_pattern.match(line)
    if article_num:
        paragraph_word_list = article_word_pattern.findall(line)
        #删掉第一列的文章标题
        del paragraph_word_list[0]

        # 判断是否为新文章
        # 开始处理新文章前需先处理上一篇文章的词
        if article_num.group() != article_num_list[article_count]:
            article_num_list.append(article_num.group())
            article_count += 1
            # 将上一篇文章的词更新到word_article_count中
            for w in article_word_count:
                if w in word_article_count:
                    word_article_count[w] += 1
                else:
                    word_article_count[w] = 1
            total_article_word_list.append(article_word_count.copy())
            # 清空文章内词数临时计数
            article_word_count.clear()

        for word in paragraph_word_list:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

            if word in article_word_count:
                article_word_count[word] += 1
            else:
                article_word_count[word] = 1

#最后一篇文章需单独处理
article_num_list.append(article_num.group())
article_count += 1
# 将上一篇文章的词更新到word_article_count中
for w in article_word_count:
    if w in word_article_count:
        word_article_count[w] += 1
    else:
        word_article_count[w] = 1
total_article_word_list.append(article_word_count.copy())
# 清空文章内词数临时计数
article_word_count.clear()

del total_article_word_list[0]
del article_num_list[0]
article_count -= 1

#构建word bag, 并且记录每个词为第x维
x = 0
for word in word_count:
    if word_count[word] >= 2:
        word_bag[word] = x
        x += 1

finish_time = time.time()
cost_time = (finish_time - start_time)
print("Read file, build word bag and article word list time cost: %ss" % cost_time)
start_time = time.time()

#计算word bag中每个词的idf值
word_idf = {}
for word in word_bag:
    word_idf[word] = math.log(article_count / (word_article_count[word] + 1.0))

term_document_lil_matrix = scipy.sparse.lil_matrix((len(word_bag), article_count), dtype='float')

#对每篇文章中每个存在于word bag的词计算tf-idf值并存入矩阵
i = 0
for a_w_c in total_article_word_list:
    #取出每篇文章的article_word_count
    for w in a_w_c:
        if w in word_bag:
            w_tfidf = (a_w_c[w] / sum(a_w_c.values())) * word_idf[w]
            term_document_lil_matrix[word_bag[w], i] = w_tfidf
    i += 1

finish_time = time.time()
cost_time = (finish_time - start_time)
print("Build tf-idf matrix time cost: %ss" % cost_time)
start_time = time.time()

U, sigma, VT = scipy.sparse.linalg.svds(term_document_lil_matrix.tocsr(), k=300)

finish_time = time.time()
cost_time = (finish_time - start_time)
print("Run svd time cost: %ss" % cost_time)
start_time = time.time()

sigma_diag = scipy.diag(sigma)

new_matrix = scipy.dot(scipy.dot(U, sigma_diag), VT)

res = scipy.dot(new_matrix.T, new_matrix)
for i in range(article_count):
    res[i, i] = 0

finish_time = time.time()
cost_time = (finish_time - start_time)
print("Compute result matrix time cost: %ss" % cost_time)
start_time = time.time()

print(word_bag)
print(total_article_word_list[0])

for i in range(10):
    pos = numpy.where(res == numpy.max(res))[0]
    row = pos[0]
    column = pos[1]
    print("Article num", row, column)
    print(article_num_list[row], total_article_word_list[row])
    print(article_num_list[column], total_article_word_list[column])
    res[row, column] = 0
    res[column, row] = 0

#scipy.savetxt("res4.txt", res)