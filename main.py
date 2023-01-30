import os
import re
import string
import math
import numpy as np

DATA_DIR = 'enron'
# target_names = ['ham', 'spam']


def get_data(DATA_DIR):
    subfolders = ['enron%d' % i for i in range(1, 7)]
    data = []
    target = []
    for subfolder in subfolders:
        # spam
        spam_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'spam'))
        for spam_file in spam_files:
            with open(os.path.join(DATA_DIR, subfolder, 'spam', spam_file), encoding="latin-1") as f:
                data.append(f.read())
                target.append(1)
        # ham
        ham_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'ham'))
        for ham_file in ham_files:
            with open(os.path.join(DATA_DIR, subfolder, 'ham', ham_file), encoding="latin-1") as f:
                data.append(f.read())
                target.append(0)
    # 乱序
    tmp = np.arange(len(target))
    np.random.shuffle(tmp)
    data1 = []
    target1 = []
    for t in tmp:
        data1.append(data[t])
        target1.append(target[t])
    return data1, target1


X, y = get_data(DATA_DIR)


class SpamDetector_1(object):
    """Implementation of Naive Bayes for binary classification"""

    # 清除特殊字符
    def clean(self, s):
        translator = str.maketrans("", "", string.punctuation)
        return s.translate(translator)

    # 分开每个单词
    def tokenize(self, text):
        text = self.clean(text).lower()
        return re.split("\W+", text)

    # 计算某个单词出现的次数
    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts


class SpamDetector_2(SpamDetector_1):
    # X:data,Y:target标签（垃圾邮件或正常邮件）
    def fit(self, X, Y):
        self.num_messages = {}
        self.log_class_priors = {}
        self.word_counts = {}
        # 建立一个集合存储所有出现的单词
        self.vocab = set()
        # 统计spam和ham邮件的个数
        self.num_messages['spam'] = sum(1 for label in Y if label == 1)
        self.num_messages['ham'] = sum(1 for label in Y if label == 0)

        # 计算先验概率，即所有的邮件中，垃圾邮件和正常邮件所占的比例
        self.log_class_priors['spam'] = math.log(
            self.num_messages['spam'] / (self.num_messages['spam'] + self.num_messages['ham']))
        self.log_class_priors['ham'] = math.log(
            self.num_messages['ham'] / (self.num_messages['spam'] + self.num_messages['ham']))

        self.word_counts['spam'] = {}
        self.word_counts['ham'] = {}

        for x, y in zip(X, Y):
            c = 'spam' if y == 1 else 'ham'
            # 构建一个字典存储单封邮件中的单词以及其个数
            counts = self.get_word_counts(self.tokenize(x))
            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)  # 确保self.vocab中含有所有邮件中的单词
                # 下面语句是为了计算垃圾邮件和非垃圾邮件的词频，即给定词在垃圾邮件和非垃圾邮件中出现的次数。
                # c是0或1，垃圾邮件的标签
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0
                self.word_counts[c][word] += count


# MNB = SpamDetector_2()
# MNB.fit(X[100:], y[100:])


class SpamDetector(SpamDetector_2):
    def predict(self, X):
        result = []
        # flag_1 = 0
        # 遍历所有的测试集
        for x in X:
            counts = self.get_word_counts(self.tokenize(x))  # 生成可以记录单词以及该单词出现的次数的字典
            spam_score = 0
            ham_score = 0
            # flag_2 = 0
            for word, _ in counts.items():
                # 下面计算P(内容|垃圾邮件)和P(内容|正常邮件),所有的单词都要进行拉普拉斯平滑
                log_w_given_spam = math.log(
                    (self.word_counts['spam'].get(word, 0) + 1) / (
                            sum(self.word_counts['spam'].values()) + len(self.vocab)))
                log_w_given_ham = math.log(
                    (self.word_counts['ham'].get(word, 0) + 1) / (sum(self.word_counts['ham'].values()) + len(
                        self.vocab)))

                # 把计算到的P(内容|垃圾邮件)和P(内容|正常邮件)加起来
                spam_score += log_w_given_spam
                ham_score += log_w_given_ham

                # flag_2 += 1

            # 最后，还要把先验加上去，即P(垃圾邮件)和P(正常邮件)
            spam_score += self.log_class_priors['spam']
            ham_score += self.log_class_priors['ham']

            # 最后进行预测，如果spam_score > ham_score则标志为1，即垃圾邮件
            if spam_score > ham_score:
                result.append(1)
            else:
                result.append(0)

            # flag_1 += 1

        return result


MNB = SpamDetector()
# split = int(len(y) * 0.8)
split = -500
MNB.fit(X[:split], y[:split])
pred = MNB.predict(X[split:])
true = y[split:]

accuracy = 0
for i in range(len(true)):
    if pred[i] == true[i]:
        accuracy += 1
print(accuracy/len(true))
# print(len(true))