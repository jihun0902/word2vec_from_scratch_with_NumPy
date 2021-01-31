import numpy as np
import re
from preprocessing import Preprocess
from collections import defaultdict
import a

settings = {}
settings['n'] = 150
settings['window_size'] = 1
settings['min_count'] = 0
settings['epochs'] = 2
settings['learning_rate'] = 0.001
np.random.seed(1)

corpus = a.corpus

class word2vec():
    def __init__(self):
        self.n = settings['n']
        self.window_size = settings['window_size']
        self.min_count = settings['min_count']
        self.epochs = settings['epochs']
        self.learning_rate = settings['learning_rate']

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def word_ohe(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec

    def generate_training_data(self, corpus):

        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())


        self.words_list = sorted(list(word_counts.keys()), reverse=False)
        self.word_index = dict((word,i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i,word) for i, word in enumerate(self.words_list))

        training_data = []

        for sentence in corpus:
            sent_len = len(sentence)


            for i, word in enumerate(sentence):
                w_target = self.word_ohe(sentence[i])


                w_context = []
                for j in range(i-self.window_size, i+self.window_size+1):
                    if j!=i and j<=sent_len-1 and j>=0:
                        w_context.append(self.word_ohe(sentence[j]))
                training_data.append([w_target, w_context])
        return np.array(training_data)



    def forward_pass(self,x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    def backpropagation(self, e, h, x):
        dl_dw2 = np.outer(h,e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        self.w1 = self.w1 - (self.learning_rate * dl_dw1)
        self.w2 = self.w2 - (self.learning_rate * dl_dw2)
        pass


    def train(self, training_data):
        self.w1 = np.random.uniform(-0.9,0.9, (self.v_count, self.n))
        self.w2 = np.random.uniform(-0.9,0.9, (self.n, self.v_count))

        for i in range(0, self.epochs):
            self.loss = 0

            for w_t, w_c in training_data:

                y_pred, h, u = self.forward_pass(w_t)

                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                self.backpropagation(EI, h, w_t)

                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))

            print('EPOCH:', i, 'LOSS:', self.loss)
    pass

    def word_sim(self, word, top_n):
        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=(lambda x: x[1]), reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)

        pass

    def analogy(self):
        f = open("../data/questions-words.txt", 'r')
        lines = f.readlines()
        scores = []
        for line in lines:
            try:
                w1 = line.split()[0]
                w2 = line.split()[1]
                w3 = line.split()[2]
                w4 = line.split()[3]

                w2_vec = self.w1[self.word_index[w2]]
                w3_vec = self.w1[self.word_index[w3]]
                w4_vec = self.w1[self.word_index[w4]]

                vec = w2_vec+w3_vec-w4_vec

                word_sim = {}
                for i in range(self.v_count):
                    v_w2 = self.w1[i]
                    theta_num = np.dot(vec, v_w2)
                    theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
                    theta = theta_num / theta_den

                    word = self.index_word[i]
                    word_sim[word] = theta

                words_sorted = sorted(word_sim.items(), key=(lambda x: x[1]), reverse=True)

                for word, sim in words_sorted[0]:
                    if word == w1:
                        scores.append(1)
                    else:
                        scores.append(0)
            except:
                continue

        f.close()

        acc = sum(scores) / (len(scores) + 1e-7)

        print(acc)
        pass

    def analogy2(self):
        f = open("../data/questions-words2.txt", 'r')
        lines = f.readlines()
        scores = []
        for line in lines:
            try:
                w1 = line.split()[0]
                w2 = line.split()[1]
                w3 = line.split()[2]
                w4 = line.split()[3]

                w2_vec = self.w1[self.word_index[w2]]
                w3_vec = self.w1[self.word_index[w3]]
                w4_vec = self.w1[self.word_index[w4]]

                vec = w2_vec+w3_vec-w4_vec

                word_sim = {}
                for i in range(self.v_count):
                    v_w2 = self.w1[i]
                    theta_num = np.dot(vec, v_w2)
                    theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
                    theta = theta_num / theta_den

                    word = self.index_word[i]
                    word_sim[word] = theta

                words_sorted = sorted(word_sim.items(), key=(lambda x: x[1]), reverse=True)

                for word, sim in words_sorted[0]:
                    if word == w1:
                        scores.append(1)
                    else:
                        scores.append(0)
            except:
                continue

        f.close()

        acc = sum(scores) / (len(scores) + 1e-7)

        print(acc)
        pass

w2v = word2vec()
training_data = w2v.generate_training_data(corpus)
w2v.train(training_data)

print('Vanilla Word2Vec')
print('semantic')
w2v.analogy()
print('syntactic')
w2v.analogy2()
'''print('====================')
w2v.word_sim('trump', 5)
print('====================')
w2v.word_sim('biden', 5)
print('====================')
w2v.word_sim('president', 5)
print('====================')'''