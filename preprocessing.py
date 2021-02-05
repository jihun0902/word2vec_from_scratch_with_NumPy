import re
import string

corpus = []
#corpus = [[open('../data/news1.txt').read()]]

#for i in range(1,10):
#    corpus.append([open('./1billion/training/news.en-0000'+str(i)+'-of-00100.txt').read()])
#for i in range(10,100):
#    corpus.append([open('./1billion/training/news.en-000' + str(i) + '-of-00100.txt').read()])

class Preprocess():
    def __init__(self):
        self.corpus = corpus

    def lower_split(self, corpus):
        data = []
        for sentence in corpus:
            for word in sentence:
                data.append(word.lower().split())
        return data

    def remove_num(self, corpus):
        data = []
        for sentence in corpus:
            dat = []
            for word in sentence:
                dat.append(re.sub(r'\d+', '', word))
            data.append(dat)
        return data

    def remove_punc(self, corpus):
        data = []
        for sentence in corpus:
            dat = []
            for text in sentence:
                dat.append((text.translate(str.maketrans('','', string.punctuation))).strip())
            data.append(dat)
        return(data)

    def overall(self, corpus):
        x = self.lower_split(corpus)
        x = self.remove_num(x)
        x = self.remove_punc(x)

        for i in x:
            for j in i:
                if j =='':
                    del i[i.index(j)]

        return x

if __name__ == '__main__':
    a = Preprocess()
    a.overall(corpus)