import numpy as np
import random
from preprocessing import Preprocess
import time

settings = {}
settings['window_size'] = 5
settings['epoch'] = 100
settings['dim'] = 100
settings['lr'] = 0.01
settings['subsampling_threshold'] = 0.00001
np.random.seed(1)

corpus = [['On the morning of January 20, Donald Trump and Melania Trump will depart the White House as President and first lady, but they will not invite their incoming counterparts, Joe and Jill Biden, inside before they do.'],
          ["Instead of a president and first lady, the Bidens will be greeted by the White House chief usher Timothy Harleth, according to a source familiar with the day's events and planning."],
          ["The afternoon of Inauguration Day, then-President Biden will participate in a ceremonial wreath-laying at Arlington National Cemetery, joined by former Presidents Bill Clinton, George W. Bush and Barack Obama."],
          ['By that time, all Trump paraphernalia will be gone, and a thorough top-to-bottom cleaning of the entire White House campus will have been completed.'],
          ["A particular focus of this move will be paid to the bedrooms in the residence, where new mattresses and box springs for the incoming first family are standard operating procedure, according to the source."],
          ["On Monday, moving trucks were spotted in the driveways of Mar-a-Lago in Palm Beach, Florida, with movers loading dollies of boxes and items and rolling them into the private club"],
          ["Trump, according to several sources, is even mulling whether to write a letter to Biden to leave for him in the Oval Office, a standard-bearing tradition."],
          ["China's move underscores the fractious and oftentimes hostile relationship between Washington and Beijing during Trump's time in office."],
          ["Many of the 28 outgoing and former officials sanctioned by Beijing were considered to be influential in helping to steer the Trump administration's more confrontational China policy, which saw clashes with Beijing on issues relating to trade, technology, regional security and human rights."],
          ["Just over one week earlier, on January 11, Pompeo announced the US would lift decades-old restrictions on contacts between American and Taiwanese officials, a decision which prompted threats from Beijing."],
          ["If the new US administration can adopt a more rational and responsible attitude in formulating its foreign policy, I think it will be warmly welcomed by everyone in the international community"],
          ["Landscape architect Phil Dunn has taken on an ambitious challenge. For one year, he is basing his diet on food grown in his local community, in an effort to highlight the issues of food sustainability and food waste."],
          ["It's made much easier by the fact that he lives in a community purpose-built to promote a sustainable lifestyle."],
          ["As well as more than 500 houses, the Sustainable City is home to 11 biodome greenhouses, where the community's management grows up to 1 million pots of leafy produce annually, including chives and chicory. The produce is given to residents or sold at local markets."],
          ["Residents can also catch fish at the community tank, breed chickens that provide them with eggs, and rent private plots in communal gardens to grow their own produce. Here, Dunn cultivates cherry tomatoes, lettuces and radishes."],
          ["For produce that cannot be grown at the Sustainable City, such as olive oil, rice and sugar, he's bartering with other residents. In exchange for those foods, Dunn supplies products he has made from recycled wood left over from his gardening work."],
          ["The idea of a circular economy is based on reusing and recycling products and materials used for things like food production, transport and clothing. The sector could generate $4.5 trillion of global economic growth by 2030, according to research by Accenture."],
          ["Established in 2015 by Dubai-based Diamond Developers, the Sustainable City extends over 46 hectares in the outskirts of Dubai."],
          ["Already shredding Donald Trump's legacy, Biden is demonstrating the vast potential of his office to turn the nation's direction on a dime. "],
          ["In the time it took him to write his name, Biden choked off funding for the border wall, which was the single most galvanizing cause of Trumpism."],
          ["They exposed a key weakness of Trump's presidency: Ill-thought-out executive power grabs to win a headline for a strongman president are easy to undo"],
          ["There was none of the walking on eggshells or fawning before the commander in chief that had been required by Trump's brittle public persona. His removal from Washington, for a few hours at least, appeared to change the character of the city"],
          ["And Trump's supporters no doubt will see the restoration of decorum at 1600 Pennsylvania Avenue as a sign the"], ["By late on a chilly Washington night, it felt like far more than a few hours since Trump had lifted off from Joint Base Andrews as the fading bars of rolled over the closing credits of an aberrant four-year term."],
          ["Those are fine thoughts from a lifetime optimist. But the coming Senate impeachment trial of ex-President Trump will be sure to reopen old wounds."],
          ["Those are fine thoughts from a lifetime optimist. But the coming Senate impeachment trial of ex-President Trump will be sure to reopen old wounds."],
          ["Biden, speaking from the Resolute Desk in the Oval Office after signing Day One executive orders, said he would not immediately reveal the contents of the letter out of respect for Trump."],
          ["A senior Trump aide described the letter to CNN as a personal note that prays for the success of the country and the new administration to care for the country. The aide said writing the letter to Biden was one of the many items on Trump's list in the Oval Office Tuesday night."],
          ["n a briefing Wednesday night, White House press secretary Jen Psaki declined to offer more details on the letter Trump left for Biden, telling reporters that, based on comments from Biden"],
          ["While it may seem surprising Trump upheld this tradition when he ignored every other element of a peaceful handoff, it is in keeping with his past enthusiasm at the letter he received from President Barack Obama."],
          ["Trump was so taken with Obama's letter that he tried phoning him as soon as he read it on Inauguration Day in 2017."],
          ["When one of Obama's aides reached back out to the White House to return the call, the new President's staffers said Trump just wanted to say thank you for the note"],
          ["In his farewell address on Wednesday, Trump did not name Biden, but said he wished the"],
          ["This comes after Trump spent months lying about the presidential election being rigged against him and spreading baseless conspiracy theories about voter fraud."],
          ["He's also got a slate of Day One executive orders meant to undo what President Donald Trump has wrought over the past four years, including"],
          ["The US abandoned the agreement late last year on former President Donald Trump's orders. Trump spent much of his time in office weakening many of the country's bedrock climate and environmental guardrails."],
          ["Biden is also signing a raft of executive orders aimed at gutting Trump's climate and environmental policies."],
          ["When Trump pulled the US out of the accord, it was not the first time that the country had left an international climate agreement after leading the negotiations"],
          ["During Trump's presidency, global average temperatures continued to climb."],
          ["Joe Biden was sworn in as the 46th president of the United States on Wednesday, calling for unity as he assumed leadership of a nation battered by a once-in-a-century pandemic and the recent deadly storming of the U.S. Capitol."],
          ["At a midday ceremony overshadowed by health and safety concerns, Mr. Biden was sworn in by Chief Justice John Roberts, repeating the oath of office on the west steps of the Capitol with his hand on a family Bible."],
          ["Former President Donald Trump became the first president to skip his successor’s inauguration in 150 years, instead leaving Washington for his Florida estate early in the morning, vowing in his final public comments as president"],
          ["After questioning the election results for months after his loss, Mr. Trump encouraged his supporters two weeks ago to march on the Capitol to stop Mr. Biden’s victory from being certified"],
          ["With his inaugural address, Mr. Biden asked a deeply divided country to look past their differences and sought to draw a contrast with Mr. Trump, who took a combative approach to politics during his four years in office and had urged supporters to march to the Capitol before the riot on Jan. 6."],
          ["Mr. Biden made no mention of his predecessor, but sought to use the moment as an inflection point for the nation."],
          ["Referencing the Jan. 6 attack by a pro-Trump mob, Mr. Biden stressed that they didn’t succeed in disrupting democracy and declared:"],
          ["Mr. Biden asked those who opposed his candidacy during the contentious presidential election to give him a chance, pledging to be a president for all Americans. After Mr. Trump had for weeks said without evidence that he was the actual winner of the election, Mr. Biden criticized"],
          ["President Biden on his first day in office took a range of executive actions, including implementing a national mask mandate on federal property, revoking a permit for the Keystone XL oil pipeline and reversing a travel ban from several largely Muslim and African countries, officials said."],
          ["Coming after Mr. Biden was inaugurated as the 46th president in a midday ceremony, the actions are intended to signal a more aggressive approach to the coronavirus pandemic and end some of President Trump’s key policies while setting the agenda for the next four years."],
          ["Mr. Biden signed executive orders from the Oval Office in the late afternoon."],
          ["Mr. Biden signed 15 executive orders and two executive actions on his first day in office, far more than any of his modern predecessors, none of whom signed more than one. President Trump signed an order on his inauguration day aimed at reversing the Affordable Care Act, while Barack Obama didn’t sign any on Jan. 20, 2009. Bill Clinton signed an ethics order on his first day. All of them signed additional orders during their first week in office."],
          ["On the same afternoon he’s sworn in as the nation’s 46th president, Joe Biden will take executive actions that will undo several of former President Donald Trump’s immigration policies, his transition team announced Wednesday."],
          ["The actions come the same day Biden will reportedly send to Congress a comprehensive immigration bill that, if passed, could provide a legal path to citizenship for millions of undocumented immigrants, including more than 1.7 million in Texas."],
          ["A separate Biden executive action seeks to roll back Trump’s interior enforcement initiatives and allow federal immigration agencies"],
          ["Biden will end an emergency declaration that allowed Trump to divert billions of dollars in military construction and payroll funds for construction of the barrier."],
          ["Bipartisan majorities in Congress refused in 2019 to fund President Trump’s plans for a massive wall along our southern border, even after he shut down the government over this issue"],
          ["A White House official confirmed earlier in the day that Trump left Biden a note, keeping with a presidential tradition where the departing commander-in-chief leaves behind a personal message for their successor"],
          ["But it was unclear whether Trump would offer any kind of personal message to Biden given he has refused to concede that he lost the election fairly and has yet to refer to Biden by name when speaking of the incoming administration."],
          ["Trump did not attend Biden's inauguration earlier Wednesday. Instead, Trump spoke at a send-off ceremony at Joint Base Andrews before flying on Air Force One to Florida before noon."],
          ["President Joe Biden is moving swiftly to dismantle Donald Trump's legacy on his first day in office, signing a series of executive actions that reverse course on immigration, climate change, racial equity and the handling of the coronavirus pandemic."],
          ["Biden wore a mask as he signed the orders in the Oval Office — a marked departure from Trump, who rarely wore a face covering in public and never during events in the Oval Office."],
          ["Former Presidents Barack Obama, George W. Bush and Bill Clinton honored President Joe Biden Wednesday evening as America's new leader in a joint message that emphasized the new President's call for national unity."],
          ["the trio each wished Biden well as he stepped into the White House."],
          ["Their message stood in stark contrast to former President Donald Trump, who didn't attend Biden's inauguration and ignored nearly every element of a peaceful handoff of presidential power."],
          ["The comment came after Trump spent months lying about the presidential election being rigged against him and spreading baseless conspiracy theories about voter fraud."],
          ["Joe Biden's transition team found a culture of coronavirus skepticism within Donald Trump's federal government as they prepared to take office, sources close to the Biden transition told CNN, with political appointees loyal to the President reflecting his dismissiveness of public health guidelines and sometimes mocking career employees for wearing masks."],
          ["The findings from Biden's agency review teams are some of the earliest readouts from the Biden officials who were tasked with preparing for the new administration and signal one of the most apparent early changes that the incoming administration will make."],
          ["The Biden transition was subject to some of these unclear guidelines, as well, with one source describing awkward moments within some agencies where the Biden team would try to avoid large in-person meetings as the political leadership at those same agencies would schedule gatherings in small conferences rooms with"],
          ["Biden's transition team also grew concerned that many in the department had not been prioritized for the coronavirus vaccine, despite high rates of infection inside the agency."],
          ["One of Biden's earliest goals in the White House will be to implement strict coronavirus rules in places where he has the most authority "],
          ["Trump refused to accept the magnitude of the coronavirus crisis throughout his final year in office, declining to wear a mask for months and downplaying the virus in an attempt to allay fears that were dominating the presidential election."],
          ["Trump's attitude permeated the White House coronavirus task force, too, which was caught between pushing best practices to battle the coronavirus and navigating a White House that did not want to be seen enforcing mask wearing."]]

a = Preprocess()
corpus = a.overall(corpus)

class SsDn():
    def __init__(self, corpus):
        self.window_size = settings['window_size']
        self.epoch = settings['epoch']
        self.dim = settings['dim']
        self.lr = settings['lr']
        self.corpus = corpus
        self.subsampling_threshold = settings['subsampling_threshold']

        self.vocab_count = {}

        for sentence in self.corpus:
            for word in sentence:
                if word not in self.vocab_count:
                    self.vocab_count[word] = 1
                else:
                    self.vocab_count[word] += 1

        self.vocab_list = sorted(self.vocab_count.keys(), reverse=False)
        self.word2id = dict((word, i) for i, word in enumerate(self.vocab_list))
        self.id2word = dict((i, word) for i, word in enumerate(self.vocab_list))

    def generate_train_data(self):

        for sentence in self.corpus:
            for word in sentence:
                if np.random.uniform() < self.subsampling(word):
                    if self.vocab_count[word] > 2:
                        self.vocab_count[word] -= 1

        self.total_count = len(self.vocab_count.keys())

        train_data = []
        for sentence in self.corpus:
            sent_len = len(sentence)
            for i, word in enumerate(sentence):
                w_target = self.ohe(sentence[i])

                w_context = []
                window_size = random.randint(1, self.window_size)
                for j in range(i - window_size, i + window_size + 1):
                    if j!=i and j >=0 and j <= sent_len-1:
                        w_context.append(self.ohe(sentence[j]))
                train_data.append((w_target, w_context))

        return np.array(train_data)

    def subsampling(self, x):
        self.prob = 1 - np.sqrt(self.subsampling_threshold/self.vocab_count[x])
        return self.prob


    def ohe(self, x):
        ohe_vec = [0 for i in range(self.total_count)]
        word_id = self.word2id[x]
        ohe_vec[word_id] = 1
        return ohe_vec

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    def forward(self, x):
        idx = x.index(1)
        h = self.w1[idx]
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    def backward(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        self.w1 = self.w1 - (self.lr * dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)

    def train(self):

        self.w1 = np.random.uniform(-0.9, 0.9, (self.total_count, self.dim))
        self.w2 = np.random.uniform(-0.9, 0.9, (self.dim, self.total_count))

        for i in range(self.epoch):
            train_data = self.generate_train_data()
            self.loss = 0

            for target, context in train_data:
                y_pred, h, u = self.forward(target)

                error = np.sum([np.subtract(y_pred, word) for word in context], axis=0)

                self.backward(error, h, target)

                self.loss += -np.sum([u[word.index(1)] for word in context]) + len(context)*np.log(np.sum(np.exp(u)))

            print('Epoch:', i, 'Loss:', self.loss)
        pass

    def word_sim(self, word, top_n):
        w1_index = self.word2id[word]
        v_w1 = self.w1[w1_index]

        word_sim = {}
        for i in range(self.total_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num/theta_den

            word = self.id2word[i]
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

                w2_vec = self.w1[self.word2id[w2]]
                w3_vec = self.w1[self.word2id[w3]]
                w4_vec = self.w1[self.word2id[w4]]

                vec = w2_vec+w3_vec-w4_vec

                word_sim = {}
                for i in range(self.total_count):
                    v_w2 = self.w1[i]
                    theta_num = np.dot(vec, v_w2)
                    theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
                    theta = theta_num / theta_den

                    word = self.id2word[i]
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

if __name__ == '__main__':
    current = time.time()
    ss = SsDn(corpus)
    training_data = ss.generate_train_data()

    ss.train()
    print(time.time() - current)

    print('Subsampling + Dynamic Window Size + Vanilla Word2Vec')
    print('====================')
    ss.word_sim('trump', 5)
    print('====================')
    ss.word_sim('biden', 5)
    print('====================')
    ss.word_sim('president', 5)
    print('====================')
