import nltk
import string
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
#nltk.download('stopwords')

#%matplotlib inline


# load data
with open('Movies_and_TV_5.json') as f:
    raw_data = [eval(x) for x in f.readlines()]

with open('ratings.txt', 'w') as f:
    f.write('reviewerID,asin,overall\n')
    for d in raw_data:
        f.write('{},{},{}\n'.format(d['reviewerID'], d['asin'], d['overall']))
     
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
        
        
def preprocess(data):
    all_words = []
    all_reviews = []
    for r in data:
        r = r['reviewText'].lower()
        r = ''.join([c for c in r if c not in punctuation])
        review = []
        for w in r.split():
            if w not in stop_words:
                all_words.append(w)
                review.append(w)
        review = ' '.join(review)
        all_reviews.append(review)
    return all_reviews, all_words

reviews, all_words = preprocess(raw_data)



lengths = []
for r in reviews:
    lengths.append(len(r.split()))
    

plt.hist(lengths, bins=100)
plt.show()


print('total words: ', len(all_words))
words_freq = nltk.FreqDist(all_words)
print('unique words: ', len(words_freq))


vocab_size = 5000
vocab = [x[0] for x in words_freq.most_common(vocab_size)]
vocab_set = set(vocab)
vocab2indx = dict(zip(vocab, range(vocab_size)))


all_users = set()
all_items = set()
for d in raw_data:
    all_users.add(d['reviewerID'])
    all_items.add(d['asin'])
print(len(all_users))
print(len(all_items))


user2indx = dict(zip(all_users, range(len(all_users))))
item2indx = dict(zip(all_items, range(len(all_items))))


def get_user_item(d):
    return (user2indx[d['reviewerID']], item2indx[d['asin']])
user_item = [get_user_item(d) for d in raw_data]


np.save('user_item.npy', user_item)


def feature(review):
    feat = []
    for w in review.split():
        if w in vocab_set:
            feat.append(vocab2indx[w])
    return feat


reviews_feats = [feature(r) for r in reviews]
ratings = [r['overall'] for r in raw_data]

np.save('processed_data.npy', {'features': reviews_feats,
                               'ratings': ratings})