import numpy as np
import matplotlib.pyplot as plt
import string 
import os
from collections import defaultdict
import pdb

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from transformTraditional.transformTraditional import TransformTraditional
from weighUnlabelled.weighUnlabelled import WeighUnlabelled
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

stop_words = set(stopwords.words('english'))
with open('additional_stopwords.txt', 'r') as f:
    for line in f:
        stop_words.add(stemmer.stem(line.strip().lower()))


topk = 50

train_split = 70
val_split = 20
test_split = 10

dictionary = defaultdict(lambda: 0)

def tokenize(text):
    tokens = word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(stemmer.stem(item))
    return stems

token_dict = {}
path = '/scratchd/home/adityaas/MyResearch/Baselines_DCAP/pu-learning/src/datasets/Samples/P/'

translator = str.maketrans('', '', string.punctuation)

dictionary = defaultdict(lambda: 0)

for dirpath, dirs, files in os.walk(path):
    for f in files:
        fname = os.path.join(dirpath, f)
        with open(fname) as pearl:
            text = pearl.read()
            token_dict[f] = text.replace('\n', ' ').lower().translate(translator)

            for word in tokenize(token_dict[f]):
                if word not in stop_words:
                    dictionary[word] += 1    

P_data = []
P_labels = []
for value in token_dict.values():
    P_data.append(value)
    P_labels.append(1)

reverse_dictionary_list = list(reversed(sorted(dictionary.items(), key=lambda x:x[1])))

encoding_list = reverse_dictionary_list[0:topk]
many_hot_list = [i[0] for i in encoding_list]

data = np.zeros((1,topk))
select = []
labels = []


for d in P_data:
    tokens = tokenize(d)
    vector = []
    for i in many_hot_list:
        if i in tokens:
            vector.append(1)
        else:
            vector.append(0)

    vector = np.asarray(vector).reshape((1, topk))
    data = np.append(data, vector, axis=0)
    labels.append(1.0)
    select.append(1.0)

data = np.delete(data, (0), axis=0)

path = '/scratchd/home/adityaas/MyResearch/Baselines_DCAP/pu-learning/src/datasets/Samples/N/'
token_dict = {}
for dirpath, dirs, files in os.walk(path):
    for f in files:
        fname = os.path.join(dirpath, f)
        with open(fname) as pearl:
            text = pearl.read()
            token_dict[f] = text.replace('\n', ' ').lower().translate(translator)

N_data = []
N_labels = []
for value in token_dict.values():
    N_data.append(value)
    N_labels.append(0)

for d in N_data:
    tokens = tokenize(d)
    vector = []
    for i in many_hot_list:
        if i in tokens:
            vector.append(1)
        else:
            vector.append(0)

    vector = np.asarray(vector).reshape((1, topk))
    data = np.append(data, vector, axis=0)
    labels.append(-1.0)
    select.append(-1.0)

token_dict = {}
path = '/scratchd/home/adityaas/MyResearch/Baselines_DCAP/pu-learning/src/datasets/Samples/Q/'
for dirpath, dirs, files in os.walk(path):
    for f in files:
        fname = os.path.join(dirpath, f)
        with open(fname) as pearl:
            text = pearl.read()
            token_dict[f] = text.replace('\n', ' ').lower().translate(translator)

Q_data = []
Q_labels = []
for value in token_dict.values():
    Q_data.append(value)
    Q_labels.append(0)

for d in Q_data:
    tokens = tokenize(d)
    vector = []
    for i in many_hot_list:
        if i in tokens:
            vector.append(1)
        else:
            vector.append(0)

    vector = np.asarray(vector).reshape((1, topk))
    data = np.append(data, vector, axis=0)
    labels.append(1.0)
    select.append(-1.0)



select = np.array(select)
labels = np.array(labels)

# Data = <X, S, Y>
all_data = np.array(list(zip(data, select, labels)))

print("Total Data Size: ", len(all_data))
print("Positive Labeled: ", len(P_data))
print("Positive Unlabeled: ", len(Q_data))
print("Negative Unlabeled: ", len(N_data))

indices = np.arange(len(all_data))

np.random.seed(357)
np.random.shuffle(all_data)

trainlen = int(0.8*len(all_data))
train_data = all_data[0:trainlen]
test_data = all_data[trainlen:]


train_X = []
train_X_temp = train_data[:,0]
for x in train_X_temp:
    train_X.append(x)

train_X = np.array(train_X)
train_S = np.array(train_data[:,1])
train_Y = np.array(train_data[:,2])

test_X = []
test_X_temp = test_data[:,0]
for x in test_X_temp:
    test_X.append(x)

test_X = np.array(test_X)
test_S = np.array(test_data[:,1])
test_Y = np.array(test_data[:,2])

print("Train Data Size: ", len(train_data))
print("Train Labeled Size: ", len(np.where(train_S == +1.)[0]))
print("Train Unlabeled Size: ", len(np.where(train_S == -1.)[0]))


print("Test Data Size: ", len(test_data))
print("Test Labeled Size: ", len(np.where(test_S == +1.)[0]))
print("Test Unlabeled Size: ", len(np.where(test_S == -1.)[0]))


n_sacrifice_iter = range(0, len(np.where(train_S == +1.)[0])-21, 5)

print(len(n_sacrifice_iter))


train_X = train_X.astype(int)
train_S = train_S.astype(int)
train_Y = train_Y.astype(int)


test_X = test_X.astype(int)
test_S = test_S.astype(int)
test_Y = test_S.astype(int)


pu_f1_scores = []
reg_f1_scores = []

for n_sacrifice in n_sacrifice_iter:

    print("PU transformation in progress.")

    train_S_pu = np.copy(train_S)
    pos = np.where(train_S == +1)[0]  
    np.random.shuffle(pos)
    sacrifice = pos[:n_sacrifice]
    train_S_pu[sacrifice] = -1
   
    estimator = RandomForestClassifier(n_estimators=100,
                                       criterion='gini', 
                                       bootstrap=True,
                                       n_jobs=1)

    pu_estimator = TransformTraditional(estimator)
    pu_estimator.fit(train_X,train_S_pu)

    y_pred = pu_estimator.predict(test_X)

    precision, recall, f1_score, _ = precision_recall_fscore_support(test_Y, y_pred)
    pu_f1_scores.append(f1_score[1])
    
    print("F1 score: ", f1_score[1])
    print("Precision: ", precision[1])
    print("Recall: ", recall[1])
    print
    
    print("Regular learning in progress...")

    estimator = RandomForestClassifier(n_estimators=100,
                                       bootstrap=True,
                                       n_jobs=1)


    estimator.fit(train_X,train_S_pu)
    y_pred = estimator.predict(test_X)
    precision, recall, f1_score, _ = precision_recall_fscore_support(test_Y, y_pred)
    reg_f1_scores.append(f1_score[1])
    print("F1 score: ", f1_score[1])
    print("Precision: ", precision[1])
    print("Recall: ", recall[1])
    print
    print


plt.title("Random forest with/without PU learning")
plt.plot(n_sacrifice_iter, pu_f1_scores, label='PU Adapted Random Forest')
plt.plot(n_sacrifice_iter, reg_f1_scores, label='Random Forest')
plt.xlabel('Number of positive examples hidden in the unlabled set')
plt.ylabel('F1 Score')
plt.legend()
plt.show()