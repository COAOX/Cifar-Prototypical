import pickle
import os
import random
def choice(i):
    reed = random.random()
    ind = random.random()
    n = 20**-ind
    #if i<50:
        #n=1.0
    #else:
        #n=0.4
    #n = 0.8
    if reed<=n:
        return True
    return False

train = {}
test = {}

with open('cifar100/train','rb') as f:
    train = pickle.load(f, encoding='latin1')
train_data = train['data']
train_labels = train['fine_labels']
res_data = []
res_labels = []
for train_d, train_l in zip(train_data, train_labels):
	if not choice(train_l):
		continue
	res_data.append(train_d)
	res_labels.append(train_l)
train['data'] = res_data
train['fine_labels'] = res_labels
with open('cifar100/train','wb') as f:
	pickle.dump(train, f)

