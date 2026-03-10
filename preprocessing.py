import re
import numpy as np

TOKEN_PATTERN = re.compile(r'[\w\']*[a-z]+[\w^\']*')

def tokenize(text):
    text = text.lower()
    return TOKEN_PATTERN.findall(text)

def map_token(tokens):
    id_map = {}
    token_map= {}
    for i,token in enumerate(set(tokens)):
        id_map[i] = token
        token_map[token] = i
    return id_map,token_map

def one_hot_encode(id,size):
    res = np.zeros(size)
    res[id] = 1
    return res

def training_data(tokens,token_map,window):
    x = []
    y = []
    for i in range(len(tokens)):
        indexes = []
        for idx in range(max(0,i-window),min(len(tokens),i+window+1)):
            indexes.append(idx)
        for j in indexes:
            if i != j:
                x.append(one_hot_encode(token_map[tokens[i]],len(token_map)))
                y.append(one_hot_encode(token_map[tokens[j]],len(token_map)))
    return x,y


tokens = tokenize("I have the world's ugliest dog and I've lost it ''3'3'3224; 123Hurrah")
id_map,token_map = map_token(tokens)
x,y = training_data(tokens,token_map,2)
x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)
