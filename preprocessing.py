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

text = "English is a West Germanic language that emerged in early medieval England and has since become a global lingua franca.[4][5][6] The namesake of the language is the Angles, one of the Germanic peoples who migrated to Britain after the end of Roman rule. English is the most spoken language in the world, primarily due to the global influences of the former British Empire (succeeded by the Commonwealth of Nations) and the United States. It is the most widely learned second language in the world, with more second-language speakers than native speakers. However, English is only the third-most spoken native language, after Mandarin Chinese and Spanish.[3]"
tokens = tokenize(text)
id_map,token_map = map_token(tokens)
x,y = training_data(tokens,token_map,2)
x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)
