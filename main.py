import numpy as np
import preprocessing as pp
import training as train
import matplotlib.pyplot as plt

with open("text.txt","r",encoding="utf-8") as f:
    text = f.read()
tokens = pp.tokenize(text)
id_map,token_map = pp.map_token(tokens)
x,y = pp.training_data(tokens,token_map,2)
x = np.array(x)
y = np.array(y)

def get_embedding(model,word):
    try:
        idx = token_map[word]
    except KeyError:
        print ("Word doesn't exist")
        return
    one_hot = pp.one_hot_encode(idx,len(token_map))
    return train.forward(model,one_hot)["A1"]

network = train.create_model(len(token_map),10)

n_iters = 50
learn_rate = 0.001

print("Starting Training")
history = [train.backward(network,x,y,learn_rate) for _ in range(n_iters)]

plt.plot(range(len(history)), history)
plt.show()

#check word predictions
learning = pp.one_hot_encode(token_map["the"],len(id_map))
result = train.forward(network,[learning])["Z"][0]

for word in (id_map[id] for id in np.argsort(result)[::-1]):
    print(word)

#word weights
print(network["W1"])






