import Bounding_box_predictor
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import ctc_decode, get_value
import numpy as np

model = load_model("./Models/converter_model.hdf5")

def Convert(path):
    sentences = Bounding_box_predictor.getSentence(path,False)
    sentences.sort(key = lambda s : (s[0],s[1]))
    words = []
    for x,y,sentence in sentences:
        sentence = sentence*255
        t=model.predict(sentence[np.newaxis,:,:,np.newaxis])
        v=get_value( ctc_decode(t,(t.shape[1],),greedy=True,top_paths=10)[0][0])
        pred=list(map(lambda c:' ' if c==36 else chr(int(c)+55) if c>9 else str(int(c)),v[0]))
        #true=list(map(lambda c:' ' if c==36 else chr(int(c)+55) if c>9 else str(int(c)),yh))
        word = ''.join([w for w in pred if w !='-1'])
        words.append([x,y,word])
        #print(''.join(true))
    py = words[0][1]
    for x,y,word in words:
        if y<py:
            print()
        print(word,end=" ")
        py = y


Convert("./TestData/10.jpg")

