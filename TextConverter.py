import Bounding_box_predictor
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import ctc_decode, get_value
import numpy as np

model = load_model("./Models/converter_model.hdf5")

def Convert(path):
    sentences = Bounding_box_predictor.getSentence(path,True)
    for sentence in sentences:
        sentence = sentence*255
        t=model.predict(sentence[np.newaxis,:,:,np.newaxis])
        v=get_value( ctc_decode(t,(t.shape[1],),greedy=True,top_paths=10)[0][0])
        pred=list(map(lambda c:' ' if c==36 else chr(int(c)+55) if c>9 else str(int(c)),v[0]))
        #true=list(map(lambda c:' ' if c==36 else chr(int(c)+55) if c>9 else str(int(c)),yh))
        print(''.join(pred))
        #print(''.join(true))


Convert("./TestData/3.jpg")

