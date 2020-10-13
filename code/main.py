import sys
import numpy as np
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization

from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from utilities import *
  

def greedySearch(photo):
    in_text = 'startseq'
    filename = '../artifacts/Flickr_8k.trainImages.txt'
    train = load_set(filename)
    train_descriptions = load_clean_descriptions('../artifacts/descriptions.txt', train)
    all_train_captions = []
    for key, val in train_descriptions.items():
        for cap in val:
            all_train_captions.append(cap)
    word_count_threshold = 10
    word_counts = {}
    nsents = 0
    for sent in all_train_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    ixtoword = {}
    wordtoix = {}
    ix = 1
    for w in vocab:    
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    
    for i in range(34):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=34)
        deepmodels=deepmodel()
        deepmodels.load_weights('../artifacts/model_30.h5')
        yhat = deepmodels.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final



def predict():
    model = InceptionV3(weights='../artifacts/inception1.h5') #'imagenet'
    model_new = Model(model.input, model.layers[-2].output)
    #model_new.load_weights()
    image = preprocess(sys.argv[2]) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    mdesc=greedySearch(fea_vec)
    print("\n\nMachine description: ",mdesc)
    return mdesc


def main():
    # construct the argument parser and parse the arguments
    if len(sys.argv) <= 2:
        print("You must set argument!!!")
        print("example: python main.py --predict ../data/1020651753_06077ec457.jpg")
    else:
        mdesc=predict()
        X = mdesc.lower() 
        Y = input("\nEnter human description separated by space: ").lower() 
        # tokenization 
        X_list = word_tokenize(X)  
        Y_list = word_tokenize(Y) 
        # sw contains the list of stopwords 
        sw = stopwords.words('english')  
        l1 =[];l2 =[]  
        # remove stop words from string 
        X_set = {w for w in X_list if not w in sw}  
        Y_set = {w for w in Y_list if not w in sw} 
  
        # form a set containing keywords of both strings  
        rvector = X_set.union(Y_set)  
        for w in rvector: 
            if w in X_set: l1.append(1) # create a vector 
            else: l1.append(0) 
            if w in Y_set: l2.append(1) 
            else: l2.append(0) 
            c = 0
  
        # cosine formula  
        for i in range(len(rvector)): 
                c+= l1[i]*l2[i] 
        cosine = c / float((sum(l1)*sum(l2))**0.5) 
        print("\nsimilarity: ", cosine) 


if __name__ == '__main__':
    main()