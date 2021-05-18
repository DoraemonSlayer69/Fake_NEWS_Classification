import pandas as pd
import  numpy as np
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,classification_report
from keras.utils import to_categorical
from keras import models
from keras import layers
import keras

def PlotGraph(history):
    import matplotlib.pyplot as plt



    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1,len(acc) + 1)

#Validation vs Training in accuracy
    plt.plot(epochs,acc,'bo',label='training accuracy')
    plt.plot(epochs,val_acc,'b',label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.figure()
    
    plt.plot(epochs,loss,'bo',label='training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.figure()


    plt.show()
dataset = pd.read_csv("D:/Personal/DeepLearning 1/data/OSNA Project/option1-data/train.csv")
test_dataset = pd.read_csv("D:/Personal/DeepLearning 1/data/OSNA Project/option1-data/test.csv")
#Conversion of text to numbers
#labels
y = dataset.iloc[:, -1].values
X = dataset.iloc[:, 3:-1].values

X_test = test_dataset.iloc[:,-2:].values

#converting array objects to a list
sentence_test = []
sentence = []
labels = []
max_len = 100  
for i in range(X.shape[0]):
    sentence.append(str(X[i]))

for k in range(X_test.shape[0]):
    sentence_test.append(str(X_test[k]))

y = y.reshape(y.shape[0],1)
onehot = OneHotEncoder()
y  = onehot.fit_transform(y).toarray()
'''
sample = []
for i in range(0,5):
    sample.append(text[i])
'''
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(sentence)
word_index_1 =tokenizer.word_index
seq_1 = tokenizer.texts_to_sequences(sentence)
seq_test = tokenizer.texts_to_sequences(sentence_test)
x = pad_sequences(seq_1,maxlen=max_len)

X_test = pad_sequences(seq_test,maxlen=max_len)
#Training on a smaller dataset first

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size = 1/3, random_state = 0)

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

'''
#OneHot encode the labels
y_train = y_train.reshape(y_train.shape[0],1)
y_val = y_val.reshape(y_val.shape[0],1)
onehot = OneHotEncoder()
y_train = onehot.fit_transform(y_train).toarray()
y_val = onehot.fit_transform(y_val).toarray()
'''

def initial_Model():
    network = models.Sequential()
    network.add(layers.Embedding(10000, 32))
    network.add(layers.SimpleRNN(32))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(3,activation='softmax'))
    network.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
    network.summary()
    return network


def LSTM_Model():
    network = models.Sequential()
    network.add(layers.Embedding(10000, 32))
    network.add(layers.LSTM(32, dropout=0.2,recurrent_dropout=0.2))
    network.add(layers.Dense(3, activation='softmax'))
    network.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
    network.summary()
    return network


def Bidirectional_LSTM():
    network = models.Sequential()
    network.add(layers.Embedding(10000, 32))
    network.add(layers.Bidirectional(layers.LSTM(units=64, recurrent_dropout=0.5,dropout=0.5,return_sequences=True)))
    network.add(layers.Bidirectional(layers.LSTM(64, dropout=0.2,recurrent_dropout=0.2)))
    network.add(layers.Dense(3, activation='softmax'))
    network.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
    network.summary()
    return network

def Bidirectional_LSTM_Hybrid_pooling():
    inputs = layers.Input(shape=(max_len,))
    embedding_layer = layers.Embedding(10000,32)(inputs)
    lstm1 = layers.Bidirectional(layers.LSTM(units=64, recurrent_dropout=0.5,dropout=0.5,return_sequences=True))(embedding_layer)
    avg_pool = layers.GlobalAveragePooling1D()(lstm1)
    max_pool = layers.GlobalMaxPooling1D()(lstm1)
    concat = layers.concatenate([avg_pool, max_pool])
    dropout = layers.Dropout(0.3)(concat)
    output = layers.Dense(3, activation="softmax")(dropout)
    model = keras.Model(inputs=inputs,outputs=output)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
    return model


network = initial_Model()
network.summary()

#network = initial_Model()

#network.load_weights("LSTM_Bidirectional_weigths.h5") 
'''
history = network.fit(X_train,y_train,batch_size = 512, validation_data=(X_val,y_val),epochs=2)
val_acc = np.mean(history.history['val_acc'])

PlotGraph(history)
'''
#network.save("SimpleRnn.rb")
#network.save("LSTM_Model")
#network = load_model("SimpleRnn.rb")

#to load the LSTM model
#network = load_model("LSTM_Model")

y_predict_val = network.predict(X_val)
result = network.evaluate(X_val,y_val)
y_predict = []
y_true = []
for i in range(y_predict_val.shape[0]):
    y_predict.append(np.argmax(y_predict_val[i])) 
    

for i in range(y_val.shape[0]):
    y_true.append(np.argmax(y_val[i]))
    

'''    
print(confusion_matrix(y_true, y_predict))
print(precision_recall_fscore_support(y_true, y_predict))
'''
print(classification_report(y_true, y_predict))


#Obtaining predictions on the test data

X_test = X_test.astype('float32')
y_predict = network.predict(X_test)
test_predictions = []
for i in range(y_predict.shape[0]):
    test_predictions.append(np.argmax(y_predict[i]))
    


for i in range(len(test_predictions)):
    if test_predictions[i] == 2:
        test_predictions[i] = 'unrelated'
    elif test_predictions[i] == 0:
        test_predictions[i] = 'agreed'
    else:
        test_predictions[i] = 'disagreed'
        
    
test_predictions = np.array(test_predictions)

ids = test_dataset.iloc[:, 0].values

ids = ids.reshape((ids.shape[0],1)).astype('int32')
test_predictions = test_predictions.reshape((test_predictions.shape[0],1))

submission = np.concatenate((ids,test_predictions),axis=1)

submission = pd.DataFrame(submission)


submission.to_csv("submission.csv",header=['id','label'],index=False)
