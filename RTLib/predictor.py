import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from scipy.stats import anderson
import statistics


def sequentialPredictor(dataTrainX: pd.DataFrame,dataTrainY: pd.DataFrame,dataTestX: pd.DataFrame,dataTestY: pd.DataFrame,modelName):
    X=dataTrainX.to_numpy();
    y=dataTrainY.to_numpy();
    model = Sequential()
    length = len(dataTrainX.columns)
    lengthPrdictor = len(dataTrainY.columns)
    model.add(Dense(1024,input_dim=length,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(lengthPrdictor))

    print(model.summary())

    XTest = dataTestX.to_numpy()
    yTest = dataTestY.to_numpy()
    model.compile(loss='mean_absolute_error',optimizer='adam', metrics=['accuracy'])

    history =model.fit(X, y, epochs=60,batch_size=32*20,validation_data=(XTest,yTest))

    plot_loss(history,modelName+" loss graph")


    ynew = model.predict(XTest)
    print(ynew.shape)
    for i in range(len(XTest)):
        print("x=%s y=%s Predicted=%s" % (XTest[i],yTest[i], ynew[i]))

    #re-mering test data with predicted
    total =dataTestX.join(dataTestY)
    for i in range(0,len(dataTrainY.columns)):
        total.insert(loc=len(total.columns), column="y"+str(i),value=ynew[:,i])


    # re-mering train data with predicted
    ytrainNew = model.predict(X)
    total_ =dataTrainX.join(dataTrainY)
    for i in range(0,len(dataTrainY.columns)):
        total_.insert(loc=len(total_.columns), column="y"+str(i),value=ytrainNew[:,i])

    return model,total_,total


def plot_loss(history,title):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, .04])
  plt.title(title)
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.show()

def normalityTest(dataArray):
    result =anderson(dataArray,dist='norm')
    return result

