import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def getVersion():
    return "0.1.0.1"

def readMetaData(fileName):
    data = pd.DataFrame()
    chunk =pd.read_csv(fileName, nrowa=5)
    return chunk.columns



def readData(fileName,chunckSize,fields):
    i=0;
    data = pd.DataFrame()
    for chunk in pd.read_csv(fileName, chunksize=chunckSize):
        print("Read data chunk "+str(i))
        i=i+1
        data= pd.concat([data,process(chunk,fields)])
    return data


def process(chunk,fields):
    return chunk[fields]

def displaySummaryData(data : pd.DataFrame):
    arrayFields = np.array(data.columns)
    for i in range(len(arrayFields)):
        print("-" * 65)
        print(data[[arrayFields[i]]].value_counts())



def displayScatter(data : pd.DataFrame, fields ):
    fig, axs = plt.subplots(len(fields), len(fields))
    fig.tight_layout()

    for i in range(len(fields)):
        axs[i, i].scatter(data[fields[i]], data[fields[i]],s=5)
        axs[i, i].set_title(fields[i],fontsize=6)
        for j in range(i+1,len(fields)):
            #print("variable i "+str(i)+" variable j "+str(j))
            axs[i, j].scatter(data[fields[i]],data[fields[j]],s=5)
            axs[i, j].set_title(fields[i]+" vs "+fields[j], fontsize=6)

    plt.show()



def recode(data : pd.DataFrame, fields):
    for i in range(len(fields)):
        color_labels = data[fields[i]].unique()
        #print(fields[i])
        #print(len(color_labels))
        grades = np.arange(start=0, stop=len(color_labels), step=1)
        data[fields[i]] = data[fields[i]].replace(color_labels, grades)



def transformDataStdNormal(dataArray):
    mean_ = np.mean(dataArray)
    sigma = np.var(dataArray)
    data =(dataArray-mean_)/sigma
    return data


def normalizeLinear(dataArray):
    return dataArray -np.min(dataArray)/np.ptp(dataArray)