import RTLib
from RTLib import core
from RTLib import predictor


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate
import seaborn as sns
import os


def main():
    # seed value for any random generation
    seed =123
    # file read chunk size- due to heavy file reading, pandas need to load as chunk
    chunkSize = 10 ** 5

    data = pd.DataFrame()  # pandas dataframe to collect data from file

    vehicleType ="5-RD"#    # model run on  selected vehicle type (other options :"9-LH","9-RD","4-RD","4-LH","10-LH" )

    displaySummaryData = False  # variable to control displaying summary data of each column

    displayScatterPlots = False # variable to controll displaying matrix scatterplots


    # data fields interested
    dataFields = ["OEM_PK_Vehicle", "Engine_FuelType", "Engine_RatedPower_kw", "Engine_Displacement_ltr",
                  "Engine_IdlingSpeed_rpm", "Engine_RatedSpeed_rpm", "LegislativeClass", "AxleConfiguration",
                  "VehicleGroup", "GrossVehicleMass_t", "CurbMassChassis_kg", "VocationalVehicle", "SleeperCab",
                  "Gearbox_TransmissionType", "Gearbox_GearsCount", "Gearbox_TransmissionRatioFinalGear", "RDL_CO2_gkm",
                  "RDR_CO2_gkm", "LHL_CO2_gkm", "LHR_CO2_gkm", "UDL_CO2_gkm", "UDR_CO2_gkm", "REL_CO2_gkm",
                  "RER_CO2_gkm", "LEL_CO2_gkm", "LER_CO2_gkm", "MUL_CO2_gkm", "MUR_CO2_gkm", "COL_CO2_gkm",
                  "COR_CO2_gkm", "VehicleSubgroup", "CO2v"]

    dataFieldsWithOutKey = ["Engine_FuelType", "Engine_RatedPower_kw", "Engine_Displacement_ltr",
                  "Engine_IdlingSpeed_rpm", "Engine_RatedSpeed_rpm", "LegislativeClass", "AxleConfiguration",
                  "VehicleGroup", "GrossVehicleMass_t", "CurbMassChassis_kg", "VocationalVehicle", "SleeperCab",
                  "Gearbox_TransmissionType", "Gearbox_GearsCount", "Gearbox_TransmissionRatioFinalGear", "RDL_CO2_gkm",
                  "RDR_CO2_gkm", "LHL_CO2_gkm", "LHR_CO2_gkm", "UDL_CO2_gkm", "UDR_CO2_gkm", "REL_CO2_gkm",
                  "RER_CO2_gkm", "LEL_CO2_gkm", "LER_CO2_gkm", "MUL_CO2_gkm", "MUR_CO2_gkm", "COL_CO2_gkm",
                  "COR_CO2_gkm", "VehicleSubgroup", "CO2v"]

    # predictor fields for model
    PredictorFields =["RDL_CO2_gkm","RDR_CO2_gkm","LHL_CO2_gkm","LHR_CO2_gkm"]

    # sub set of independent field used for the model
    IndependantVarFields =["Engine_RatedPower_kw", "Engine_Displacement_ltr", "Engine_IdlingSpeed_rpm", "Engine_RatedSpeed_rpm","GrossVehicleMass_t", "CurbMassChassis_kg","Gearbox_TransmissionRatioFinalGear","Gearbox_TransmissionType","Gearbox_GearsCount"]



   #when multiple times running code; if specific data set been loaded from master dataset, WE can use it directly
   # this is used to reduce file loading and parameter conversion time on each run
    if os.path.exists("../data/cleaned.csv"):
        data = core.readData("../data/cleaned.csv",chunkSize,dataFields) #preloade/saved data (delete file to load from master)
    else:
        data = core.readData('../data/CO2EmissionHDV_VehicleExtract_02062021.csv',chunkSize,dataFields) #original master data


    # cleaning  data
    # step one remove unwanted data; we are only interest in Diesel vehicles
    data_diesel =data[data["Engine_FuelType"]=="Diesel CI"]
    print(data_diesel["VehicleSubgroup"].describe())

    # select Vehicle type to run the model
    data_diesel = data_diesel[data_diesel["VehicleSubgroup"]==vehicleType]
    #remove specific C02 not recoded data, if CO2 emission data not avaible means that sample has little use in model train and validation
    data_diesel = data_diesel.dropna(subset=['CO2v'])

    # remove duplicated if Available (here data set key is removed for selecting duplicates)
    data_diesel.drop_duplicates(dataFieldsWithOutKey, keep='last')

    #save cleared data
    if not (os.path.exists("../data/cleaned.csv")):
        data_diesel.to_csv("../data/cleaned.csv",index=False)


    # Data Discription

    if displaySummaryData:
        core.displaySummaryData(data_diesel)

    if displayScatterPlots:
        for i in PredictorFields:
            core.displayScatter(data_diesel,["Engine_RatedPower_kw","Engine_Displacement_ltr","Engine_RatedSpeed_rpm",i])
            core.displayScatter(data_diesel,["GrossVehicleMass_t", "CurbMassChassis_kg", "Gearbox_GearsCount", i])

    #print(tabulate(dataTrainY_.head(500), headers = 'keys', tablefmt = 'psql'))  # visualize data in terminal

    # recoding categorical data to normalize
    categoricalGroups =["LegislativeClass","AxleConfiguration","VehicleGroup","VocationalVehicle","SleeperCab","Gearbox_TransmissionType","VehicleSubgroup"]
    core.recode(data_diesel,categoricalGroups)


    # Select data column to fed model Independent variables and dependent variables
    dataTrain = data_diesel[IndependantVarFields]
    dataPred= data_diesel[PredictorFields]

    #Data Normalization

    dataPred = dataPred/3000.0   # CO2 emissions for each mission profile normalized only by scale down

    dataTrain =(dataTrain-dataTrain.min())/(dataTrain.max()-dataTrain.min()) # linear normalization


    # join dataframes for train and test sampling
    dataTrain =dataTrain.join(dataPred).head(20000) #


    dataTrainX = dataTrain.sample(frac=0.8,random_state=seed)  # training data sepection 80% to 20%

    dataTestX = dataTrain.drop(dataTrainX.index)  # test Data set

    print(dataTestX.describe().transpose())
    print(dataTrainX.describe().transpose())


    # y Variables
    dataTrainY_ = dataTrainX[PredictorFields]
    dataTestY_ =dataTestX[PredictorFields]

    # X variables
    dataTrainX_ = dataTrainX.drop(columns=PredictorFields)
    dataTestX_ = dataTestX.drop(columns=PredictorFields)

    #calling DNN function to train
    model,train,test =predictor.sequentialPredictor(dataTrainX_, dataTrainY_, dataTestX_, dataTestY_,vehicleType)

    test.to_csv("../data/test_res_" + vehicleType + ".csv", index=False)
    train.to_csv("../data/train_res_" + vehicleType + ".csv", index=False)

    model.save('../saved_model/'+str(vehicleType))


# visual validation
def testResult():

    vehicleType ="5-RD"#"9-LH"#"9-RD"#"4-RD" #"4-LH"#"10-LH"
    data_test = pd.DataFrame()
    data_train = pd.DataFrame()
    chunkSize = 10 ** 5
    dataFields = [ "Engine_RatedPower_kw", "Engine_Displacement_ltr",
                  "Engine_IdlingSpeed_rpm", "Engine_RatedSpeed_rpm", "GrossVehicleMass_t", "CurbMassChassis_kg",
                  "Gearbox_TransmissionType", "Gearbox_GearsCount", "Gearbox_TransmissionRatioFinalGear", "RDL_CO2_gkm",
                  "RDR_CO2_gkm", "LHL_CO2_gkm", "LHR_CO2_gkm","y0","y1","y2","y3"]

    PredictorFields =["RDL_CO2_gkm","RDR_CO2_gkm","LHL_CO2_gkm","LHR_CO2_gkm"]

    # read dumped test data
    if os.path.exists("../data/test_res_"+vehicleType+".csv"):
        data_test = core.readData("../data/test_res_"+vehicleType+".csv", chunkSize, dataFields)
    # read dumped training data
    if os.path.exists("../data/train_res_"+vehicleType+".csv"):
        data_train = core.readData("../data/train_res_"+vehicleType+".csv", chunkSize, dataFields)

    # plotting training set and test set in same scatter plot
    for i in range(0,4):
        plt.scatter(data_train[PredictorFields[i]], data_train[["y"+str(i)]], marker='o', label="train")
        plt.scatter(data_test[PredictorFields[i]],data_test[["y"+str(i)]],marker='+',label="test")
        plt.xlabel("actual")
        plt.ylabel("predicted")
        plt.legend()
        plt.title(vehicleType+" "+PredictorFields[i]+" validation graph")
        plt.savefig("../assets/"+vehicleType+" "+PredictorFields[i]+" validation graph.png")
        plt.show()

    # mission profiles taken from Regulation (EU) 198/.... this is needed for Specific CO2 emission calculation
    dataMissionProfile =pd.read_csv("../data/missionProfile.csv")

if __name__ == "__main__":
    main()
    testResult()
