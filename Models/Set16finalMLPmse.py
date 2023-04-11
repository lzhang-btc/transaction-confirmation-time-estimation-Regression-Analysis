'''
Created on 4 Jun. 2019

@author: limengzhang
'''
import multiprocessing
from keras.utils.vis_utils import plot_model

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from TestFeeClass.SharedFunction import mean_absolute_percentage_error
from TestFeeClass.SharedFunction import NpEncoder
import keras
import math
import random
from keras.layers import Input, Embedding, LSTM,GRU, Dense,Dropout,Bidirectional
from keras.models import Model
from keras.layers.core import Flatten
from keras_self_attention import SeqWeightedAttention

from math import sqrt
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

from keras.layers import Dense, Lambda, dot, Activation, concatenate
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import time
import random
import numpy as np


import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

sys.path.append("..")
import G_Variables
import os


###tx feature
confirmedtimeintx = G_Variables.confirmedtimeintx
feerateintx = G_Variables.feerateintx
enterBlockintx=G_Variables.enterBlockintx
waitingblockintx=G_Variables.waitingblockintx
intx=G_Variables.intx
outtx=G_Variables.outtx
vertx=G_Variables.vertx
sizeintx=G_Variables.sizeintx
weightintx=G_Variables.weightintx
receivetimeintx = G_Variables.receivetimeintx
relayintx=G_Variables.relayintx
lockintx=G_Variables.lockintx
feeintx=G_Variables.feeintx
blockHeightintx = G_Variables.blockHeightintx
confirmedtimeintx = G_Variables.confirmedtimeintx
waitingtimeinx= G_Variables.waitingtimeinx
feerateintx = G_Variables.feerateintx
enterBlockintx=G_Variables.enterBlockintx
waitingblockintx=G_Variables.waitingblockintx
#Because of locktime info
validtimeintx=G_Variables.validtimeintx
validblockintx=G_Variables.validblockintx
validwaitingintx=G_Variables.validwaitingintx
#RelatedTo observation time
lastBlockIntervalintx=G_Variables.lastBlockIntervalintx## obsertime-latblocktime


###block feature
blockHeightBinx=G_Variables.blockHeightBinx
n_txBinx=G_Variables.n_txBinx
sizeBinx=G_Variables.sizeBinx
bitsBinx=G_Variables.bitsBinx
feeBinx=G_Variables.feeBinx
verBinx=G_Variables.verBinx
timeBinx=G_Variables.timeBinx
intervalBinx=G_Variables.intervalBinx
valid_weightBinx=G_Variables.valid_weightBinx
valid_sizeBinx=G_Variables.valid_sizeBinx
avg_feerateBinx=G_Variables.avg_feerateBinx
avg_waitingBinx=G_Variables.avg_waitingBinx
med_waitingBinx=G_Variables.med_waitingBinx


##Mempoolfile Feature
blockHeightMinx=0

# ######Scanning timeinterval(s)
# train_timeinterval=10*60
# test_timeinterval=60
# ######





training_blocks=G_Variables.training_blocks
lstmunits=G_Variables.lstmunits
lstmtimestamps=G_Variables.lstmtimestamps
layers=G_Variables.layers
prediction_epoch=G_Variables.prediction_epoch*5
bachsize=G_Variables.bachsize



optimizer_model=G_Variables.optimizer_model
dropout_factor=G_Variables.dropout_factor
lossfunction='mse'
target = 'FEL'


######FilePath
TestGroup=16
TestGroup=str(TestGroup)

START_EstimateBlock=getattr(G_Variables,'START_EstimateBlock_S'+TestGroup)
total_EstimateBlock=45








result_path = getattr(G_Variables,'result_path_S'+TestGroup)

blockfile= '../'+getattr(G_Variables,'blockfile_S'+TestGroup)

txfile= '../'+getattr(G_Variables,'txfile_S'+TestGroup)
totalblockfile='../1000BlockTotal.csv'
SelectionModule='Time'


dir_abbv='finalMLPBlocksigmoid'

result_path='.'+result_path
dirs= result_path+SelectionModule+dir_abbv

timeinterval=random.uniform(1,5.0)
time.sleep(timeinterval)
if not os.path.exists(dirs):
    os.makedirs(dirs)
result_path=result_path+SelectionModule+dir_abbv+'/'



weightVectorLen=100

addaccount=-64


currentHeight = START_EstimateBlock + addaccount








TxFeatureSelection=[intx, outtx, vertx, sizeintx, weightintx, relayintx,feeintx,feerateintx,lastBlockIntervalintx]
#TxFeatureSelection=[intx, outtx, vertx, sizeintx, weightintx, relayintx,feeintx,feerateintx]
# Tx will append a dim presenting the unconfirmed weights of txs with higher feerate trasanctions in the mempool.
TxFeaLens=len(TxFeatureSelection)####additional 1 dim for higherfeerate weights




PredictSelection=[waitingtimeinx]
# it represents the time interval between current time and its final confirmation block time












def MLPBlock(layers,txFeatureDim):
  np.random.seed(9)
  model2_auxiliary_input = Input(shape=(txFeatureDim,), name='model2_tx_input')

  model2_merged_vector = model2_auxiliary_input
  model2_merged_vector = Dropout(dropout_factor)(model2_merged_vector)
  model2_layer1_vector = Dense(layers[0],kernel_initializer='uniform', activation='relu')(model2_merged_vector)
  model2_layer1_vector=Dropout(dropout_factor)(model2_layer1_vector)
  model2_layer2_vector = Dense(layers[1], activation='relu')(model2_layer1_vector)
  model2_layer2_vector=Dropout(dropout_factor)(model2_layer2_vector)
  model2_predictions = Dense(layers[2], activation='sigmoid')(model2_layer2_vector)
  model2 = Model(inputs=[model2_auxiliary_input], outputs=model2_predictions)
  model2.compile(loss=lossfunction, optimizer=optimizer_model,metrics=[lossfunction])
  return model2



def getHeightTime(totalblockfile,startEsitimateBlock):
    totalblock = pd.read_csv(totalblockfile, sep=",", header=None)
    block=totalblock[totalblock[blockHeightBinx]==startEsitimateBlock]
    HeightTime=block[timeBinx]
    return HeightTime.values[0]


def getLastBlockTime(totalblockfile,obserTime):
    totalblock = pd.read_csv(totalblockfile, sep=",", header=None)
    totalblock_array=np.array(totalblock)
    for blc in range(totalblock_array.shape[0]):
        if totalblock_array[blc][timeBinx]<obserTime and totalblock_array[blc+1][timeBinx]>=obserTime:
            break
    lastBlockHeight=totalblock_array[blc][blockHeightBinx]
    lastBlocktime=totalblock_array[blc][timeBinx]
    return lastBlockHeight,lastBlocktime


















def txDatasetConstruction(txfile,startSearchBlock,endSearchBlock):
  txcollection = pd.read_csv(txfile, sep=",",header=None)
  txcollection = txcollection[txcollection[waitingtimeinx]>0]



  txfeatureList=[]
  txOutputList=[]
  blockSeqList=[]
  memSeqList=[]

  for h_index in range(startSearchBlock,endSearchBlock):
      txsSelected = txcollection[txcollection[enterBlockintx] ==h_index]
      txsSelected = txsSelected.copy()
      txsArray = np.array(txsSelected)
      UnconfirmedTxs=txcollection[(txcollection[enterBlockintx] <=h_index)&(txcollection[blockHeightintx] >=h_index)]
      for tx in txsArray:
          feerate=tx[feerateintx]
          weight=tx[weightintx]
          higherUnconftx=np.array(UnconfirmedTxs[UnconfirmedTxs[feerateintx]>=feerate])
          higherWeights=sum(higherUnconftx[:,weightintx])-weight####Minplus selfWeight
          txfeatureay= tx[TxFeatureSelection]
          fea_list=txfeatureay.tolist()
          fea_list.append(higherWeights)
          txfeatureList.append(fea_list)

      txOutputArray = txsArray[:, PredictSelection]
      txOutputList.extend(txOutputArray.tolist())
      
  return txfeatureList,txOutputList



def CheckInvalidValue(txOutputList):
    temp=[]
    for output in txOutputList:
        if output==0:
            # mape may occure invalid value when the true output is zero
            output=output+math.exp(-20)
        temp.append(output)

    return temp



####Construct Training and Testing dataset


train_txfeatureList1,train_txOutputList1=txDatasetConstruction(txfile,currentHeight-training_blocks,currentHeight+total_EstimateBlock)
#######Scale Features
trn_X1=np.array(train_txfeatureList1)
trn_Y1=np.array(train_txOutputList1)
#Normalization
ss_x= MinMaxScaler()
strain_txfeatureList1 = ss_x.fit_transform(trn_X1)





train_txfeatureList,train_txOutputList=txDatasetConstruction(txfile,currentHeight-training_blocks,currentHeight)
#######Scale Features
trn_X=np.array(train_txfeatureList)
trn_Y=np.array(train_txOutputList)
#Normalization
strain_txfeatureList = ss_x.transform(trn_X)



def scaleOutput(txfile, trn_Y):
    txcollection = pd.read_csv(txfile, sep=",", header=None)
    txcollection = txcollection[txcollection[waitingtimeinx] > 0]
    txcollection = txcollection[(txcollection[enterBlockintx] >=currentHeight-training_blocks)&(txcollection[enterBlockintx] <=currentHeight+total_EstimateBlock)]
    temp=np.array(txcollection)
    tmax=max(temp[:,PredictSelection])+1
    strn_Y=trn_Y/tmax
    return strn_Y.reshape(-1),tmax


strain_txOutputList,tmax=scaleOutput(txfile,trn_Y)

strain_txOutputList = strain_txOutputList.tolist()
strain_txOutputList=CheckInvalidValue(strain_txOutputList)








fel_model = MLPBlock(layers, TxFeaLens+1)
plot_model(fel_model, to_file=dir_abbv+'.png', show_shapes=True)




filepath_fel = './TrainedModels/Set'+str(TestGroup)+str(currentHeight)+'MLP.h5'



checkpoint = ModelCheckpoint(
  filepath=filepath_fel,
  monitor='loss',
  save_best_only=True,
  verbose=0,
  mode='auto',
  save_weights_only=False,
  period=1)
print('16MLP')
print(len(strain_txOutputList))
currenttime=time.time()
fel_hist = fel_model.fit([ strain_txfeatureList], strain_txOutputList, epochs=prediction_epoch,
                         batch_size=bachsize, verbose=0,callbacks=[checkpoint])

endttime=time.time()
print(str(endttime-currenttime))







Result = {}
# for i in range(total_EstimateBlock):

Result[str(currentHeight)] = {}
# for i in range(1):
    # update_model= load_model(result_path + SelectionModule + lossfunction + target+str(lastUpdateHeight)+'.h5', custom_objects=SeqSelfAttention.get_custom_objects())


test_txfeatureList,test_txOutputList= txDatasetConstruction(txfile,currentHeight, currentHeight + 1)
tst_X = np.array(test_txfeatureList)
tst_Y = np.array(test_txOutputList)




Result[str(currentHeight)]['count'] = len(test_txOutputList)
print(Result[str(currentHeight)]['count'])

if len(test_txOutputList)>0:

    stest_txfeatureList = ss_x.transform(tst_X)
    Result[str(currentHeight)][dir_abbv] = []

    pre_y = fel_model.predict([stest_txfeatureList])
    Ogn_tst_Y = np.array(tst_Y).reshape(-1)
    Ogn_pre_y = np.array(pre_y).reshape(-1, 1).reshape(-1) * tmax


    model_metrics_name = [mean_absolute_error, mean_squared_error]

    for mdl in model_metrics_name:
        #
        Ogn_tmp_score = mdl(Ogn_tst_Y, Ogn_pre_y)
        Result[str(currentHeight)][dir_abbv].append(Ogn_tmp_score)
    Result[str(currentHeight)][dir_abbv].append(mean_absolute_percentage_error(Ogn_tst_Y, Ogn_pre_y))
else:
    Result[str(currentHeight)][dir_abbv] = [0,0,0]

with open('../ResultAnalysisRW03NNmodels/'+'S' + TestGroup + 'ResultAnalysis' + dir_abbv  + str(currentHeight) + '.txt', 'w') as file:
    file.write(json.dumps(Result, cls=NpEncoder))
    file.write('\n')






# for i in range(total_EstimateBlock-1):
    # update_model= load_model(result_path + SelectionModule + lossfunction + target+str(lastUpdateHeight)+'.h5')
    # test_txfeatureList, test_txOutputList = txDatasetConstruction(txfile,lastUpdateHeight, lastUpdateHeight + 1)
    # tst_X = np.array(test_txfeatureList)
    # tst_Y = np.array(test_txOutputList)
    # stest_txfeatureList = ss_x.transform(tst_X)
    # stest_txOutputList = ss_y.transform(tst_Y.reshape([-1, 1])).reshape(-1)
    # stest_txfeatureList = stest_txfeatureList.tolist()
    # stest_txOutputList = stest_txOutputList.tolist()
    # stest_txOutputList=CheckInvalidValue(stest_txOutputList)
    # lastUpdateHeight = lastUpdateHeight + 1
    # filepath_fel = result_path + SelectionModule + lossfunction + target + str(lastUpdateHeight) + '.h5'
    # if len(stest_txOutputList)==0:
        # update_model.save(filepath_fel)
    # else:
        # checkpoint = ModelCheckpoint(
            # filepath=filepath_fel,
            # monitor='loss',
            # save_best_only=True,
            # verbose=0,
            # mode='auto',
            # save_weights_only=False,
            # period=1)
        # update_model = update_model.fit([stest_txfeatureList], stest_txOutputList, epochs=200,
                                 # batch_size=20, verbose=0,callbacks=[checkpoint]
                                 # )



