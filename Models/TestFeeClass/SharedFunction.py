import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import json
import os
######加载全局变量
################Tx Features

intx=1
outtx=2
vertx=3
sizeintx=4
weightintx=5
receivetimeintx = 6
relayintx=7
lockintx=8
feeintx=9
blockHeightintx = 10
blockindexintx=11
confirmedtimeintx = 12
waitingtimeinx=13
feerateintx = 14
enterBlockintx=15
waitingblockintx=16
#Because of locktime info
validtimeintx=17
validblockintx=18
validwaitingintx=19
#RelatedTo observation time
lastBlockIntervalintx=20## obsertime-latblocktime(obseveBased)
waitedTimeintx=21# obsertime-receivetime
timeToConfirmintx=22# confirmtime-obsertime


#############Block Features
blockHeightBinx=1
n_txBinx=2
sizeBinx=3
bitsBinx=4
feeBinx=5
verBinx=6
timeBinx=7
intervalBinx=8
valid_weightBinx=9
valid_sizeBinx=10
avg_feerateBinx=11
avg_waitingBinx=12
med_waitingBinx=13


TxFeatureSelection=[intx, outtx, vertx, sizeintx, weightintx, relayintx,feeintx,feerateintx,lastBlockIntervalintx,waitedTimeintx]
####实际还需要添加与观测点相关的数据
####Input
#  Waiting time since last block
#  waitedtime since entering
####Output
# 还需要等待的确认时间=confrimationTime-观测点
BocFeatureSelection=[n_txBinx,sizeBinx,bitsBinx,intervalBinx,valid_weightBinx,valid_sizeBinx,avg_feerateBinx,avg_waitingBinx,med_waitingBinx]
PredictSelection=[timeToConfirmintx]

def mean_absolute_percentage_error(inv_y, inv_yHat):
    inv_y = np.array(inv_y)
    n = inv_y.shape[0]
    inv_yHat = np.array(inv_yHat)
    mape = sum(np.abs((inv_y - inv_yHat) / inv_y)) / n * 100
    return mape

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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def calculateinterval(train_timeinterval):
    if train_timeinterval/60==3:
        intervallabel=''
        t_value=''
    elif train_timeinterval/60==7:
        intervallabel = '7min'
        t_value='420'
    elif train_timeinterval/60==10:
        intervallabel = '10min'
        t_value='600'
    return intervallabel,t_value

def FindBestModelFromCheckPointByInterval(modelpath, maxEpoch):
    items = os.listdir(modelpath)
    ep_best = 1
    model_best = ''
    for names in items:
        if names.endswith('.h5'):
            begin = names.find('_')
            end = names.find('.')
            prev = names[begin + 1:end]
            ep = int(prev)
            if ep >= ep_best and ep <= maxEpoch:
                ep_best = ep
                model_best = names
    return model_best
