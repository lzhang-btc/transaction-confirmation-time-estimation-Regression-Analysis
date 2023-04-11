'''
Created on 4 Jun. 2019

@author: limengzhang
'''

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append("..")
import G_Variables



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
feerateintx = G_Variables.feerateintx
enterBlockintx=G_Variables.enterBlockintx
waitingblockintx=G_Variables.waitingblockintx
#Because of locktime info
validtimeintx=G_Variables.validtimeintx
validblockintx=G_Variables.validblockintx
validwaitingintx=G_Variables.validwaitingintx

lastBlockIntervalintx=G_Variables.lastBlockIntervalintx









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



##########Block new feature
# intervalBinx: interval since last block
#valid_weightBinx: sum of tx weight in block
#valid_sizeBinx: sum of tx weight in block
# avg_feerateBinx= 4*overallfee/valid_weightBinx
# avg_waitingtime: average confirmation time for txs in a block
# med_waitingtime: median confirmation time for txs in a block
totalBlockfile='1000BlockTotal.csv'


def getHeightTime(blockfile,blockheight):
    blockdata = pd.read_csv(blockfile, sep=",",header=None)
    block_array=np.array(blockdata)
    heighttime = -1
    for i in range(block_array.shape[0]):
        if block_array[i][blockHeightBinx]==blockheight:
            heighttime=block_array[i][timeBinx]
            break
    return heighttime

for start_selection in [621500,622000,622500]:
    print(start_selection)
    txfile='../TimetxinBlock'+str(start_selection)+'.csv'
    txdata = pd.read_csv(txfile, sep=",",header=None)
    txdata_array=np.array(txdata)
    recordsNum=txdata_array.shape[0]

    lastBlockInterval=[]
    for i in range(recordsNum):
        enterBlock=txdata_array[i][validblockintx]
        LastBlockTime=getHeightTime(totalBlockfile,enterBlock-1)
        lastBlockInterval.append(txdata_array[i][validtimeintx]-LastBlockTime)

    txdata[lastBlockIntervalintx]=pd.Series(lastBlockInterval)
    txdata.to_csv('../newTimetxinBlock'+str(start_selection)+'.csv', index=False, header=False)
