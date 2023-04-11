'''
Created on 4 Jun. 2019

@author: limengzhang
'''

import numpy as np

import pandas as pd
from scipy import stats

from sklearn.preprocessing import LabelEncoder

import sys
import joblib

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





feerate_Interval = 0.001
numBins = int(100/feerate_Interval)+1
### Set a flexibile fee limit
##622500>100 10%











blocks_overall=499
feerateBound=1
bins=100
feerate_space=0.1
####1---(0,3]
#   2---(3,3*1.1]
#   3---(3*1.1,3*1.1^2)....

IntervalLabel='ClassifiedFeerateBin'+str(bins)



#for start_selection in [621500,622000,622500]:
def CalculateArrayDis(txs_temp, bins,feerate_startbound,feerate_space):
    dis_vector=[]
    classNum=bins

    tx_c=0
    for i in range(classNum):
        upperbound=feerate_startbound*pow(1+feerate_space,i)
        if i==classNum-1:
            upperbound=1000000########The last bin 35 contains tx with feerate over the 35
        if i==0:
            lowerbound=0
        else:
            lowerbound=feerate_startbound*pow(1+feerate_space,i-1)
        slected_tx=txs_temp[np.where((txs_temp[:, feerateintx] < upperbound)&(txs_temp[:, feerateintx] >= lowerbound))]
        if slected_tx.shape[0]==0:
            dis_vector.append(0)
        else:
            dis_vector.append(sum(slected_tx[:,weightintx]))
        tx_c=tx_c+slected_tx.shape[0]
    if tx_c!=txs_temp.shape[0]:
        print('Logical Error')
    return dis_vector







def CalculateMemDis(start_selection, block_height,bins,feerate_startbound,feerate_space):
    txfile = '../TimetxinBlock' + str(start_selection) + '.csv'
    txcollection = pd.read_csv(txfile, sep=",", header=None)
    df_temp=txcollection[(txcollection[enterBlockintx]<=block_height) &(txcollection[blockHeightintx]>=block_height)]
    txs_temp=np.array(df_temp)
    cal_dis=CalculateArrayDis(txs_temp,bins,feerate_startbound,feerate_space)
    return cal_dis


def GenerateMemFeerateVector(start_selection,bins,feerate_startbound,feerate_space):
    currentHeight=start_selection-blocks_overall+1
    MemContents=[]
    for block_height in range(start_selection-blocks_overall+1,start_selection+1):
        temp_height=[]
        temp_height.append(block_height)
        memDis=CalculateMemDis(start_selection,block_height,bins,feerate_startbound,feerate_space)
        temp_height.extend(memDis)
        MemContents.append(temp_height)
    pd_contents=pd.DataFrame(np.array(MemContents),index=None)
    pd_contents.to_csv('../MemFeerateVectorTimeBlock'+str(start_selection)+'.csv',index=False,header=False)


def CalculateBloDis(start_selection, block_height, bins,feerate_startbound,feerate_space):
    txfile = '../TimetxinBlock' + str(start_selection) + '.csv'
    txcollection = pd.read_csv(txfile, sep=",", header=None)
    df_temp = txcollection[txcollection[blockHeightintx] == block_height]
    txs_temp = np.array(df_temp)
    cal_dis = CalculateArrayDis(txs_temp, bins,feerate_startbound,feerate_space)
    return cal_dis


def GenerateBlockFeerateVector(start_selection, bins,feerate_startbound,feerate_space):
    currentHeight=start_selection-blocks_overall+1
    BlockContents=[]
    blockfile = '../TimeBlock' + str(start_selection) + '.csv'
    block_df = pd.read_csv(blockfile, sep=",", header=None)
    for block_height in range(start_selection - blocks_overall+1, start_selection+1):
        print('GenerateBlock'+str(block_height))
        temp_height = []
        temp_block=np.array(block_df[block_df[blockHeightBinx]==block_height]).reshape(-1).tolist()
        temp_height.extend(temp_block)
        BloDis = CalculateBloDis(start_selection, block_height, bins,feerate_startbound,feerate_space)
        temp_height.extend(BloDis)
        BlockContents.append(temp_height)
    pd_contents=pd.DataFrame(np.array(BlockContents),index=None)
    pd_contents.to_csv('../FeerateVectorTimeBlock'+str(start_selection)+'.csv',index=False,header=False)



for start_selection in [621500]:
    print(start_selection)

    GenerateMemFeerateVector(start_selection,bins,feerateBound,feerate_space)
    GenerateBlockFeerateVector(start_selection,bins,feerateBound,feerate_space)



    print('hello')



