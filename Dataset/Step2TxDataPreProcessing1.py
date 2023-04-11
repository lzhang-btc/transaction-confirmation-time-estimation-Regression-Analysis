import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import sys
sys.path.append("..")
import G_Variables
import os


blockHeightintx = G_Variables.blockHeightintx
confirmedtimeintx = G_Variables.confirmedtimeintx
feerateintx = G_Variables.feerateintx
enterBlockintx=G_Variables.enterBlockintx

feeintx=G_Variables.feeintx
waitingblockintx=G_Variables.waitingblockintx
intx=G_Variables.intx
outtx=G_Variables.outtx
vertx=G_Variables.vertx
sizeintx=G_Variables.sizeintx
weightintx=G_Variables.weightintx
receivetimeintx = G_Variables.receivetimeintx
relayintx=G_Variables.relayintx






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



start_selection=621500

print(start_selection)








def getHeight(tx_time, blockHeight, blockTime):
    blockNum=blockHeight.size
    start=0
    end=blockNum-1

    if tx_time > blockTime[start] or tx_time < blockTime[end]:
        pos = -2000
        return pos
    else:
        while start <= end:
            mid = (start + end) // 2
            if tx_time > blockTime[mid]:
                end = mid - 1
            elif tx_time < blockTime[mid]:
                start = mid + 1
            elif tx_time == blockTime[mid]:
                pos = mid
                break
        if (start > end):
            pos = end
        return blockHeight[pos]












totalblockData=pd.read_csv('1000BlockTotal.txt',header=None,sep=" ")
totalblockData=totalblockData[::-1]
blockHeight=totalblockData[blockHeightBinx]
blockTime=totalblockData[timeBinx]
totalblockData.to_csv('1000BlockTotal.csv',index=False,header=False)






# construct dataset




blockData=pd.read_csv('Block'+str(start_selection)+'.txt',header=None,sep=" ")
blockData=blockData[::-1]
blockData.to_csv('Block'+str(start_selection)+'.csv',index=False,header=False)


txdata = pd.read_csv('txinBlock'+str(start_selection)+'.txt',header=None,sep=" ")
txdata = txdata[::-1]
encoder=LabelEncoder()
txdata[relayintx]=encoder.fit_transform(txdata[relayintx])
recordsNo=txdata.shape[0]
txTime = txdata[receivetimeintx]
txNum=recordsNo

#Calculate enterheight(The height to construct next block)
txReceiveHeight=[]
for i in range(txNum):
    tx_time=txTime[i]
    tx_height=getHeight(tx_time,blockHeight,blockTime)
    txReceiveHeight.append(tx_height)
txdata[enterBlockintx]=pd.Series(txReceiveHeight)
#Calculate waiting block interval
txdata[waitingblockintx]=txdata[blockHeightintx]-txdata[enterBlockintx]+1
txdata.to_csv('txinBlock'+str(start_selection)+'.csv',index=False,header=False)

data=pd.read_csv('txinBlock'+str(start_selection)+'.csv',sep=",",names=['tx_index','vin_sz','vout_sz','ver','size', 'weight', 'time','relayed_by', 'lock_time','fee',
                                                 'block_height', 'block_index','confirmedtime','watingtime','feerate','enterBlock','watiingblock'])
data=data[data.feerate>0]
data1=data[data.enterBlock<0]
data1.to_csv('Invalid_txinblock'+str(start_selection)+'.csv',index=False,header=False)
data=data[data.enterBlock>0]
data['feerate']=4*data['fee']//data['weight']

data=data[data.watiingblock>=1]
data.to_csv('txinBlock'+str(start_selection)+'.csv',index=False,header=False)
