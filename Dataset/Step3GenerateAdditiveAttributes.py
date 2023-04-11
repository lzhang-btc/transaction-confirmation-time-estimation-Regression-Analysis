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
start_selection=621500
txfile='txinBlock'+str(start_selection)+'.csv'
blockfile='Block'+str(start_selection)+'.csv'
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




txdata = pd.read_csv(txfile, sep=",",
                   names=['tx_index', 'vin_sz', 'vout_sz', 'ver', 'size', 'weight', 'time', 'relayed_by', 'lock_time',
                          'fee',
                          'block_height', 'block_index', 'confirmedtime', 'watingtime', 'feerate', 'enterBlock',
                          'watiingblock'])
txdata['feerate'] = 4 * txdata['fee'] / txdata['weight']





txdata[validtimeintx]=txdata['time']
txdata[validblockintx]=txdata['enterBlock']
###Construct valiwaiting TIme based on locktime
txdata_array=np.array(txdata)
recordsNum=txdata_array.shape[0]
validTime = []
validBlock = []
for i in range(recordsNum):
    #print(i)
    # locktime<500000000, it refers to height. Otherwise, it represent unix timestamp
    if txdata_array[i][lockintx]<500000000 and txdata_array[i][enterBlockintx]<txdata_array[i][lockintx]:
        # when locktime is set as blockheight, tx can be  in Memepool at lockheight
        validTime.append(getHeightTime(totalBlockfile,txdata_array[i][lockintx]-1)+0.01) # small increment
        validBlock.append(txdata_array[i][lockintx])
    else:
        validTime.append(txdata_array[i][receivetimeintx])
        validBlock.append(txdata_array[i][enterBlockintx])

txdata[validtimeintx]=pd.Series(validTime)
txdata[validblockintx]=pd.Series(validBlock).astype('int')
txdata[validwaitingintx]=txdata['confirmedtime']-txdata[validtimeintx]


txdata2=txdata[txdata[validtimeintx]<0]
txdata2.to_csv('../invalidTimetxinBlock'+str(start_selection)+'.csv', index=False, header=False)
txdata=txdata[txdata[validtimeintx]>=0]
txdata.to_csv('../TimetxinBlock'+str(start_selection)+'.csv', index=False, header=False)

blockdataFram = pd.read_csv(blockfile, sep=",",
                   names=['block_index', 'height', 'n_tx', 'size', 'bits', 'fee', 'ver', 'time'])
blockdataFram['interval']=blockdataFram['time'].diff()
blockdataFram['n_tx']=blockdataFram['n_tx'].apply(lambda x: x-1)
encoder1=LabelEncoder()
blockdataFram['bits']=encoder1.fit_transform(blockdataFram['bits'])
encoder2=LabelEncoder()
blockdataFram['ver']=encoder2.fit_transform(blockdataFram['ver'])


######Intial Zero
# valid_weight
blockdataFram[valid_weightBinx] = blockdataFram['time'] - blockdataFram['time']
# valid_size
blockdataFram[valid_sizeBinx] = blockdataFram['time'] - blockdataFram['time']
# valid_feerate
blockdataFram[avg_feerateBinx] = blockdataFram['time'] - blockdataFram['time']
# avg and med  waiting time for txs in the whole block
blockdataFram[avg_waitingBinx]=blockdataFram['time'] - blockdataFram['time']
blockdataFram[med_waitingBinx]=blockdataFram['time'] - blockdataFram['time']

#####array operation
data = np.array(blockdataFram)
weightBlock=[]
sizeBlock=[]
avg_feerateBlock=[]
avg_waitingtimeinBlock=[]
med_waitingtimeinBlock=[]




block_records = data.shape[0]
for i in range(block_records):
    block_inx = data[i][1]
    temp_fram = txdata[txdata.block_height == block_inx]
    temp = np.array(temp_fram)
    if temp.shape[0] == 0:
        aggred_weight = 0
        aggred_size=0
        avg_feerate=0
        avg_time=0
        med_time=0
# med_waitingtime: median confirmation time for txs in a block
    else:
        aggred_fee = sum(temp[:, feeintx])
        aggred_weight = sum(temp[:, weightintx])
        aggred_size = sum(temp[:, sizeintx])
        avg_feerate=4 * aggred_fee / aggred_weight
        waitingtimeList=temp[:, validwaitingintx].tolist()
        waitingtimeList.sort()
        avg_time = np.mean(waitingtimeList)
        med_time=np.median(waitingtimeList)
    weightBlock.append(aggred_weight)
    sizeBlock.append(aggred_size)
    avg_feerateBlock.append(avg_feerate)
    avg_waitingtimeinBlock.append(avg_time)
    med_waitingtimeinBlock.append(med_time)


blockdataFram[valid_weightBinx] = pd.Series(weightBlock)
# valid_size
blockdataFram[valid_sizeBinx] = pd.Series(sizeBlock)
# valid_feerate
blockdataFram[avg_feerateBinx] = pd.Series(avg_feerateBlock)
# avg and med  waiting time for txs in the whole block
blockdataFram[avg_waitingBinx]=pd.Series(avg_waitingtimeinBlock)
blockdataFram[med_waitingBinx]=pd.Series(med_waitingtimeinBlock)
blockdataFram.to_csv('../TimeBlock'+str(start_selection)+'.csv',index=False,header=False)
