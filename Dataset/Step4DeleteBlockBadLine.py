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
for start_selection in [621500,622000,622500]:

    blockfile='../TimeBlock'+str(start_selection)+'.csv'
    blockdataFram = pd.read_csv(blockfile, sep=",",header=None)
    blockdataFram=blockdataFram[blockdataFram[blockHeightBinx]>start_selection-499]

    blockdataFram.to_csv('../TimeBlock'+str(start_selection)+'.csv',index=False,header=False)
