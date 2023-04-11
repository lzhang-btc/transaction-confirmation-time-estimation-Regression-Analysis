
from enum import Enum

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import time
 


import sys
sys.path.append("..")
import G_Variables



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
timeToConfirmintx=22# confirm





######加载全局变量
###tx feature
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





 



SHORT_BLOCK_PERIODS = 12
SHORT_SCALE = 1
MED_BLOCK_PERIODS = 24
MED_SCALE = 2
LONG_BLOCK_PERIODS = 1000
LONG_SCALE = 1
OLDEST_ESTIMATE_HISTORY = 6 * 1008


SHORT_DECAY = .962
MED_DECAY = .9952
LONG_DECAY = .99931
LONG_DECAY=SHORT_DECAY

# Require greater than 60% of X feerate transactions to be confirmed within Y/2 blocks*/
HALF_SUCCESS_PCT = .6
#Require greater than 85% of X feerate transactions to be confirmed within Y blocks*/
SUCCESS_PCT = .85
# Require greater than 95% of X feerate transactions to be confirmed within 2 * Y blocks*/
DOUBLE_SUCCESS_PCT = .95

# Require an avg of 0.1 tx in the combined feerate bucket per block to have stat significance */
SUFFICIENT_FEETXS = 0.1
# Require an avg of 0.5 tx when using short decay since there are fewer blocks considered*/
SUFFICIENT_TXS_SHORT = 0.5




MIN_BUCKET_TIME=100
MAX_BUCKET_TIME=600*50
FEE_SPACING = 1.05

training_blocks = G_Variables.training_blocks

TestGroup='16'


Start_Block=getattr(G_Variables,'START_EstimateBlock_S'+TestGroup)
total_EstimateBlock=getattr(G_Variables,'total_EstimateBlock_S'+TestGroup)
txinblockfile= '../'+getattr(G_Variables,'txfile_S'+TestGroup)
AlgName='DecayMean'
resultfileName='S'+TestGroup+AlgName


# # ##########ForTest
# #
# training_blocks = 3
# total_EstimateBlock=2






class StatusVector:
    def __init__(self,block_periods,decay,scale,sufficentTx):
        self.block_periods=block_periods
        self.decay=decay
        self.scale=scale
        self.maxConfirms=block_periods*scale
        self.sufficentTx=sufficentTx


class Status(Enum):
    short_status=StatusVector(SHORT_BLOCK_PERIODS,SHORT_DECAY,SHORT_SCALE,SUFFICIENT_TXS_SHORT)
    med_status=StatusVector(MED_BLOCK_PERIODS,MED_DECAY,MED_SCALE,SUFFICIENT_FEETXS)
    long_status=StatusVector(LONG_BLOCK_PERIODS,LONG_DECAY,LONG_SCALE,SUFFICIENT_FEETXS)


class MaxConfirmation(Enum):
  LongMaxConfirms=LONG_BLOCK_PERIODS*LONG_SCALE
  ShortMaxConfirms=SHORT_BLOCK_PERIODS*SHORT_SCALE
  MedMaxConfirms=MED_BLOCK_PERIODS*MED_SCALE


class FeeReason(Enum):
    NONE='NONE'
    HALF_ESTIMATE='HALF_ESTIMATE'
    FULL_ESTIMATE='FULL_ESTIMATE'
    DOUBLE_ESTIMATE='DOUBLE_ESTIMATE'
    CONSERVATIVE='CONSERVATIVE'
    MEMPOOL_MIN='MEMPOOL_MIN'
    PAYTXFEE='PAYTXFEE'
    FALLBACK='FALLBACK'
    REQUIRED='REQUIRED'
    MAXTXFEE='MAXTXFEE'

    
class FeeEstimateMode(Enum):
    UNSET='UNSET'      #! Use default settings based on other criteria
    ECONOMICAL='ECONOMICAL'   #! Force estimateSmartFee to use non-conservative estimates
    CONSERVATIVE='CONSERVATIVE' #! Force estimateSmartFee to use conservative estimates
 
 
class EstimatorBucket:
  def __init__(self):
    self.start = -1
    self.end = -1
    self.withinTarget = 0
    self.totalConfirmed = 0
    self.inMempool = 0
    self.leftMempool = 0
 
 
class EstimationResult:
  def __init__(self):
    self.Pass=EstimatorBucket()
    self.Fail=EstimatorBucket()
    self.decay=0
    self.scale=0


class FeeCalculation:
  def __init__(self):
   self.est=EstimationResult()
   self.reason = FeeReason.NONE.value
   self.desiredTarget = 0
   self.returnedTarget = 0

  

def calBucketIndex(confTime, bucketBoundary,fee_spacing):
    minTime=bucketBoundary[0]
    if confTime<MIN_BUCKET_TIME:
        index=0
    elif confTime>bucketBoundary[len(bucketBoundary)-1]:
        index=len(bucketBoundary)-1
    else:
        cTime=confTime/minTime
        index=math.log(cTime,fee_spacing)
    if index-int(index)>0:
        index=int(index+1)
    else:
        index=int(index)
    return index
  
 
def EstimateMedianVal(confTarget,label_status):

  block_periods=label_status.value.block_periods
  # sufficentTx=label_status.value.sufficentTx
  # decay=label_status.value.decay
  scale=label_status.value.scale
  
  if block_periods==SHORT_BLOCK_PERIODS:
    choice='short'
  elif block_periods==MED_BLOCK_PERIODS:
    choice='med'
  else:
    choice='long'
    
  confCt= confCts[choice]
  confVal=confVals[choice]

  periodTarget = int((confTarget + scale - 1) / scale)
  if sum(confCt[periodTarget-1])==0:
      meanTime=-1
  else:
      meanTime=sum(confVal[periodTarget-1]) / sum(confCt[periodTarget-1])

      

  return meanTime
    

def constructConfAVG(confirmed_Tx):
    rows=confirmed_Tx.shape[0]
    cols=confirmed_Tx.shape[1]
    confAVG = np.zeros((rows,cols))
    for i in range(rows):
        confAVG[i,:]=confirmed_Tx[i:rows,:].sum(axis=0)
    return confAVG
  
   

  
# Meaning CurblockConf: within Y Blocks
def curBlockInfo(txcollection,bucketBoundary,blockheight,label_status):
  block_periods=label_status.value.block_periods
  scale=label_status.value.scale
  curBlockConf = np.zeros((block_periods, bucketBoundary.__len__()))
  curBlockConfVal = np.zeros((block_periods, bucketBoundary.__len__()))

  tempcollection = txcollection[np.where(txcollection[:, blockHeightintx]==blockheight)]
  if tempcollection.shape[0]>0:
      for tx in tempcollection:
          conftime=tx[waitingtimeinx]
          bucketIndex=calBucketIndex(conftime,bucketBoundary,FEE_SPACING)
          txfeerate=int(tx[feerateintx])
          periodsToConfirm=int((txfeerate+scale-1)/scale)
          if periodsToConfirm>block_periods:
              periodsToConfirm=block_periods
          curBlockConf[periodsToConfirm-1][bucketIndex]=curBlockConf[periodsToConfirm-1][bucketIndex]+1
          curBlockConfVal[periodsToConfirm - 1][bucketIndex] = curBlockConfVal[periodsToConfirm - 1][bucketIndex] + conftime

  return curBlockConf,curBlockConfVal
  









def txDatasetProcessing(txcollection,bucketBoundary,currentblockHeight,label_status):
    # collect txs in [LastUpdataBlockHeights[choice]+1,currentblockHeight-1]
  block_periods=label_status.value.block_periods
  decay=label_status.value.decay
  if block_periods==SHORT_BLOCK_PERIODS:
    choice='short'
  elif block_periods==MED_BLOCK_PERIODS:
    choice='med'
  else:
    choice='long'
  startUpdate=LastUpdataBlockHeights[choice]+1
  endUpdate=currentblockHeight
  for i in range(startUpdate,endUpdate):
      curBlockConf, curBlockConfVal=curBlockInfo(txcollection,bucketBoundary,i,label_status)
      confCts[choice]=confCts[choice]*decay+curBlockConf
      confVals[choice] = confVals[choice] * decay + curBlockConfVal
      LastUpdataBlockHeights[choice]=i
      print(LastUpdataBlockHeights)


 
 

 
 
 
 
 
 
 
  




  






def estimateCombinedFee(txcollection,bucketBoundary,confTarget,currentblockHeight):
   estimate = -1
   if confTarget >= 1 and confTarget <= MaxConfirmation.LongMaxConfirms.value:
       #################horizon=long
          # # Find estimate from shortest time horizon possible
          # if confTarget <= MaxConfirmation.ShortMaxConfirms.value: # short horizon
          #   label_status=Status.short_status
          # elif confTarget <= MaxConfirmation.MedMaxConfirms.value: # medium horizon
          #   label_status=Status.med_status
          # else: # long horizon
          #   label_status=Status.long_status
          label_status = Status.long_status

          txDatasetProcessing(txcollection, bucketBoundary, currentblockHeight, label_status)
          estimate= EstimateMedianVal(confTarget, label_status)




   return estimate
  

def MeanMethod_estimateSmartFee(txcollection,bucketBoundary,confTarget
                                ,currentblockHeight):

  median=-1
  if confTarget<0:
    print('error targetBLocks')
    return -1

  maxUsableEstimate =MaxConfirmation.LongMaxConfirms.value
  if confTarget>maxUsableEstimate:
    confTarget=maxUsableEstimate
  actualEst=estimateCombinedFee(txcollection,bucketBoundary,confTarget,currentblockHeight)
  print(' actualEst:'+str(actualEst))
  #MeanMethod.write(' actualEst:'+str(actualEst)+'\n')
  if actualEst>median:
    median=actualEst
  if median<0:
    print('error')
  return median


##################################Main Function
MeanMethod = open('../NeuarlResult/'+AlgName+'/'+resultfileName+'.txt', 'a')

data = pd.read_csv(txinblockfile, sep=",",header=None)
data[feerateintx] = 4 * data[feeintx] // data[weightintx]
#data = data[data.feerate > 0]
data = data[data[waitingblockintx] > 0]
# data.to_csv("txinblock2_bitcore.csv",index=False,header=False)
txdatafram=data
txdata = np.array(data)
# txdata = np.loadtxt('txinblock2_bitcore.csv', delimiter=",")

tempdata = data[(data[enterBlockintx] >=Start_Block)&(data[enterBlockintx]<Start_Block+total_EstimateBlock)]
Targets=np.unique(np.array(tempdata[feerateintx]))



minRelayTime = MIN_BUCKET_TIME

maxTime = txdata[:, waitingtimeinx].max(axis=0)

if maxTime < MAX_BUCKET_TIME:
    maxTime = MAX_BUCKET_TIME

dbucketBoundary = minRelayTime
bucketBoundary = [minRelayTime]
while dbucketBoundary <= maxTime:
    dbucketBoundary = dbucketBoundary * FEE_SPACING
    bucketBoundary.append(dbucketBoundary)




start=time.time()
localtime = time.asctime(time.localtime(start))
MeanMethod.write('Experiment Time: '+localtime+'\n')

MeanMethod.write('minRelayFee: '+str(MIN_BUCKET_TIME)+'\n')

avg = np.zeros(bucketBoundary.__len__())
txCtAvg = np.zeros(bucketBoundary.__len__())
confCt_short = np.zeros((SHORT_BLOCK_PERIODS, bucketBoundary.__len__()))
confCt_med = np.zeros((MED_BLOCK_PERIODS, bucketBoundary.__len__()))
confCt_long = np.zeros((LONG_BLOCK_PERIODS, bucketBoundary.__len__()))

confVal_short = np.zeros((SHORT_BLOCK_PERIODS, bucketBoundary.__len__()))
confVal_med = np.zeros((MED_BLOCK_PERIODS, bucketBoundary.__len__()))
confVal_long = np.zeros((LONG_BLOCK_PERIODS, bucketBoundary.__len__()))

confCts = {'short': confCt_short, 'med': confCt_med, 'long': confCt_long}
confVals = {'short': confVal_short, 'med': confVal_med, 'long': confVal_long}

intialUpdataBlockHeight = Start_Block-training_blocks-1
LastUpdataBlockHeights = {'short': intialUpdataBlockHeight, 'med': intialUpdataBlockHeight,
                          'long': intialUpdataBlockHeight}
for i in range(total_EstimateBlock):

    currentblockHeight = Start_Block + i


    print(LastUpdataBlockHeights)

    print('\n CurrentBlockHeight: '+str(currentblockHeight))
    MeanMethod.write('CurrentBlockHeight: '+str(currentblockHeight))

    Isconservative = False
    for confTarget in Targets:
        MeanMethod.write('\nConfTarget:'+str(confTarget) + '\n')
        start = time.time()
        median=MeanMethod_estimateSmartFee(txdata,bucketBoundary,confTarget,currentblockHeight)
        end = time.time()

        MeanMethod.write('Result:'+str(median))
    end = time.time()
    MeanMethod.write('\n\n totalTIme: '+str(end-start)+'\n')
MeanMethod.close()



