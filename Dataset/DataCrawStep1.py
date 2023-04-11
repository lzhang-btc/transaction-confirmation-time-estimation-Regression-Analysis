import urllib.request


start_selection=621500


mw_bock=open('./Block'+str(start_selection)+'.txt','a')

mw_tx=open('./txinBlock'+str(start_selection)+'.txt','a')

#mw_blockhash=open('./Blockhash'+str(start_selection)+'.txt','a')
#mw_txHash=open('./Txhash'+str(start_selection)+'.txt','a')

#mw_INOUT=open('./TxINOUT'+str(start_selection)+'.txt','a')




url = 'https://blockchain.info/rawblock/0000000000000000000af8bb10e85ea7623f17126ab81b3087f6bcd36b46b536'
blockNum=5
for i in range(blockNum):

    res = urllib.request.urlopen(url)
    html = res.read().decode('utf-8')
    result_html = html.replace('":null', '":"null"').replace('":true', '":"true"').replace('":false', '":"false"')

    dic = eval(result_html)
    print(str(dic['height']) + 'start')
    pre_block=dic['prev_block']

    txData=dic['tx']

    # tx fatures ['block_height', 'balance', 'relayed_by', 'fee', 'result', 'hash', 'inputs', 'size', 'double_spend', 'vin_sz', 'lock_time', 'out', 'weight', 'ver', 'vout_sz', 'block_index', 'tx_index', 'time'])

    tx_features=['tx_index','vin_sz','vout_sz','ver','size', 'weight', 'time','relayed_by', 'lock_time']
    ##Finaal  Final_tx_features=['tx_index','vin_sz','vout_sz','ver','size', 'weight', 'time','relayed_by', 'lock_time','fee',
    ###############block_height', 'block_index','confirmedtime','watingtime','feerate']
    #delete input and output script
    # if txData.__len__()==1:
    #     txelement = ''
    #     for j in range(tx_features.__len__()):
    #         txelement = txelement + '0' + " "
    #     tx_fee_in = 0
    #     tx_fee_out = 0
    #     tx_fee = tx_fee_in - tx_fee_out
    #     tx_feerate = tx_fee / txData[i][tx_features[4]]
    #     waitingtime = 0
    #     txelement = txelement + str(tx_fee) + " " + str(dic['height']) + " " + str(dic['block_index']) + " " + str(
    #         dic['time']) + " " + str(waitingtime) + " " + str(tx_feerate) + '\n'
    #     mw_tx.write(str(txelement))

    for i in range(1,txData.__len__()):
      # Delete the gerneration tx
      txelement=''
      for j in range(tx_features.__len__()):
        txelement=txelement+str(txData[i][tx_features[j]])+" "
      tx_fee_in=0
      tx_fee_out=0
      for f in range(txData[i]['vin_sz']):
        tx_fee_in=tx_fee_in+txData[i]['inputs'][f]['prev_out']['value']

      for o in range(txData[i]['vout_sz']):
        tx_fee_out=tx_fee_out+txData[i]['out'][o]['value']

      tx_fee=tx_fee_in-tx_fee_out
      tx_feerate=tx_fee/txData[i][tx_features[4]]
      waitingtime=dic['time']-txData[i][tx_features[6]]
      txelement=txelement+str(tx_fee)+" "+str(dic['height'])+" "+str(dic['block_index'])+" "+str(dic['time'])+" "+str(waitingtime)+" "+str(tx_feerate)+'\n'

      txHash = str(txData[i]['tx_index']) + " " + str(txData[i]['hash']) + " " + str(dic['block_index']) + '\n'
      #mw_txHash.write(txHash)
      #mw_INOUT.write(str(txData[i]['tx_index'])+" "+str(tx_fee_in)+" "+str(tx_fee_out)+'\n')
      mw_tx.write(str(txelement))
          # Generate txHash Information




    # ['nonce', 'n_tx', 'size', 'bits', 'weight', 'fee', 'ver', 'block_index', 'next_block', 'hash', 'prev_block', 'mrkl_root', 'main_chain', 'time', 'height'])

    keys=['block_index','height' ,'n_tx', 'size', 'bits', 'fee', 'ver','time']


    blockelement=[]
    for i in range(keys.__len__()):
      blockelement.append(str(dic[keys[i]]))
    strBlock=" ".join(blockelement)
    mw_bock.write(strBlock+'\n')

    # Generate BlockHash Information
    blockHash=str(dic['block_index'])+" "+str(dic['hash'])+'\n'
    #mw_blockhash.write(blockHash)
    print(str(dic['height'])+'end')


    url='https://blockchain.info/rawblock/'+pre_block

mw_bock.close()
mw_tx.close()
#mw_txHash.close()
#mw_blockhash.close()
#mw_INOUT.close()




