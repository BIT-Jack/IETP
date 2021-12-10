from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
import time
import numpy as np

from collections import Counter

import warnings
warnings.filterwarnings('ignore')

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = True
args['train_flag'] = False


# Evaluation metric:
metric = 'rmse'  #or nll



Num_of_nets = 20

NETS = []


##Initialize the networks and put them into NETS
for n in range(1, Num_of_nets+1):
    net = highwayNet(args)
    net.load_state_dict(torch.load('trained parameters/standard_trained/base{:.0f}.tar'.format(n)))
    if args['use_cuda']:
        net = net.cuda()
    
    NETS.append(net)

#----------------------------------------------------------------------------------------------------------------

if args['use_cuda']:
    for m in range(0,Num_of_nets):
        NETS[m] = NETS[m].cuda()

tsSet = ngsimDataset('test data/TestSet.mat')
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(25).cuda()
counts = torch.zeros(25).cuda()

Time_list = []
Time_list_net = []
##lists below are used to save the every predicted values from different bases
Fut_pred_list = []
Lat_pred_list = []
Lon_pred_list = []


for i, data in enumerate(tsDataloader):

    st_time = time.time()
##get the input data_batch
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

    # Initialize Variables
    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.cuda()
        lat_enc = lat_enc.cuda()
        lon_enc = lon_enc.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()
        
        
    Fut_pred_list.clear()
    Lat_pred_list.clear()
    Lon_pred_list.clear()    

    ##record the predicting time:
    time_begin = time.time()#--------------------------------------------------------timebegin
    ##get the predicted values and save them into
    with torch.no_grad():
        
        for p in range(0,Num_of_nets):
            fut_pred, lat_pred, lon_pred = NETS[p](hist, nbrs, mask, lat_enc, lon_enc)
            Fut_pred_list.append(fut_pred)
            Lat_pred_list.append(lat_pred)
            Lon_pred_list.append(lon_pred)

    
    ##fut_pred average--------------------------------------------------------------------------------------------------------
    fut_pred_avg = Fut_pred_list[0] #initial
    for mv in range(0,6):
        for kk in range(0, 5):
            for q in range(1,Num_of_nets):
                fut_pred_avg[mv][:,:,kk] = fut_pred_avg[mv][:,:,kk] + Fut_pred_list[q][mv][:,:,kk]
                
            fut_pred_avg[mv][:,:,kk] = fut_pred_avg[mv][:,:,kk]/Num_of_nets    

    mv = 0
    kk = 0

   # print("finish the average of fut_pred...")

    ##lat_pred voting----------------------------------------------------------------------------------------------------------
    List1 =[]#save probability form (tensors), as value
    List2 =[]#save tuple class form, as key
    
   #tensor to list
    Lat_pred__listtype_LIST = []
    for n in range(0,Num_of_nets):
        Lat_pred__listtype_LIST.append(Lat_pred_list[n].tolist())

    for v in range(0,Num_of_nets):
        List1.append(Lat_pred_list[v])
  

    #probability to detailed class
    Lat_pred_classtype_LIST = []
    for n in range(0,Num_of_nets):
        Lat_pred_classtype_LIST.append(np.argmax(Lat_pred__listtype_LIST[n], axis=1))
    
    
    #class list to tuple
    Lat_pred_tupletype_LIST = []
    for n in range(0,Num_of_nets):
        Lat_pred_tupletype_LIST.append(tuple(Lat_pred_classtype_LIST[n]))

    
    for v in range(0,Num_of_nets):
        List2.append(Lat_pred_tupletype_LIST[v])

    #get the voting result
    voting_result = Counter(List2)
    bagging_pred_class = voting_result.most_common(1)[0][0]
    
    #versus to get the bagging lat_pred
    dic = dict(zip(List2,List1))
    lat_pred_bagging = dic.get(bagging_pred_class) #this is what we want, i.e. the ensemble input
    #print("lat_pred_bagging:", lat_pred_bagging.size())

   #print("finish voting lat...")
    ##lon_pred voting-----------------------------------------------------------------------------------
    List3 =[]#save probability form
    List4 =[]#save tuple form
    
    #tensor to list
    Lon_pred_listtype_LIST = []
    for n in range(0,Num_of_nets):
        Lon_pred_listtype_LIST.append(Lon_pred_list[n].tolist())
    
    for v in range(0,Num_of_nets):
        List3.append(Lon_pred_list[v])      

    #probability to detailed class
    Lon_pred_classtype_LIST = []
    for n in range(0,Num_of_nets):
        Lon_pred_classtype_LIST.append(np.argmax(Lon_pred_listtype_LIST[n], axis=1))
    
    #class list to tuple
    Lon_pred_tupletype_LIST = []
    for n in range(0,Num_of_nets):
        Lon_pred_tupletype_LIST.append(tuple(Lon_pred_list[n]))
    
    
    for v in range(0,Num_of_nets):
        List4.append(Lon_pred_tupletype_LIST[v])


    #get the voting result
    voting_result_ = Counter(List4)
    bagging_pred_class_ = voting_result_.most_common(1)[0][0]

    #versus to get the bagging lat_pred
    dic_ = dict(zip(List4,List3))
    lon_pred_bagging = dic_.get(bagging_pred_class_) #this is what we want to be the ensemble input
 #   print("bagging lon pred:", lon_pred_bagging.size())    
  #  print("finish voting lon...")
    time_end = time.time()#-------------------------------------timeend
    duration = time_end - time_begin
    Time_list.append(duration)
    
    if metric == 'nll':
        # Forward pass
        if args['use_maneuvers']:
            l,c = maskedNLLTest(fut_pred_avg, lat_pred_bagging, lon_pred_bagging, fut, op_mask)
        else:
            l, c = maskedNLLTest(fut_pred_avg, 0, 0, fut, op_mask,use_maneuvers=False)
    else:
        # Forward pass
        if args['use_maneuvers']:
            fut_pred_max = torch.zeros_like(fut_pred_avg[0])
            for k in range(lat_pred_bagging.shape[0]):
                lat_man = torch.argmax(lat_pred_bagging[k, :]).detach()
                lon_man = torch.argmax(lon_pred_bagging[k, :]).detach()
                indx = lon_man*3 + lat_man
                fut_pred_max[:,k,:] = fut_pred_avg[indx][:,k,:]
            l, c = maskedMSETest(fut_pred_max, fut, op_mask)
        else:
#            fut_pred_avg = net(hist, nbrs, mask, lat_enc, lon_enc)
            l, c = maskedMSETest(fut_pred_avg, fut, op_mask)


    lossVals +=l.detach()
    counts += c.detach()
    if i%100 == 99:
        print("i==",i)
    #print("rmse:", torch.pow(lossVals / counts,0.5)*0.3048)
    #print("nll:",lossVals / counts )

if metric == 'nll':
    print(lossVals / counts)
else:
    RMSE = torch.pow(lossVals / counts,0.5)*0.3048
    print(RMSE)   # Calculate RMSE and convert from feet to meters
    avg_time_net = np.mean(Time_list_net)
    avg_time = np.mean(Time_list)   
    #result saving and displaying    
    result_rmse = open("result/RMSE/N{:.0f}_RMSE.txt".format(Num_of_nets),'w')
    result_rmse.writelines(str(RMSE.cpu().numpy()))
    result_rmse.close()    
    
    runtime = open("result/RMSE/N{:.0f}_runtime.txt".format(Num_of_nets),'w')
    runtime.writelines(str(avg_time))
    runtime.close()

    #runtime = open("single_runtime/bagging_runtime/sch2_bagging_runtime_netonly.txt",'w')
    #runtime.writelines(str(avg_time_net))
    #runtime.close()


