import torch
import numpy as np
import sys
sys.path.append('../NCKHECG')
from ModelArch.BackBone import SingleBackBoneNet
from DataProcessing.LoadTrainData import Singledata_train_test 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
model=SingleBackBoneNet(num_class=2)
model.load_state_dict(torch.load('Save_model/SingleWoProto/save_intra_2_NV_only.pth'))
model.to('cpu')
model.eval()
val_data = Singledata_train_test(train_or_test='test',test_mode='intra')
dataloader_val = DataLoader(val_data, batch_size=1, shuffle=True)
num_true=0
total=0
with torch.no_grad():    
    for i,sample_batched in enumerate(dataloader_val):
        inputs,(labels,one_hot_labels)=sample_batched
        inputs=torch.stack(inputs).permute(1,0,2)
        predict_logits,attention_score=model(inputs.float())
        predict_labels=torch.argmax(predict_logits,dim=1).cpu().data.numpy()
        labels=labels.cpu().data.numpy()

        print('pred:')
        print(F.softmax(predict_logits).data.numpy())
        print(predict_labels)
        print(attention_score.data.numpy()[:3,0,:])
        print('real:')
        print(labels)
        print('-------------------')
        if predict_labels!=labels:
            break
            # plt.plot(inputs[0][0])
            # plt.text(100,1,str(predict_labels))
            # plt.text(200,1,str(labels))
            # plt.show()
        else:
            num_true+=1
        total+=1
        print('acc so far :'+ str(num_true/total))


