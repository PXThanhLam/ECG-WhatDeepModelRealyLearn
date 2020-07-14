from torch.utils.data import DataLoader,Dataset
import sys
sys.path.append('../NCKHECG')
from DataProcessing.WaveFormReader import get_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from functools import reduce
import numpy as np


class Singledata_train_test(Dataset):
    def __init__(self,test_mode='intra',train_or_test='train',dataset='mit'):
        super(Singledata_train_test,self).__init__()
        if test_mode=='intra':
            if dataset!='mit':
                raise('Intra dataset use for mit-bih only')
            if train_or_test=='train':
                data_patience_train=get_data('Data/mit-bih',mode='train')
                self.signals,self.labels=self.convert_dict_data_to_list(data_patience_train)
            else:
                data_patience_test=get_data('Data/mit-bih',mode='test')
                self.signals,self.labels=self.convert_dict_data_to_list(data_patience_test)

        else:
            data_patience_train=get_data('Data/mit-bih',mode='train',test_mode='cross') if dataset=='mit' else\
                                get_data('Data/StPeterburg',mode='train',test_mode='cross')
            self.signals,self.labels=self.convert_dict_data_to_list(data_patience_train)

        print(len(self.signals))
        print(len(self.labels))
    def __getitem__(self, item):
        return self.signals[item],self.convert_ann_to_num_and_onehot(self.labels[item])
    
    def convert_ann_to_num_and_onehot(self,anno):
        if anno=='N':
            return 0,[1,0,0,0]
        elif anno=='V':
            return 1,[0,1,0,0]
        elif anno=='F':
            return 2,[0,0,1,0]
        elif anno=='S':
            return 3,[0,0,0,1]

    def __len__(self):
        return len(self.signals)

    # def convert_dict_data_to_list(self,data_patiences):
    #     signals={'N':[],'V':[],'F':[],'S':[]}
        
    #     for patience in data_patiences:
    #         for idx,anno in enumerate(data_patiences[patience][1]):
    #             signals[anno].append([data_patiences[patience][0][idx]])
    #     return signals['N']+signals['V']*12+signals['F']*110+signals['S']*49,\
    #             ['N']*len(signals['N'])+['V']*len(signals['V'])*12+['F']*len(signals['F'])*110+['S']*len(signals['S'])*49
    def convert_dict_data_to_list(self,data_patiences):
        signals=[]
        annos=[]
        for patience in data_patiences:
            signals.append(data_patiences[patience][0])
            annos.append(data_patiences[patience][1])
        signals = [[signal] for patience_signal in signals for signal in patience_signal]
        annos   = [anno for patience_anno in annos for anno in patience_anno]
        return signals,annos



class Multidata_train_test(Dataset):
    def __init__(self,test_mode='intra',train_or_test='train',dataset='mit',seq_len=24):
        super(Multidata_train_test,self).__init__()
        if test_mode=='intra':
            if dataset!='mit':
                raise('Intra dataset use for mit-bih only')
            if train_or_test=='train':
                data_patience_train=get_data('Data/mit-bih',mode='train')
                self.signals,self.labels=self.convert_dict_data_to_list(data_patience_train)
            else:
                data_patience_test=get_data('Data/mit-bih',mode='test')
                self.signals,self.labels=self.convert_dict_data_to_list(data_patience_test)

        else:
            data_patience_train=get_data('Data/mit-bih',mode='train',test_mode='cross') if dataset=='mit' else\
                                get_data('Data/StPeterburg',mode='train',test_mode='cross')
            self.signals,self.labels=self.convert_dict_data_to_list(data_patience_train)
        self.seq_len=seq_len
        self.len_signal_per_patience=[len(patience_signal) for patience_signal in self.signals]
    def convert_dict_data_to_list(self,data_patiences):
        signals=[]
        annos=[]
        for patience in data_patiences:
            signals.append(data_patiences[patience][0])
            annos.append(data_patiences[patience][1])
        return signals,annos
    def convert_ann_to_num(self,anno):
        if anno=='N':
            return 0
        elif anno=='V':
            return 1
        elif anno=='F':
            return 2
        elif anno=='S':
            return 3
    def convert_ann_to_one_hot(self,anno):
        if anno=='N':
            return torch.Tensor([1,0,0,0])
        elif anno=='V':
            return torch.Tensor([0,1,0,0])
        elif anno=='F':
            return torch.Tensor([0,0,1,0])
        elif anno=='S':
            return torch.Tensor([0,0,0,1])
    def __getitem__(self, item):
        patience_index=None
        current_item=item
        for idx,len_patience in enumerate(self.len_signal_per_patience):
            if current_item<=len_patience-1:
                patience_index=idx
                break
            else:
                current_item-=len_patience
        if current_item>=self.seq_len:
            left_index=np.random.choice(range(current_item-self.seq_len,current_item-self.seq_len+min(self.seq_len,len(self.signals[patience_index])-current_item)),1)[0]
        else:
            left_index=np.random.choice(range(0,current_item),1)[0] if current_item !=0 else 0
        # print('item '+str(item))
        # print('pt ind '+str(patience_index))
        # print('left ind ' +str(left_index))
        return self.signals[patience_index][left_index:left_index+self.seq_len],\
            [self.convert_ann_to_num(anno) for anno in self.labels[patience_index][left_index:left_index+self.seq_len]],\
            [self.convert_ann_to_one_hot(anno) for anno in self.labels[patience_index][left_index:left_index+self.seq_len]]

    def __len__(self):
        return reduce(lambda a,b : a+b,self.len_signal_per_patience)
        

if __name__=='__main__':
    # test_dataset=Singledata_train_test('intra','test')
    # dataloader= DataLoader(test_dataset, batch_size=16, shuffle=True)
    # for i, sample_batched in enumerate(dataloader):
    #     signals,(labels,one_hot)=sample_batched
    #     signals=torch.stack(signals).permute(1,0,2)
    #     one_hot=torch.stack(one_hot).transpose(1,0)
    #     print(labels)
    #     print(one_hot)
    #     print(signals.shape)
    #     for idx in range(len(signals)):
    #         plt.plot(signals[idx][0])
    #         plt.text(100,0,labels[idx])
    #         plt.show()
    #     print('--------------------')
    test_dataset=Multidata_train_test('intra','train',seq_len=10)
    dataloader= DataLoader(test_dataset, batch_size=16, shuffle=True)
    for i, sample_batched in enumerate(dataloader):
        inputs,labels,one_hot_labels=sample_batched
        inputs=torch.stack(inputs).permute(1,0,2).unsqueeze(2)
        one_hot_labels= torch.stack(one_hot_labels).permute(1,0,2)
        one_hot_labels=torch.reshape(one_hot_labels,(one_hot_labels.shape[0]*one_hot_labels.shape[1],one_hot_labels.shape[2]))
        labels = torch.stack(labels).permute(1,0)
        labels=torch.reshape(labels,(labels.shape[0]*labels.shape[1],))   

        inputs_to_plot=inputs.data.numpy()[0][:,0,:].reshape(10*280)
        print(inputs_to_plot.shape)
        labels=labels.data.numpy()[:10]
        one_hot_labels=one_hot_labels.data.numpy()[:10,:]

        plt.plot(inputs_to_plot)
        print(labels)
        print(one_hot_labels)
        plt.show()
        # signals=torch.stack(signals).permute(1,0,2)
        # one_hot=torch.stack(one_hot).transpose(1,0)
        # print(labels)
        # print(one_hot)
        # print(signals.shape)
        # for idx in range(len(signals)):
        #     plt.plot(signals[idx][0])
        #     plt.text(100,0,labels[idx])
        #     plt.show()
        # print('--------------------')
        
            

