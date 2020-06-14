from torch.utils.data import DataLoader,Dataset
import sys
sys.path.append('../NCKHECG')
from DataProcessing.WaveFormReader import get_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch

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
    
    def __getitem__(self, item):
        return self.signals[item],self.convert_ann_to_num_and_onehot(self.labels[item])
    
    def convert_ann_to_num_and_onehot(self,anno):
        if anno=='N':
            return 0,[1,0,0,0]
        elif anno=='V':
            return 1,[0,1,0,0]
        if anno=='F':
            return 2,[0,0,1,0]
        if anno=='S':
            return 3,[0,0,0,1]

    def __len__(self):
        return len(self.signals)

    def convert_dict_data_to_list(self,data_patiences):
        signals=[]
        annos=[]
        for patience in data_patiences:
            signals.append(data_patiences[patience][0])
            annos.append(data_patiences[patience][1])
        signals = [[signal] for patience_signal in signals for signal in patience_signal]
        annos   = [anno for patience_anno in annos for anno in patience_anno]
        return signals,annos

if __name__=='__main__':
    test_dataset=Singledata_train_test('intra','test')
    dataloader= DataLoader(test_dataset, batch_size=16, shuffle=True)
    for i, sample_batched in enumerate(dataloader):
        signals,(labels,one_hot)=sample_batched
        signals=torch.stack(signals).permute(1,0,2)
        one_hot=torch.stack(one_hot).transpose(1,0)
        print(labels)
        print(one_hot)
        print(signals.shape)
        for idx in range(len(signals)):
            plt.plot(signals[idx][0])
            plt.text(100,1,labels[idx])
            plt.show()
        print('--------------------')
        
            

