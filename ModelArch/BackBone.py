from torch import nn
import copy
import torch
from torchsummary import summary
class ConvLayer1D(nn.Module):
    def __init__(self,in_chanels,out_chanels,kernel_size,stride=1,padding=0,isbatchnorm=True):
        super(ConvLayer1D, self).__init__()

        self.conv1d=nn.Conv1d(in_chanels,out_chanels,kernel_size,
                            stride=stride,padding=padding)
        if isbatchnorm:
            self.batchnorm1d=nn.BatchNorm1d(out_chanels)
        self.isbatchnorm=isbatchnorm
    
    def forward(self, x):
        x=self.conv1d(x)
        if self.isbatchnorm:
            x=self.batchnorm1d(x)
        x=nn.ReLU()(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,in_chanel,out_chanels,kernel_sizes,strides,paddings,isbatchnorm=True):
        assert len(out_chanels)==len(kernel_sizes)==len(paddings)==len(paddings)
        super(ResBlock, self).__init__()
        num_stack=len(out_chanels)
        self.conv_lists=[]
        for idx in range(num_stack):
            if idx==0:
                self.conv_lists.append(ConvLayer1D(in_chanel,out_chanels[0],kernel_sizes[0],strides[0],paddings[0],isbatchnorm=False))
            else:
                if idx!=num_stack-1:
                    self.conv_lists.append(ConvLayer1D(out_chanels[idx-1],out_chanels[idx],kernel_sizes[idx],strides[idx],paddings[idx],isbatchnorm=False))
                else:
                    self.conv_lists.append(ConvLayer1D(out_chanels[idx-1],out_chanels[idx],kernel_sizes[idx],strides[idx],paddings[idx],isbatchnorm=True))
    
    def forward(self, x,use_batchnorm=True,resiudal=True):
        x_copy=copy.copy(x)
        for layer in self.conv_lists:
            x_copy=layer(x_copy)
        return x+x_copy if resiudal else x_copy   


class AttentionWithContext(nn.Module):
    def __init__(self,input_channel,hidden_channel):
        super(AttentionWithContext,self).__init__()
        self.W = nn.Linear(input_channel,hidden_channel)
        self.U = nn.Linear(hidden_channel,1)
    
    def forward(self, x):
        hidden_x=nn.Tanh()(self.W(x.permute(0,2,1)))
        attention_score=nn.Softmax()(self.U(hidden_x)).permute(0,2,1)
        return torch.sum(attention_score*x,dim=2),attention_score


class SingleBackBoneNet(nn.Module):
    def __init__(self,input_channel=1,num_class=4):
        super(SingleBackBoneNet,self).__init__()

        self.ResBlock1=ResBlock(in_chanel=input_channel,out_chanels=[4,4,4],kernel_sizes=[3,5,7],strides=[1,1,1],paddings=[1,2,3])
        self.lstm1=nn.LSTM(input_size=4,hidden_size=4,num_layers=1,bidirectional=True)
        self.maxpool1=nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
        self.batch_norm1= nn.BatchNorm1d(8)

        self.ResBlock2=ResBlock(in_chanel=8,out_chanels=[8,8,8],kernel_sizes=[3,5,7],strides=[1,1,1],paddings=[1,2,3])
        self.lstm2=nn.LSTM(input_size=8,hidden_size=8,num_layers=1,bidirectional=True)
        self.maxpool2=nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
        self.batch_norm2= nn.BatchNorm1d(16)

        self.ResBlock3=ResBlock(in_chanel=16,out_chanels=[16,16,16],kernel_sizes=[3,5,7],strides=[1,1,1],paddings=[1,2,3])
        self.lstm3=nn.LSTM(input_size=16,hidden_size=16,num_layers=1,bidirectional=True)
        self.maxpool3=nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
        self.batch_norm3= nn.BatchNorm1d(32)

        self.ResBlock4=ResBlock(in_chanel=32,out_chanels=[32,32,32],kernel_sizes=[3,5,7],strides=[1,1,1],paddings=[1,2,3])
        self.lstm4=nn.LSTM(input_size=32,hidden_size=32,num_layers=1,bidirectional=True)
        self.maxpool4=nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
        self.batch_norm4= nn.BatchNorm1d(64)

        self.ResBlock5=ResBlock(in_chanel=64,out_chanels=[64,64,64],kernel_sizes=[3,5,7],strides=[1,1,1],paddings=[1,2,3])
        self.lstm5=nn.LSTM(input_size=64,hidden_size=64,num_layers=1,bidirectional=True)
        self.maxpool5=nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
        self.batch_norm5= nn.BatchNorm1d(128)

        self.AttentionBlock=AttentionWithContext(input_channel=128,hidden_channel=128)
        self.classifier=nn.Linear(128,num_class)
    
    def forward(self,x):
        x=self.ResBlock1(x,resiudal=False)
        x=self.lstm1(x.permute(2,0,1))[0].permute(1,2,0)
        x=self.maxpool1(x)
        x=self.batch_norm1(x)

        x=self.ResBlock2(x)
        x=self.lstm2(x.permute(2,0,1))[0].permute(1,2,0)
        x=self.maxpool2(x)
        x=self.batch_norm2(x)

        x=self.ResBlock3(x)
        x=self.lstm3(x.permute(2,0,1))[0].permute(1,2,0)
        x=self.maxpool3(x)
        x=self.batch_norm3(x)

        x=self.ResBlock4(x)
        x=self.lstm4(x.permute(2,0,1))[0].permute(1,2,0)
        x=self.maxpool4(x)
        x=self.batch_norm4(x)

        x=self.ResBlock5(x)
        x=self.lstm5(x.permute(2,0,1))[0].permute(1,2,0)
        x=self.maxpool5(x)
        x=self.batch_norm5(x)
        
        x,attention_score=self.AttentionBlock(x)
        output=self.classifier(x)

        return output,attention_score



if __name__=='__main__':
    x=torch.zeros(size=(4,1,280))
    model=SingleBackBoneNet()
    summary(model,input_data=x)