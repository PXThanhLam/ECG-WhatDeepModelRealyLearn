from torch import nn
import copy
import torch
from torchsummary import summary
import numpy as np
import torch.nn.functional as F
from sparsemax import Sparsemax
import sys
sys.path.append('../NCKHECG')
from ModelArch.BackBone import SingleBackBoneNet,MultiBackBoneNet
sparsemax = Sparsemax(dim=-1)

class PrototypeNet(nn.Module):
    def __init__(self,SingleBackBone,attention_dim,feature_dim,hidden_dim,class_hidden_dim,input_channel=1,num_class=4):
        super(PrototypeNet,self).__init__()
        self.attention_dim=attention_dim
        self.BackBone=SingleBackBone
        self.proj_feature=nn.Linear(in_features=feature_dim,out_features=hidden_dim)
        self.layernorm=nn.LayerNorm(hidden_dim)
        self.encoded_key=nn.Linear(in_features=hidden_dim,out_features=attention_dim)
        self.encoded_querry=nn.Linear(in_features=hidden_dim,out_features=attention_dim)
        self.encoded_value=nn.Linear(in_features=hidden_dim,out_features=attention_dim)
        self.classifier_hidden=nn.Linear(in_features=attention_dim,out_features=class_hidden_dim)
        self.classifier=nn.Linear(in_features=class_hidden_dim,out_features=num_class)
    def relational_attention(self,encoded_queries,candidate_keys,
                            candidate_values,normalization="softmax"):
        activations=torch.matmul(candidate_keys,torch.transpose(encoded_queries,0,1))
        activations/=np.sqrt(self.attention_dim)
        activations = torch.transpose(activations, 0, 1)
        if normalization == "softmax":
            weight_coefs =F.softmax(activations)
        elif normalization == "sparsemax":
            weight_coefs = sparsemax(activations)
        else:
            weight_coefs = activations
        weighted_encoded = torch.matmul(weight_coefs, candidate_values)

        return weighted_encoded, weight_coefs
    def encoder(self,x):
        feature_embeding,_=self.BackBone(x,return_feature=True)
        print(feature_embeding.shape)
        feature_embeding=self.proj_feature(feature_embeding)
        feature_embeding=self.layernorm(feature_embeding)
        key=self.encoded_key(feature_embeding)
        querry=self.encoded_querry(feature_embeding)
        value=self.encoded_value(feature_embeding)
        return key,querry,value
    def forward(self, signal_batch,signal_cand,norm_method="sparsemax",alpha_intermediate=0.5):
        _, encoded_batch_queries, encoded_batch_values=self.encoder(signal_batch)
        encoded_cand_keys, _, encoded_cand_values=self.encoder(signal_cand)
        weighted_encoded_batch, weight_coefs_batch=self.relational_attention(encoded_batch_queries,
                    encoded_cand_keys,
                    encoded_cand_values,norm_method)
        joint_encoded_batch = (1 - alpha_intermediate) * encoded_batch_values \
                               + alpha_intermediate * weighted_encoded_batch
        logits_joint_batch = self.classifier(self.classifier_hidden(joint_encoded_batch))
        logits_orig_batch  = self.classifier(self.classifier_hidden(encoded_batch_values))
        logits_weighted_batch=self.classifier(self.classifier_hidden(weighted_encoded_batch))
        return logits_orig_batch,logits_weighted_batch,logits_joint_batch,weight_coefs_batch


            



if __name__=='__main__':
    batch=torch.randn(size=(16,1,280))
    cand=torch.randn(size=(128,1,280))
    model=PrototypeNet(SingleBackBoneNet(),attention_dim=27,feature_dim=128,hidden_dim=128,class_hidden_dim=128)
    summary(model,input_data=(batch,cand))
    logits_orig_batch,logits_weighted_batch,logits_joint_batch,weight_coefs_batch=model(batch,cand)
    
    




