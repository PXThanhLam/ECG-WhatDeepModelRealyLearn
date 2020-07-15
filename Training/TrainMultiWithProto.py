import torch
import matplotlib.pyplot as plt
import argparse,os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('../NCKHECG')
from  Training.Loss import FocalLoss,MacroF1Loss,CrossEntropyWithWeights
from DataProcessing.LoadTrainData import Singledata_train_test,Multidata_train_test
from ModelArch.BackBone import SingleBackBoneNet,MultiBackBoneNet
from ModelArch.ProtoAttendModel import PrototypeNet
import torch.optim as optim
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from utils import AvgrageMeter

def train(args,test_mode='intra',):
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    log_file=open(args.log+'/'+test_mode+'_log.txt','w')
        
    print('Start Training !')
    log_file.write('Start Training !')
    log_file.flush()

    model=PrototypeNet(MultiBackBoneNet(SingleBackBoneNet()),attention_dim=128,feature_dim=256,hidden_dim=128,class_hidden_dim=128)
    model=model.cuda() if args.gpu else model.cpu()
    for name,param in model.named_parameters():
        print(name)    


    lr=args.lr
    optimizer=optim.Adam(model.parameters(),lr=lr)
    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=args.step_size,gamma=args.gamma)

    focal_lost=FocalLoss(alpha=torch.Tensor([0.5,5,50,20]))
    focal_lost=focal_lost.cuda() if args.gpu else focal_lost.cpu()
    macro_f1_lost=MacroF1Loss().cuda() if args.gpu else MacroF1Loss().cpu()
    ce_weights_lost=CrossEntropyWithWeights(weigths=[0.5,5,50,20])
    ce_weights_lost=ce_weights_lost.cuda() if args.gpu else ce_weights_lost.cpu()
    train_data=Multidata_train_test(train_or_test='train',test_mode=test_mode)
    dataloader_train_batch=DataLoader(train_data,batch_size=args.batchsize,shuffle=True)
    for epoch in range(args.epochs):
        loss_log=AvgrageMeter()
        f1_macro_log=AvgrageMeter()
        precision_macro_log=AvgrageMeter()
        recall_macro_log=AvgrageMeter()
        accuracy_log=AvgrageMeter()
        print("######################  TRAIN  #####################")
        model.train()
        for i, sample_batched in enumerate(dataloader_train_batch):
            signal_batch,label_batch,one_hot_label_batch=sample_batched
            dataloader_train_cand=DataLoader(train_data,batch_size=args.candsize,shuffle=True)
            dataloader_train_cand_iter=iter(dataloader_train_cand)
            signal_cand, label_cand,one_hot_label_cand=next(dataloader_train_cand_iter)

            signal_batch=torch.stack(signal_batch).permute(1,0,2).unsqueeze(2)
            one_hot_label_batch= torch.stack(one_hot_label_batch).permute(1,0,2)
            one_hot_label_batch=torch.reshape(one_hot_label_batch,(one_hot_label_batch.shape[0]*one_hot_label_batch.shape[1],one_hot_label_batch.shape[2]))
            label_batch = torch.stack(label_batch).permute(1,0)
            label_batch=torch.reshape(label_batch,(label_batch.shape[0]*label_batch.shape[1],))


            signal_cand=torch.stack(signal_cand).permute(1,0,2).unsqueeze(2)
            one_hot_label_cand= torch.stack(one_hot_label_cand).permute(1,0,2)
            one_hot_label_cand=torch.reshape(one_hot_label_cand,(one_hot_label_cand.shape[0]*one_hot_label_cand.shape[1],one_hot_label_cand.shape[2]))
            label_cand = torch.stack(label_cand).permute(1,0)
            label_cand=torch.reshape(label_cand,(label_cand.shape[0]*label_cand.shape[1],))
           
            if args.gpu:
                signal_batch,label_batch,one_hot_label_batch=signal_batch.cuda(),label_batch.cuda(),one_hot_label_batch.cuda()
                signal_cand,label_cand,one_hot_label_cand=signal_cand.cuda(),label_cand.cuda(),one_hot_label_cand.cuda()
            else:
                signal_batch,label_batch,one_hot_label_batch=signal_batch.cpu(),label_batch.cpu(),one_hot_label_batch.cpu()
                signal_cand,label_cand,one_hot_label_cand=signal_cand.cpu(),label_cand.cpu(),one_hot_label_cand.cpu()

            optimizer.zero_grad()
            
            logits_orig_batch,logits_weighted_batch,logits_joint_batch,weight_coefs_batch=model(signal_batch.float(),signal_cand.float())
            softmax_joint_op=0.9*macro_f1_lost(y_pred=logits_joint_batch, y_true=one_hot_label_batch.float())\
                 +0.1*ce_weights_lost(y_pred=logits_joint_batch, y_true=label_batch.long())
            softmax_orig_key_op=0.9*macro_f1_lost(y_pred=logits_orig_batch, y_true=one_hot_label_batch.float())\
                 +0.1*ce_weights_lost(y_pred=logits_orig_batch, y_true=label_batch.long())
            softmax_weighted_op=0.9*macro_f1_lost(y_pred=logits_weighted_batch, y_true=one_hot_label_batch.float())\
                 +0.1*ce_weights_lost(y_pred=logits_weighted_batch, y_true=label_batch.long())
            entropy_weights=torch.sum(-weight_coefs_batch*torch.log(args.epsilon_sparsity+weight_coefs_batch),dim=1)
            sparsity_loss=torch.mean(entropy_weights)
            loss=softmax_orig_key_op + softmax_weighted_op + \
                 softmax_joint_op + args.sparsity_weight * sparsity_loss

            loss.backward()
            optimizer.step()

            n = signal_batch.size(0)
            predict_labels=torch.argmax(logits_weighted_batch,dim=1).cpu().data.numpy()
            label_batch=label_batch.cpu().data.numpy()
            loss_log.update(loss.cpu().data,n)

            print(label_batch)
            print(predict_labels)
            f1_macro_log.update(f1_score(y_true=label_batch,y_pred=predict_labels,average='macro'),n)
            precision_macro_log.update(precision_score(y_true=label_batch,y_pred=predict_labels,average='macro'),n)
            recall_macro_log.update(recall_score(y_true=label_batch,y_pred=predict_labels,average='macro'),n)
            accuracy_log.update(accuracy_score(y_true=label_batch,y_pred=predict_labels),n)

            if i % args.echo_batches == args.echo_batches - 1:
                print('TRAIN epoch:%d, mini-batch:%3d, lr=%f, Loss= %.4f, f1= %.4f, precision= %.4f, recall= %.4f, acc= %.4f' % (
                epoch + 1, i + 1, lr, loss_log.avg, f1_macro_log.avg,precision_macro_log.avg,recall_macro_log.avg,accuracy_log.avg))
        
        print('epoch:%d, TRAIN : Loss= %.4f, f1= %.4f, precision= %.4f, recall= %.4f\n, acc= %.4f' % (
        epoch + 1, loss_log.avg, f1_macro_log.avg,precision_macro_log.avg,recall_macro_log.avg,accuracy_log.avg))
        log_file.write('epoch:%d, TRAIN : Loss= %.4f, f1= %.4f, precision= %.4f, recall= %.4f, acc= %.4f\n' % (
        epoch + 1, loss_log.avg, f1_macro_log.avg,precision_macro_log.avg,recall_macro_log.avg,accuracy_log.avg))
        log_file.flush()
        scheduler.step()
        torch.save(model.state_dict(),args.model_save_dir +'/'+'save_' +str(test_mode)+'_' +str(epoch)+'.pth')



        print("######################  VAL   #####################")
        model.eval()
        f1_macro_log_val=AvgrageMeter()
        precision_macro_log_val=AvgrageMeter()
        recall_macro_log_val=AvgrageMeter()
        accuracy_log_val=AvgrageMeter()
        with torch.no_grad():
            val_data = Singledata_train_test(train_or_test='test',test_mode=test_mode)
            dataloader_val_batch=DataLoader(val_data,batch_size=args.batchsize,shuffle=True)
            dataloader_val_cand=DataLoader(val_data,batch_size=len(val_data),shuffle=True)

            for i,sample_batched in enumerate(dataloader_val_batch):
                signal_batch,label_batch,one_hot_label_batch=sample_batched
                dataloader_val_cand=DataLoader(val_data,batch_size=len(val_data),shuffle=True)
                dataloader_val_cand_iter=iter(dataloader_val_cand)
                signal_cand, label_cand,one_hot_label_cand=next(dataloader_val_cand_iter)
                signal_batch=torch.stack(signal_batch).permute(1,0,2)
                one_hot_label_batch=torch.stack(one_hot_label_batch).transpose(1,0)
                signal_cand=torch.stack(signal_cand).permute(1,0,2)
                one_hot_label_cand=torch.stack(one_hot_label_cand).transpose(1,0)
            
                if args.gpu:
                    signal_batch,label_batch,one_hot_label_batch=signal_batch.cuda(),label_batch.cuda(),one_hot_label_batch.cuda()
                    signal_cand,label_cand,one_hot_label_cand=signal_cand.cuda(),label_cand.cuda(),one_hot_label_cand.cuda()
                else:
                    signal_batch,label_batch,one_hot_label_batch=signal_batch.cpu(),label_batch.cpu(),one_hot_label_batch.cpu()
                    signal_cand,label_cand,one_hot_label_cand=signal_cand.cpu(),label_cand.cpu(),one_hot_label_cand.cpu()
                    optimizer.zero_grad()

                n = signal_batch.size(0)
                predict_labels=torch.argmax(logits_weighted_batch,dim=1).cpu().data.numpy()
                label_batch=label_batch.cpu().data.numpy()

                f1_macro_log_val.update(f1_score(y_true=label_batch,y_pred=predict_labels,average='macro'),n)
                precision_macro_log_val.update(precision_score(y_true=label_batch,y_pred=predict_labels,average='macro'),n)
                recall_macro_log_val.update(recall_score(y_true=label_batch,y_pred=predict_labels,average='macro'),n)
                accuracy_log_val.update(accuracy_score(y_true=label_batch,y_pred=predict_labels),n)
                if i % args.echo_batches == args.echo_batches - 1:
                    print('VAL epoch:%d, mini-batch:%3d, lr=%f, f1= %.4f, precision= %.4f, recall= %.4f, acc= %.4f' % (
                    epoch + 1, i + 1, lr, f1_macro_log_val.avg,precision_macro_log_val.avg,recall_macro_log_val.avg,accuracy_log_val.avg))
            
            print('epoch:%d, VAL : f1= %.4f, precision= %.4f, recall= %.4f, acc= %.4f\n' % (
            epoch + 1, f1_macro_log_val.avg,precision_macro_log_val.avg,recall_macro_log_val.avg,accuracy_log_val.avg))
            log_file.write('epoch:%d, VAL : f1= %.4f, precision= %.4f, recall= %.4f, acc= %.4f\n' % (
            epoch + 1, f1_macro_log_val.avg,precision_macro_log_val.avg,recall_macro_log_val.avg,accuracy_log_val.avg))
            log_file.flush()
            
                

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--log', action='store_true', default='Training_log/SingleWithProto', help='training log path')
    parser.add_argument('--gpu', type=bool, default=False, help='Use gpu or cpu')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='initial batchsize')
    parser.add_argument('--candsize', type=int, default=256, help='candidate size')
    parser.add_argument('--step_size', type=int, default=5, help='how many epochs lr decays once')
    parser.add_argument('--val_per_epoch', type=int, default=1, help='how many train epoch per val')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--epochs', type=int, default=1000, help='total training epochs')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--echo_batches', type=int, default=1, help='how many batches display once')
    parser.add_argument('--model_save_dir', action='store_true', default='Save_model/SingleWithProto', help='Model save dir')
    parser.add_argument('--epsilon_sparsity', type=float, default='0.000001', help='epsilon_sparsity')
    parser.add_argument('--sparsity_weight', type=float, default='0.0001', help='sparsity_weight')
    args = parser.parse_args()
    args = parser.parse_args()
    print(args.gpu)

    train(args)