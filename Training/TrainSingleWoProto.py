import torch
import matplotlib.pyplot as plt
import argparse,os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('../NCKHECG')
from  Training.Loss import FocalLoss,MacroF1Loss,CrossEntropyWithWeights
from DataProcessing.LoadTrainData import Singledata_train_test 
from ModelArch.BackBone import SingleBackBoneNet
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

    model=SingleBackBoneNet()
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

    for epoch in range(args.epochs):
        loss_log=AvgrageMeter()
        f1_macro_log=AvgrageMeter()
        precision_macro_log=AvgrageMeter()
        recall_macro_log=AvgrageMeter()
        accuracy_log=AvgrageMeter()
        print("######################  TRAIN  #####################")
        model.train()

        train_data=Singledata_train_test(train_or_test='train',test_mode=test_mode)
        dataloader_train=DataLoader(train_data,batch_size=args.batchsize,shuffle=True)
        for i, sample_batched in enumerate(dataloader_train):
            inputs,(labels,one_hot_labels)=sample_batched
            inputs=torch.stack(inputs).permute(1,0,2)
            one_hot_labels=torch.stack(one_hot_labels).transpose(1,0)

            if args.gpu:
                inputs,labels,one_hot_labels=inputs.cuda(),labels.cuda(),one_hot_labels.cuda()
            else:
                inputs,labels,one_hot_labels=inputs.cpu(),labels.cpu(),one_hot_labels.cpu()

            optimizer.zero_grad()
            
            predict_logits,attention_score=model(inputs.float())
            loss=0.9*macro_f1_lost(y_pred=predict_logits, y_true=one_hot_labels.float())\
                 +0.1*ce_weights_lost(y_pred=predict_logits, y_true=labels.long())

            loss.backward()
            optimizer.step()

            n = inputs.size(0)
            predict_labels=torch.argmax(predict_logits,dim=1).cpu().data.numpy()
            labels=labels.cpu().data.numpy()
            loss_log.update(loss.cpu().data,n)

            print(labels)
            print(predict_labels)
            f1_macro_log.update(f1_score(y_true=labels,y_pred=predict_labels,average='macro'),n)
            precision_macro_log.update(precision_score(y_true=labels,y_pred=predict_labels,average='macro'),n)
            recall_macro_log.update(recall_score(y_true=labels,y_pred=predict_labels,average='macro'),n)
            accuracy_log.update(accuracy_score(y_true=labels,y_pred=predict_labels),n)

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
            dataloader_val = DataLoader(val_data, batch_size=args.batchsize, shuffle=True)

            for i,sample_batched in enumerate(dataloader_val):
                inputs,(labels,one_hot_labels)=sample_batched
                inputs=torch.stack(inputs).permute(1,0,2)
                one_hot_labels=torch.stack(one_hot_labels).transpose(1,0)

                if args.gpu:
                    inputs,labels,one_hot_labels=inputs.cuda(),labels.cuda(),one_hot_labels.cuda()
                else:
                    inputs,labels,one_hot_labels=inputs.cpu(),labels.cpu(),one_hot_labels.cpu()

                optimizer.zero_grad()

                n = inputs.size(0)
                predict_logits,attention_score=model(inputs.float())

                predict_labels=torch.argmax(predict_logits,dim=1).cpu().data.numpy()
                labels=labels.cpu().data.numpy()

                f1_macro_log_val.update(f1_score(y_true=labels,y_pred=predict_labels,average='macro'),n)
                precision_macro_log_val.update(precision_score(y_true=labels,y_pred=predict_labels,average='macro'),n)
                recall_macro_log_val.update(recall_score(y_true=labels,y_pred=predict_labels,average='macro'),n)
                accuracy_log_val.update(accuracy_score(y_true=labels,y_pred=predict_labels),n)
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

    parser.add_argument('--log', action='store_true', default='Training_log/SingleWoProto', help='training log path')
    parser.add_argument('--gpu', type=bool, default=False, help='Use gpu or cpu')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batchsize', type=int, default=256, help='initial batchsize')
    parser.add_argument('--step_size', type=int, default=5, help='how many epochs lr decays once')
    parser.add_argument('--val_per_epoch', type=int, default=1, help='how many train epoch per val')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--epochs', type=int, default=1000, help='total training epochs')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--echo_batches', type=int, default=1, help='how many batches display once')
    parser.add_argument('--model_save_dir', action='store_true', default='Save_model/SingleWoProto', help='Model save dir')
    args = parser.parse_args()
    print(args.gpu)

    train(args)



