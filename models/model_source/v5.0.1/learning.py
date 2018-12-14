import sys
from dataset import Dataset
import tqdm
import time
from cnn import Net32
import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
import sendAMail as send

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getArgs():
    parser = argparse.ArgumentParser(description='Setting model')
    parser.add_argument('-l','--learning',type=str, choices=['images','landmarks','images_landmarks'], default='images', help='Select the learning type')
    args = parser.parse_args()
    return args.learning




def save_checkpoint(state, is_best, checkpoint_path,best_model_path):
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, best_model_path)

def startLearning(mode:str,bs,me,f,p,l):
    print(device)
    #cudnn.Benchmark = True
    writer = SummaryWriter()
    

    batch_size = bs
    max_epochs = me
    factor = f
    patience = p
    lr = l
    print("load dataset")
    training_loader = torch.utils.data.DataLoader(dataset=Dataset('training_dataset_pack.h5',mode,"std_training.png","mean_training.png"), batch_size=batch_size, shuffle=True,num_workers=0)
    validation_loader = torch.utils.data.DataLoader(dataset=Dataset('validation_dataset_pack.h5',mode,"std_training.png","mean_training.png"), batch_size=batch_size, shuffle=True,num_workers=0)

    print("Done")

    
    
    model = Net32(training_loader.dataset.getInputSize()).cuda()
    


    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999))  
    #déclarer scheduler reduce plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=factor,patience=patience)
    best_loss = None
    ######

    
    ######
    for epoch in range(1,max_epochs+1):  # loop over the dataset multiple times

        
        is_best= False
        print("Training...")
        pbart = tqdm.tqdm(total=int(len(training_loader.dataset)/batch_size),postfix={"loss":None,"accuracy":None},desc="Epoch: {}/{}".format(epoch,max_epochs))
        loss = 0.
        acc = 0.
        val_loss = 0.
        val_acc =0.

        for i, data in enumerate(training_loader, 0):
            # get the inputs
            images, expressions = data
            images = images.to(device)
            expressions = expressions.to(device).long()
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(images)
            current_loss = criterion(outputs, expressions)
            current_loss.backward()
            optimizer.step()

            # print statistics
            loss += current_loss.item()
            _,predicted = torch.max(outputs,1)
            acc += (predicted == expressions).sum().float()/batch_size

            pbart.update(1)
            pbart.set_postfix({"loss": loss/(i+1),"accuracy":acc.item()/(i+1)})

        pbart.close()
        loss /= (len(training_loader.dataset)/batch_size)
        acc /= (len(training_loader.dataset)/batch_size)
        
        with open("log/training_"+mode+".log","a") as f:
            f.write("epoch: {} / {} loss: {} accuracy: {}\n".format(epoch,max_epochs,loss,acc))
        f.close()

        
        print("Validation...")
        pbarv = tqdm.tqdm(total=int(len(validation_loader.dataset)/batch_size),postfix={"loss":None,"accuracy":None},desc="Epoch: {}/{}".format(epoch,max_epochs))

        
        with torch.no_grad():
            for i, data in enumerate(validation_loader, 0):
                # get the inputs
                images, expressions = data
                images = images.to(device)
                expressions = expressions.to(device).long()

                outputs = model(images)
                loss = criterion(outputs, expressions)
                val_loss += loss.item()

                _,predicted = torch.max(outputs,1)
                val_acc  +=(predicted == expressions).sum().float()/batch_size

                pbarv.update(1)
                pbarv.set_postfix({"loss": val_loss/(i+1),"accuracy":val_acc.item()/(i+1)})

            
        pbarv.close()
        val_loss /= (len(validation_loader.dataset)/batch_size) 
        val_acc /= (len(validation_loader.dataset)/batch_size) 
        
        

        if best_loss == None or val_loss < best_loss:
            best_loss = val_loss
            is_best = True
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss': val_loss,
            'accuracy': val_acc,
            'optimizer' : optimizer.state_dict()}, is_best,"save_model/checkpoint_"+mode+".pth","save_model/best_model_"+mode+".pth")
        is_best = False

        scheduler.step(val_loss)
        #loss accuracy training et validation dans les logs
        
        with open("log/validation_"+mode+".log","a") as f:
            f.write("epoch: {} / {} loss: {} accuracy: {}\n".format(epoch,max_epochs,val_loss,val_acc))
        f.close()

        writer.add_scalar('data/Loss training', loss, epoch)
        writer.add_scalar('data/Loss validation', val_loss, epoch)
        writer.add_scalar('data/Accuracy training', acc, epoch)
        writer.add_scalar('data/Accuracy validation', val_acc, epoch)

        if (epoch%30 == 1 or epoch == max_epochs):
            desc = dict()
            desc["bs"] = batch_size
            desc["lr"] = lr
            desc["f"] = factor
            desc["p"] = patience
            desc["d"] = 0
            desc["weights"] = [1,1]
            desc["epoch"] = epoch
            desc["nbepochs"] = max_epochs

            send.sendInfos("1 (3 models)",desc,loss,acc,"...","...","...","...",val_loss,val_acc,"...","...","...","...")






    
if __name__ == "__main__":
    print(sys.version)
    args =getArgs()
    mode = args
    with open("log/training_"+mode+".log","w") as f:
        f.close()
    with open("log/validation_"+mode+".log","w") as f:
        f.close()

    batch_size = 64
    max_epochs = 150
    factor = 0.1
    patience = 10
    lr = 0.0001
    startLearning(mode,batch_size,max_epochs,factor,patience,lr)
