import sys
from dataset import Dataset
import tqdm
import time
from cnn import Net32, Net256
import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getArgs():
    parser = argparse.ArgumentParser(description='Setting model')
    parser.add_argument('-m','--model', type=int, choices=[32,256], default=32, help='Choose the model')
    parser.add_argument('-l','--learning',type=str, choices=['images','landmarks','images_landmarks'], default='images', help='Select the learning type')
    args = parser.parse_args()
    return args.model,args.learning




def save_checkpoint(state, is_best, checkpoint_path,best_model_path):
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, best_model_path)

def startLearning(model_type:int ,mode:str):
    print(device)
    #cudnn.Benchmark = True
    resnet18 = models.resnet18(False)
    writer = SummaryWriter()
    batch_size = 32
    max_epochs = 150
    print("load dataset")
    loader = torch.utils.data.DataLoader(dataset=Dataset('data.h5',model_type), batch_size=batch_size, shuffle=True,num_workers=0)
    print("Done")

    if model_type ==256:
        model = Net256(loader.dataset.getInputSize()).cuda()
    elif model_type == 32:
        model = Net32(loader.dataset.getInputSize()).cuda()
    else:
        print("Model doesn't exist")
        return


    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9,0.999))  
    #déclarer scheduler reduce plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=10)
    best_loss = None
    ######

    
    ######
    for epoch in range(1,max_epochs+1):  # loop over the dataset multiple times

        running_loss = 0.
        is_best= False
        loader.dataset.training()
        print("Training...")
        pbart = tqdm.tqdm(total=int(len(loader.dataset)/batch_size),postfix={"loss":None,"accuracy":None},desc="Epoch: {}/{}".format(epoch,max_epochs))
        acc = 0
        val_loss = 0.
        val_acc =0.

        for i, data in enumerate(loader, 0):
            # get the inputs
            images, expressions = data
            images = images.to(device)
            expressions = expressions.to(device).long()
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, expressions)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _,predicted = torch.max(outputs,1)
            acc += (predicted == expressions).sum().float()/batch_size

            pbart.update(1)
            pbart.set_postfix({"loss": running_loss/(i+1),"accuracy":acc.item()/(i+1)})

        pbart.close()
        running_loss /= (len(loader.dataset)/batch_size)
        acc /= (len(loader.dataset)/batch_size)
        
        with open("log/training"+"_"+str(model_type)+"_"+mode+".log","a") as f:
            f.write("epoch: {} / {} loss: {} accuracy: {}\n".format(epoch,max_epochs,running_loss,acc))
        f.close()

        
        print("Validation...")
        loader.dataset.validation()
        pbarv = tqdm.tqdm(total=int(len(loader.dataset)/batch_size),postfix={"loss":None,"accuracy":None},desc="Epoch: {}/{}".format(epoch,max_epochs))

        
        with torch.no_grad():
            for i, data in enumerate(loader, 0):
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
        val_loss /= (len(loader.dataset)/batch_size) 
        val_acc /= (len(loader.dataset)/batch_size) 
        
        

        if best_loss == None or val_loss < best_loss:
            best_loss = val_loss
            is_best = True
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss': val_loss,
            'accuracy': val_acc,
            'optimizer' : optimizer.state_dict()}, is_best,"save_model/checkpoint"+"_"+str(model_type)+"_"+mode+".pth","save_model/best_model"+"_"+str(model_type)+"_"+mode+".pth")
        is_best = False

        scheduler.step(val_loss)
        #loss accuracy training et validation dans les logs
        
        with open("log/validation"+"_"+str(model_type)+"_"+mode+".log","a") as f:
            f.write("epoch: {} / {} loss: {} accuracy: {}\n".format(epoch,max_epochs,val_loss,val_acc))
        f.close()

        writer.add_scalar('data/Loss training', running_loss, epoch)
        writer.add_scalar('data/Loss validation', val_loss, epoch)
        writer.add_scalar('data/Accuracy training', acc, epoch)
        writer.add_scalar('data/Accuracy validation', val_acc, epoch)

        loader.dataset.shuffle_training()







    
if __name__ == "__main__":
    print(sys.version)
    args =getArgs()
    model_type = args[0]
    mode = args[1]
    with open("log/training"+"_"+str(model_type)+"_"+mode+".log","w") as f:
        f.close()
    with open("log/validation"+"_"+str(model_type)+"_"+mode+".log","w") as f:
        f.close()

    
    startLearning(model_type,mode)
