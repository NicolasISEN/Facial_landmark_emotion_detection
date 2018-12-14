import sys
from dataset import Dataset
import tqdm
import time
from autoencoder import AutoEncoder
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


def save_checkpoint(state, is_best, checkpoint_path,best_model_path):
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, best_model_path)

def startLearning(bs,me,f,p,l):
    #Init Tensorboard
    writer = SummaryWriter()
    batch_size = bs
    max_epochs = me
    factor = f
    patience = p
    lr = l
    #Define batch size the number of epoch
    
    print("load dataset")
    #Load Dataset
    training_loader = torch.utils.data.DataLoader(dataset=Dataset('training_dataset_pack.h5',"std_training.png","mean_training.png"), batch_size=batch_size, shuffle=True,num_workers=0)
    validation_loader = torch.utils.data.DataLoader(dataset=Dataset('validation_dataset_pack.h5',"std_validation.png","mean_validation.png"), batch_size=batch_size, shuffle=True,num_workers=0)

    print("Done")

    #Make model
    model = AutoEncoder(training_loader.dataset.getInputSize()).cuda()
    


    #Define loss type
    criterion_expressions = nn.CrossEntropyLoss().cuda()
    criterion_landmarks = nn.MSELoss().cuda()

    #Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999))  

    #Define the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=factor,patience=patience)


    best_loss = None
    
    #Main loop (epoch)
    for epoch in range(1,max_epochs+1):  

        
        is_best= False
        print("Training...")
        #Init progress bar for training
        pbart = tqdm.tqdm(total=int(len(training_loader.dataset)/batch_size),postfix={"loss":None,"accuracy":None},desc="Epoch: {}/{}".format(epoch,max_epochs))

        #Init metrics
        loss = 0.
        loss_lm = 0.
        loss_expr = 0.
        acc = 0.
        val_loss = 0.
        val_acc =0.
        val_loss_lm = 0.
        val_loss_expr = 0.

        #Training loop
        for i, data in enumerate(training_loader, 0):
            #Zero the parameter gradients
            optimizer.zero_grad()

            #Get the inputs
            images, landmarks, expressions = data
            images = images.to(device)
            landmarks = landmarks.to(device).float()
            expressions = expressions.to(device).long()
            
            #Get the outputs
            outputs = model(images)

            #Calculate metrics
            #Loss
            loss_landmarks = criterion_landmarks(outputs[0], landmarks.float())
            loss_expressions = criterion_expressions(outputs[1], expressions)
            current_loss = loss_landmarks + 0.0001*loss_expressions
            loss_expr += loss_expressions.item()
            loss_lm += loss_landmarks.item()
            loss += current_loss.item()
            #Accuracy
            _,predicted_expressions = torch.max(outputs[1],1)
            acc += (predicted_expressions == expressions).sum().float()/batch_size

            #Backpropagation
            current_loss.backward()

            #Reduce learning rate if we are on a plateu
            optimizer.step()

            #Update 
            pbart.update(1)
            pbart.set_postfix({"loss": loss/(i+1),"e_loss": loss_expr/(i+1),"l_loss": loss_lm/(i+1),"acc_e":acc.item()/(i+1)})
        pbart.close()

        #Calculate metrics on one epoch
        loss /= (len(training_loader.dataset)/batch_size)
        acc /= (len(training_loader.dataset)/batch_size)
        
        #Save metrics in a log file
        with open("log/training_hourglass5.0.4.log","a") as f:
            f.write("epoch: {} / {} loss: {} e_loss:{} l_loss: {} accuracy: {}\n".format(epoch,max_epochs,loss,loss_expr,loss_lm,acc))
        f.close()

        
        print("Validation...")
        #Init progress bar for validation
        pbarv = tqdm.tqdm(total=int(len(validation_loader.dataset)/batch_size),postfix={"loss":None,"accuracy":None},desc="Epoch: {}/{}".format(epoch,max_epochs))

        #Validation loop
        with torch.no_grad():
            for i, data in enumerate(validation_loader, 0):
                #Get the inputs
                images, landmarks, expressions = data
                images = images.to(device)
                landmarks = landmarks.to(device).float()
                expressions = expressions.to(device).long()

                #Get the outputs
                outputs = model(images)

                #Calculate metrics
                #Loss
                loss_landmarks = criterion_landmarks(outputs[0], landmarks.float())
                loss_expressions = criterion_expressions(outputs[1], expressions)
                loss = loss_landmarks + 0.0001*loss_expressions
                val_loss += loss.item()
                val_loss_expr += loss_expressions.item()
                val_loss_lm += loss_landmarks.item()

                #Accuracy
                _,predicted_expressions = torch.max(outputs[1],1)
                val_acc += (predicted_expressions == expressions).sum().float()/batch_size

                #Uptate validation progress bar
                pbarv.update(1)
                pbarv.set_postfix({"loss": val_loss/(i+1),"e_loss": val_loss_expr/(i+1),"l_loss": val_loss_lm/(i+1),"acc_e":val_acc.item()/(i+1)})

            
        pbarv.close()

        #Calculate metrics on one epoch
        val_loss /= (len(validation_loader.dataset)/batch_size) 
        val_acc /= (len(validation_loader.dataset)/batch_size) 
        
        
        #Save the weights of the model
        if best_loss == None or val_loss < best_loss:
            best_loss = val_loss
            is_best = True
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss': val_loss,
            'loss_expressions': val_loss_expr,
            'loss_landmarks' : val_loss_lm, 
            'accuracy': val_acc,
            'optimizer' : optimizer.state_dict()}, is_best,"save_model/checkpoint_hourglass5.0.4.pth","save_model/best_model_validation5.0.4.pth")
        is_best = False

        scheduler.step(val_loss)
        
        #Save metrics in a log file
        with open("log/validation_hourglass5.0.4.log","a") as f:
            f.write("epoch: {} / {} loss: {} e_loss:{} l_loss: {} accuracy: {}\n".format(epoch,max_epochs,val_loss,val_loss_expr,val_loss_lm,val_acc))
        f.close()

        #Construct tensorboard graph
        writer.add_scalar('data/Loss training', loss, epoch)
        writer.add_scalar('data/Loss landmarks training', loss_lm, epoch)
        writer.add_scalar('data/Loss expressions training', loss_expr, epoch)

        writer.add_scalar('data/Loss validation', val_loss, epoch)
        writer.add_scalar('data/Loss landmarks validation', val_loss_lm, epoch)
        writer.add_scalar('data/Loss expressions validation', val_loss_expr, epoch)

        writer.add_scalar('data/Accuracy training', acc, epoch)
        writer.add_scalar('data/Accuracy validation', val_acc, epoch)

        if (epoch%5 == 1 or epoch == max_epochs):
            desc = dict()
            desc["bs"] = batch_size
            desc["lr"] = lr
            desc["f"] = factor
            desc["p"] = patience
            desc["d"] = 0
            desc["weights"] = [1,0.0001]
            desc["epoch"] = epoch
            desc["nbepochs"] = max_epochs

            try:
                send.sendInfos("4 (Hourglass 5.0.4)",desc,loss,acc,loss_lm,"...",loss_expr,"...",val_loss,val_acc,val_loss_lm,"...",val_loss_expr,val_acc)
            except Exception as e:
                pass
            
    
if __name__ == "__main__":
    print(sys.version)
    with open("log/training_hourglass5.0.4.log","w") as f:
        f.close()
    with open("log/validation_hourglass5.0.4.log","w") as f:
        f.close()

    batch_size = 8
    max_epochs = 30
    factor = 0.1
    patience = 10
    lr = 0.0001
    startLearning(batch_size,max_epochs,factor,patience,lr)
