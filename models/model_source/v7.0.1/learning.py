import sys
from dataset import Dataset
import tqdm
import time
from emotionnet import EmotionNet
import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(state, is_best, checkpoint_path,best_model_path):
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, best_model_path)

def startLearning():
    #Init Tensorboard
    writer = SummaryWriter()

    #Define batch size the number of epoch
    batch_size = 16
    max_epochs = 100
    

    #Make model
    model = EmotionNet().cuda()
    


    #Define loss type
    criterion_expressions = nn.CrossEntropyLoss().cuda()

    #Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999))  

    #Define the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=5)


    best_acc = None
    

    print("load dataset")
    #Load Dataset

    loader_filenames = np.asarray(["training_dataset_pack2.h5","training_dataset_pack3.h5","training_dataset_pack0.h5","training_dataset_pack1.h5"])
    validation_loader = torch.utils.data.DataLoader(dataset=Dataset('validation_dataset_pack0.h5',"std_training.png","mean_training.png"), batch_size=batch_size, shuffle=True,num_workers=2)

    print("Done")


    #Main loop (epoch)
    for epoch in range(1,max_epochs+1):  

        np.random.shuffle(loader_filenames)
        is_best= False
        print("Training...")
        

        #Init metrics
        loss_exp = 0.
        acc_exp = 0.
        expressions_loss = 0.
        expressions_acc = 0.
        count = 0
        for loader_filename in loader_filenames:
            training_loader = torch.utils.data.DataLoader(dataset=Dataset(loader_filename,"std_training.png","mean_training.png"), batch_size=batch_size, shuffle=True,num_workers=0)
            #Init progress bar for training
            pbart = tqdm.tqdm(total=int(len(training_loader.dataset)/batch_size),postfix={"loss_e":None, "acc_e":None},desc="Epoch: {}/{}".format(epoch,max_epochs))
            #Training loop
            for i, data in enumerate(training_loader, 0):
                count+=1
                #Zero the parameter gradients
                optimizer.zero_grad()
    
                #Get the inputs
                images, expressions = data
                images = images.to(device)
                expressions = expressions.to(device).long()
                
                #Get the outputs
                output = model(images)
    
                #Calculate metrics
                #Loss
                expressions_loss = criterion_expressions(output, expressions)
                loss_exp += expressions_loss.item()

                #Accuracy
                _,predicted_expressions = torch.max(output,1)
                expressions_acc += (predicted_expressions == expressions).sum().float()/batch_size
                acc_exp += (predicted_expressions == expressions).sum().float().item()/batch_size
    
                #if epoch==2 and i ==5:
                #    print(predicted_landmarks.detach().cpu().numpy().shape,landmarks.cpu().numpy().shape)
                #    print(predicted_landmarks.detach().cpu().numpy()[0,:],landmarks.cpu().numpy()[0,:])
                #    print(np.abs(predicted_landmarks.detach().cpu().numpy()[0,:]-landmarks.cpu().numpy()[0,:]))


                #Backpropagation
                expressions_loss.backward()


                #Reduce learning rate if we are on a plateau
                optimizer.step()
    
                #Update 
                pbart.update(1)
                pbart.set_postfix({"loss_e":loss_exp/count, "acc_e":expressions_acc.item()/count})
            pbart.close()

        #Calculate metrics on one epoch
        loss_exp /=  float(count)/batch_size
        acc_exp /= float(count)/batch_size
        #Save metrics in a log file
        with open("log/training_DAN.log","a") as f:
            f.write("epoch: {} / {} loss_e: {} acc_e: {}\n".format(epoch,max_epochs,loss_exp,acc_exp))
        f.close()

        #Construct tensorboard graph
        writer.add_scalar('data/Loss expressions training', loss_exp, epoch)
        writer.add_scalar('data/Accuracy expressions training', acc_exp, epoch)


        #Init metrics
        loss_exp = 0.
        acc_exp = 0.
        expressions_loss = 0.
        expressions_acc = 0.
        count = 0

        print("Validation...")
        #Init progress bar for validation
        pbarv = tqdm.tqdm(total=int(len(validation_loader.dataset)/batch_size),postfix={"loss_e":None, "acc_e":None},desc="Epoch: {}/{}".format(epoch,max_epochs))

        #Validation loop
        with torch.no_grad():
            for i, data in enumerate(validation_loader, 0):
                count+=1
                #Get the inputs
                images, expressions = data
                images = images.to(device)
                expressions = expressions.to(device).long()

                #Get the outputs
                output = model(images)
                
                #Calculate metrics
                #Loss
                expressions_loss = criterion_expressions(output, expressions)
                loss_exp += expressions_loss.item()

                #Accuracy
                _,predicted_expressions = torch.max(output,1)
                expressions_acc += (predicted_expressions == expressions).sum().float()/batch_size
                acc_exp += (predicted_expressions == expressions).sum().float().item()/batch_size
                
                #Uptate validation progress bar
                pbarv.update(1)
                pbarv.set_postfix({"loss_e":loss_exp/count, "acc_e":expressions_acc.item()/count})

            
        pbarv.close()

        #Calculate metrics on one epoch
        loss_exp /=  float(count)/batch_size
        acc_exp /= float(count)/batch_size
 
        
        #Save the weights of the model
        if best_acc == None or acc_exp >= best_acc:
            best_acc = loss_exp
            is_best = True
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss expressions': loss_exp,
            'accuracy expressions': acc_exp,
            'optimizer' : optimizer.state_dict()}, is_best,"save_model/checkpoint_DAN.pth","save_model/best_model_validation.pth")
        is_best = False

        scheduler.step(loss_exp)
        
        #Save metrics in a log file
        with open("log/validation_DAN.log","a") as f:
            f.write("epoch: {} / {} loss_e: {} acc_e:{}\n".format(epoch,max_epochs,loss_exp,acc_exp))
        f.close()

        #Construct tensorboard graph
        writer.add_scalar('data/Loss expressions validation', loss_exp, epoch)
        writer.add_scalar('data/Accuracy expressions validation', acc_exp, epoch)

    
if __name__ == "__main__":
    print(sys.version)
    with open("log/training_DAN.log","w") as f:
        f.close()
    with open("log/validation_DAN.log","w") as f:
        f.close()

    
    startLearning()
