import os
import time
import argparse

import torch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from model import SSD300
from model import MultiBoxLoss
from data import TextDataset, MyTransform
# precision = 'fp32'
# ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision, pretrained=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://zhuanlan.zhihu.com/p/37626738

def train_loop(model, loss_func, optim, epoch, iteration, train_dataloader, writer, args):
    start = time.time()
    for nbatch, (images, bboxes, labels) in enumerate(train_dataloader):
        # images = torch.tensor(images).to(device)
        # bboxes = torch.tensor(bboxes).to(device)
        # labels = torch.tensor(labels).to(device)
        images = images.to(device)
        bboxes = [b.to(device) for b in bboxes]
        labels = [l.to(device) for l in labels]
        
        ploc, plabel = model(images)
        # ploc, plabel = ploc.float(), plabel.float()
        
        loss = loss_func(ploc, plabel, bboxes, labels)
        
        if args.log_to_board:
            writer.add_scalar('Loss/Loss', loss, iteration)
            writer.add_scalar(f'Epoch_Loss/Epoch_{epoch}_Loss', loss, nbatch)

        if args.log_to_console:
            print(f'Epoch : {epoch}, Num of Batch : {nbatch}, Loss : {loss}')
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        iteration += 1
    
    end = time.time()
    if args.log_to_board:
        writer.add_text('Epoch_Time', f'Time : {end - start}', epoch)
    print(f'Epoch Time : {end - start}')
    
    return iteration

def validation_loop():
    pass

def make_parser():
    parser = argparse.ArgumentParser(description="Train Single Shot MultiBox Detector")
    parser.add_argument('--root', type=str, default='./manga109')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--log_to_console', type=bool, default=False)
    parser.add_argument('--log_to_board', type=bool, default=True)
    return parser

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    
    num_classes = 2
    
    # text and another
    model = SSD300(n_classes=num_classes)
    model.to(device)
    
    loss_func = MultiBoxLoss(priors_cxcy=model.priors_cxcy)
    loss_func.to(device)
    
    optim = torch.optim.SGD(model.parameters(), 
                                lr=0.001,
                                momentum=0.9,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.1)
    
    transform = MyTransform()
    root_folder = args.root
    # create dataloader
    train_dataset = TextDataset(root_folder, 'train', transform)
    # val_dataset = TextDataset(root_folder, 'test', transform)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batchSize,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        num_workers=0)
    # val_dataloader = DataLoader(
        # val_dataset, 
        # batch_size=args.batchSize,
        # collate_fn=val_dataset.collate_fn,
        # pin_memory=True,
        # num_workers=0)
    
    writer = SummaryWriter()
    
    iteration = 0
    model.train()
    start = time.time()
    for epoch in range(30):
        iteration = train_loop(model, loss_func, optim, epoch, iteration, train_dataloader, writer, args)
        
        if args.save:
            print(f'saving model...{epoch}')
            obj = { 
                'epoch'     : epoch,
                'iteration' : iteration,
                'optimizer' : optim.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'model'     : model.state_dict()
                # 'model'     : model
            }
            save_path = os.path.join('./models', f'epoch_{epoch}.pt')
            torch.save(obj, save_path)
            
        # run validate function
        scheduler.step()
    end = time.time()
    
    if args.log_to_board:
        writer.add_text('Total Train Time', f'Time : {end - start}', 0)
    if args.log_to_console:
        print(f'Total Train Time : {end-start}')
    
    writer.close()
    
    