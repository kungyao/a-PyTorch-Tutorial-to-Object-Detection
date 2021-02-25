import os
import time
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from model_fork import build_fork_model_and_loss_function
from data import TextDataset, MyTransform, my_collate_fn
# precision = 'fp32'
# ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision, pretrained=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://zhuanlan.zhihu.com/p/37626738
log_threshold = 100
def train_loop(model, loss_func, optim, epoch, iteration, train_dataloader, writer, args):
    start = time.time()
    avg_loss = 0
    for nbatch, (images, bboxes, labels) in enumerate(train_dataloader):
        #
        images = images.to(device)
        for nb in range(len(bboxes)):
            for c in range(args.n_classes):
                bboxes[nb][c] = bboxes[nb][c].to(device)
                labels[nb][c] = labels[nb][c].to(device)
        
        ploc, plabel = model(images)
        loss = loss_func(ploc, plabel, bboxes, labels)
        
        avg_loss = avg_loss + loss
        if (nbatch + 1) % log_threshold == 0:
            avg_loss = avg_loss / log_threshold
            writer.add_scalar('Loss/Loss', avg_loss, iteration)
            writer.add_scalar(f'Epoch_Loss/Epoch_{epoch}_Loss', avg_loss, nbatch + 1)
            print(f'Epoch : {epoch}, Num of Batch : {nbatch + 1}, Loss : {avg_loss}')
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        iteration += 1
    
    end = time.time()
    print(f'Epoch Time : {end - start}')
    return iteration

def validation_loop():
    pass

def get_args():
    parser = argparse.ArgumentParser(description="Train Single Shot MultiBox Detector")
    parser.add_argument('--root', type=str, default='./manga109')
    parser.add_argument('--batchsize', type=int, default=1)
    # default class is 5, include (negative、frame、text、face、body). but ssd-fork class count do not need negative, so n_classes is 4.
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=5)
    return parser

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    
    model, loss_func = build_fork_model_and_loss_function(args.n_classes)
    model.to(device)
    loss_func.to(device)
    
    optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.1)
    
    # create dataloader
    dataset = TextDataset(args.root, model_type='ssd-fork', transforms=MyTransform())

    # according to fork paper
    test_size = 927
    # test_size = 20
    indices = list(range(len(dataset)))
    # split data set to training and testing
    train_set = torch.utils.data.Subset(dataset, indices[:-test_size]) 
    test_set = torch.utils.data.Subset(dataset, indices[-test_size:])
    
    train_dataloader = DataLoader(
        train_set,
        batch_size=args.batchsize,
        collate_fn=partial(my_collate_fn),
        shuffle=True,
        num_workers=4)
    val_dataloader = DataLoader(
        test_set, 
        batch_size=args.batchsize,
        collate_fn=partial(my_collate_fn),
        shuffle=False,
        num_workers=4)
    
    writer = SummaryWriter()
    
    iteration = 0
    model.train()
    start = time.time()
    for epoch in range(args.epoch):
        iteration = train_loop(model, loss_func, optim, epoch, iteration, train_dataloader, writer, args)
        
        print(f'saving model...{epoch}')
        obj = { 
            'epoch'     : epoch,
            'iteration' : iteration,
            'optimizer' : optim.state_dict(),
            'scheduler' : scheduler.state_dict(),
            # 'model'     : model.state_dict(), 
            'model'     : model, # save model module
        }
        save_path = os.path.join('./models', f'epoch_{epoch}.pt')
        torch.save(obj, save_path)
            
        # run validate function
        scheduler.step()
    end = time.time()
    
    writer.add_text('Total Train Time', f'Time : {end - start}', 0)
    print(f'Total Train Time : {end-start}')
    
    writer.close()
    
    