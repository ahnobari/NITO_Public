import torch

from torch import nn
from torch.nn import functional as F
import numpy as np

from tqdm import tqdm, trange

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os

class Trainer:
    def __init__(self, model, lr=1e-4, weight_decay=1e-4, cosine_schedule=True, lr_final=1e-5,
                 schedule_max_steps=100, SDF=False, Multi_Class=False, nabla_coef=0.1, device=None, 
                 multi_gpu=False, mixed_precision=True, DDP_train=True, Compile=True, checkpoint_path=None):
        
        self.multi_gpu = multi_gpu
        self.DDP = DDP_train if multi_gpu else False
        self.mixed_precision = mixed_precision
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model = model

        if hasattr(self.model, 'compile') and Compile:
            self.model.compile()
        
        if self.DDP:
            self.setup_ddp()
        elif self.multi_gpu and type(self.multi_gpu) is list:
            self.model = nn.DataParallel(self.model, device_ids=multi_gpu)
        elif self.multi_gpu:
            self.model = nn.DataParallel(self.model)
        else:
            self.model = self.model.to(self.device)
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.cosine_schedule = cosine_schedule
        self.lr_final = lr_final
        self.schedule_max_steps = schedule_max_steps
        
        if self.cosine_schedule:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=schedule_max_steps, eta_min=lr_final)
        else:
            self.scheduler = None
        
        self.current_epoch = 0
        
        self.SDF = SDF
        self.Multi_Class = Multi_Class
        self.nabla_coef = nabla_coef
        
        if self.SDF:
            self.loss_fn = nn.MSELoss()
            self.activation = nn.Identity()
        elif self.Multi_Class:
            self.loss_fn = nn.CrossEntropyLoss()
            self.activation = nn.Softmax(dim=1)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.activation = nn.Identity()
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def setup_ddp(self):
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12355'
        
        self.rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.rank)
        
        self.model = self.model.to(self.rank)
        
        self.model = DDP(self.model, device_ids=[self.rank])

    def cleanup_ddp(self):
        if self.DDP:
            dist.destroy_process_group()

    def is_main_process(self):
        return self.rank == 0 if self.DDP else True

    def save_checkpoint(self, path):
        if self.is_main_process():
            checkpoint = {
                'model_state_dict': self.model.module.state_dict() if isinstance(self.model, (nn.DataParallel, DDP)) else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'current_epoch': self.current_epoch,
            }
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['current_epoch']
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def train(self, loader_fn, data_idx, batch_size, epochs=100, continue_loop=True, verbose=True, checkpoint_interval=10, checkpoint_dir='Checkpoints', **kwargs):
        if not continue_loop:
            self.model.train()
            self.current_epoch = 0
            self.reset_optimizer()
        
        if self.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
        
        # split data for DDP
        if self.DDP:
            data_idx = np.array_split(data_idx, self.world_size)[self.rank]

        steps_per_epoch = int(np.ceil(len(data_idx) / batch_size))
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(epochs):
            if verbose and self.is_main_process():
                prog = tqdm(range(steps_per_epoch))
            else:
                prog = range(steps_per_epoch)

            epoch_loss = 0
            
            shuffle_idx = np.random.permutation(len(data_idx))
            
            for i in prog:
                if self.DDP:
                    if shuffle_idx[i*batch_size:(i+1)*batch_size].shape[0] < batch_size:
                        continue
                self.optimizer.zero_grad()
                inputs, labels = loader_fn(data_idx[shuffle_idx[i*batch_size:(i+1)*batch_size]], self.device)

                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        pred_labels = self.activation(self.model(inputs, **kwargs))
                        loss = self.loss_fn(pred_labels, labels)
                else:
                    pred_labels = self.activation(self.model(inputs, **kwargs))
                    loss = self.loss_fn(pred_labels, labels)
                
                if self.SDF:
                    loss_del = loss * 0.0
                    loss = loss + loss_del * self.nabla_coef
                    
                if self.mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                epoch_loss += loss.item()
                
                if self.is_main_process() and verbose:
                    if self.SDF:
                        prog.set_postfix_str(f"Epoch Loss: {epoch_loss/(i+1):.7f}, Recon Loss: {loss.item()-loss_del.item()*self.nabla_coef:.7f}, Loss Del: {loss_del.item():.7f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                    else:
                        prog.set_postfix_str(f"Epoch Loss: {epoch_loss/(i+1):.7f}, Loss: {loss.item():.7f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            self.current_epoch += 1
            if self.cosine_schedule and self.current_epoch <= self.schedule_max_steps:
                self.scheduler.step()
                
            if verbose and self.is_main_process():
                print(f'Epoch {self.current_epoch}, Loss: {epoch_loss/steps_per_epoch}')
            

            self.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}.pth'))

            if (self.current_epoch-1) % checkpoint_interval == 0:
                pass
            else:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint_epoch_{self.current_epoch-1}.pth'))

        if self.DDP:
            dist.barrier()

    def __del__(self):
        self.cleanup_ddp()