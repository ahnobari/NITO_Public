import torch

from torch import nn
from torch.nn import functional as F
import numpy as np

from tqdm import tqdm, trange

class Trainer:
    def __init__(self, model, lr = 1e-4, weight_decay=1e-4, cosine_schedule = True, lr_final=1e-5,
                 schedule_max_steps = 100, SDF=False, Multi_Class=False, nabla_coef = 0.1, device=None, multi_gpu=False, mixed_precision = True):
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.model = model.to(device)

        self.multi_gpu = multi_gpu
        self.mixed_precision = mixed_precision

        if self.multi_gpu:
            if type(self.multi_gpu) is list:
                self.model = nn.DataParallel(self.model, device_ids = multi_gpu)
            else:
                self.model = nn.DataParallel(self.model)
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(list(self.model.parameters()), lr = lr, weight_decay=weight_decay)
        
        self.cosine_schedule = cosine_schedule
        self.lr_final = lr_final
        self.schedule_max_steps = schedule_max_steps
        
        if self.cosine_schedule:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = schedule_max_steps, eta_min = lr_final)
        
        self.current_epoch = 0
        
        self.device = device
        self.SDF = SDF
        self.Multi_Class = Multi_Class
        self.nabla_coef = nabla_coef
        
        if self.SDF:
            self.loss_fn = torch.nn.MSELoss()
            self.activation = torch.nn.Identity()
        elif self.Multi_Class:
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.activation = torch.nn.Softmax(dim=1)
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
            self.activation = torch.nn.Identity()
            
    def reset_optimizer(self):
        self.optimizer = torch.optim.AdamW(list(self.model_input.parameters()) + list(self.model_base.parameters()), lr = self.lr, weight_decay=self.weight_decay)
        if self.cosine_schedule:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.schedule_max_steps, eta_min = self.lr_final)
            
    
    def train(self, loader_fn, data_idx ,batch_size, epochs = 100, continue_loop=True, verbose = True, **kwargs):
        if continue_loop:
            self.model.train()
        else:
            self.model.train()
            self.current_epoch = 0
            self.reset_optimizer()

        self.model.compile()

        if self.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
        
        steps_per_epoch = int(np.ceil(len(data_idx)/batch_size))
        
        for epoch in range(epochs):
            if verbose:
                prog = tqdm(range(steps_per_epoch))
            else:
                prog = range(steps_per_epoch)

            epoch_loss = 0
            
            shuffle_idx = np.random.permutation(len(data_idx))
            
            for i in prog:
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
                else:
                    loss.backward()
                
                #gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20.0)
                
                # skip if loss is nan
                # if torch.isnan(loss) or torch.isinf(loss):
                #     continue

                if self.mixed_precision:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                
                epoch_loss += loss.item()
                
                if self.SDF:
                    prog.set_postfix_str(f"Epoch Loss: {epoch_loss/(i+1):.7f}, Recon Loss: {loss.item()-loss_del.item()*self.nabla_coef:.7f}, Loss Del: {loss_del.item():.7f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    prog.set_postfix_str(f"Epoch Loss: {epoch_loss/(i+1):.7f}, Loss: {loss.item():.7f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
            self.current_epoch += 1
            if self.cosine_schedule and self.current_epoch <= self.schedule_max_steps:
                self.scheduler.step()
                
            if verbose:
                print(f'Epoch {self.current_epoch}, Loss: {epoch_loss/steps_per_epoch}')
    
    def eval(self, loader_fn, data_idx, batch_size, verbose = True, **kwargs):
        self.model.eval()
        
        steps_per_epoch = int(np.ceil(len(data_idx)/batch_size))
        
        epoch_loss = 0
        
        if verbose:
            prog = tqdm(range(steps_per_epoch))
        else:
            prog = range(steps_per_epoch)
        
        pred = []
        
        with torch.no_grad():
            for i in prog:
                inputs, labels = loader_fn(data_idx[i*batch_size:(i+1)*batch_size], self.device)
                
                pred_labels = self.activation(self.model(inputs, **kwargs))
                
                loss = self.loss_fn(pred_labels, labels)
                
                epoch_loss += loss.item()
                
                pred.append(pred_labels.detach().cpu().numpy())
                
        labels = np.concatenate(pred)
                
        if verbose:
            print(f'Validation Loss: {epoch_loss/steps_per_epoch}')
            
        return labels