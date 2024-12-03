import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import socket

import time

from metrics.spot import SPOT
from utils.utils import *
from model.TMino_prediction import TMino


from sklearn.preprocessing import StandardScaler,MinMaxScaler
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
torch.set_printoptions(precision=15)              


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
            
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def compute_loss(q, k, k_negs, T=0.5):          
        
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, k_negs])

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = F.cross_entropy(logits, labels)

        return loss
 
def hierarchical_contrastive_loss(z1, z2, z3, zneg1):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)

    z = torch.cat([z1, z2, z3, zneg1], dim=0)  
    z = z.transpose(0, 1)  
    sim = torch.matmul(z, z.transpose(1, 2))  

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]

    logits = F.softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    temp1 = -torch.log(logits[:, i, B + i - 1] + logits[:, i, 2 * B + i - 1])

    temp2 = -torch.log(logits[:, B + i, i] + logits[:, 2 * B + i, i])

    loss = (temp1.mean() + temp2.mean()) / 2

    return loss


def temporal_contrastive_loss(z1, z2, zneg1, zneg2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  
    sim = torch.matmul(z, z.transpose(1, 2))  
    
    sim_neg1 = torch.matmul(z1, zneg1.transpose(1, 2))  
    sim_neg2 = torch.matmul(z2, zneg2.transpose(1, 2))  
    
    logits_pos = torch.tril(sim, diagonal=-1)
    logits_neg1 = sim_neg1.diagonal(dim1=-2, dim2=-1)
    logits_neg2 = sim_neg2.diagonal(dim1=-2, dim2=-1)
    
    loss_pos = -F.log_softmax(logits_pos, dim=-1).mean()
    loss_neg1 = -F.log_softmax(logits_neg1, dim=-1).mean()
    loss_neg2 = -F.log_softmax(logits_neg2, dim=-1).mean()
    
    loss = (loss_pos + (loss_neg1 + loss_neg2) / 2) / 2
    return loss


def getThreshold(init_score, test_score, q=1e-2):
    s = SPOT(q=q)
    s.fit(init_score, test_score)
    s.initialize(verbose=False)
    ret = s.run()
    threshold = np.mean(ret['thresholds'])

    return threshold

def instance_point_wise_contrastive_loss(z1, z2, z3, zneg1):
    B, M = z1.size(0), z1.size(1) 
    if B == 1:
        return z1.new_tensor(0.)

    z = torch.cat([z1, z2, z3, zneg1], dim=0)  

    norms = torch.norm(z, p=2, dim=-1, keepdim=True) + 1e-10

    z = z / norms
    z = z.transpose(0, 1)  
    sim = torch.matmul(z, z.transpose(1, 2)) 

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]

    logits = F.softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    temp1 = -torch.log(logits[:, i, B + i - 1] + logits[:, i, 2 * B + i - 1])

    temp2 = -torch.log(logits[:, B + i, i] + logits[:, 2 * B + i, i])

    loss = (temp1.mean() + temp2.mean()) / 2

    return loss

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int,the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True).to(device)
        self.params = nn.Parameter(params)
        

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        a = 0.5 / (self.params[i] ** 2)
        return loss_sum

class ProjectionHead(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=128) -> None:    
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        
        self.proj_head = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        ).to(device)
        self.repr_dropout = nn.Dropout(p=0.1).to(device)
    
    def forward(self, x):
        x = self.repr_dropout(self.proj_head(x))
        return x



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint_detection.pth'))
        self.val_loss_min = val_loss




        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)
        self.config = config
        self.train_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset)   
        self.vali_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)
        
        self.build_model()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.k = 512   
        self.model.register_buffer('queue', F.normalize(torch.randn(self.d_model, self.k), dim=0))     
        self.model.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
        self.output_dims = 32
        self.proj = ProjectionHead(self.d_model, self.output_dims)


    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def build_model(self):             
        self.model = TMino(win_size=self.win_size, mask_ratio = self.mask_ratio, noise_ratio = self.noise_ratio,stride=self.stride,
                                enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads, 
                                d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size,
                                channel=self.input_c)


        if torch.cuda.is_available():
            self.model.cuda()
            
       
        self.optimizer = torch.optim.Adam(params=[{'params': self.model.parameters()}], lr=self.lr)

 
    def _denormalize(self, x):
        self.affine_bias = nn.Parameter(torch.zeros(self.output_c))
        self.affine_weight = nn.Parameter(torch.ones(self.output_c))
        self.eps = 1e-5
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

        x = x * self.stdev
        x = x + self.mean
        return x 
        
        
    @torch.no_grad()
    def _dequeue_and_enqueue0(self, keys):  
        batch_size = keys.shape[0]    

        ptr = int(self.model.queue_ptr)
        assert self.k % batch_size == 0           

        self.model.queue[:, ptr:ptr + batch_size] = keys.T         

        ptr = (ptr + batch_size) % self.win_size
        self.model.queue_ptr[0] = ptr
    
    
    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys):   
        batch_size = keys.shape[0]      

        ptr = int(self.model.queue_ptr)
        assert self.k % batch_size == 0       


        self.model.queue[:, ptr:ptr + batch_size] = keys.T         

        ptr = (ptr + batch_size) % self.win_size
        self.model.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):   
        batch_size = keys.shape[0]      

        ptr = int(self.model.queue_ptr)
        assert self.k % batch_size == 0       

   
        self.model.queue[:, ptr:ptr + batch_size] = keys.T       

        ptr = (ptr + batch_size) % self.win_size
        self.model.queue_ptr[0] = ptr
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        criterion = self._select_criterion()                
        
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            
            output0, output1, output2, outputs0_not, outputs1_not, outputs2_not,  contra0, contra1, contra2, noise0, noise1, noise2 = self.model(input) 


            loss0 = criterion(output0, input)      
            loss1 = criterion(output1, input)
            loss2 = criterion(output2, input)


                
            

            loss_contras0 = instance_point_wise_contrastive_loss(contra0, contra1, contra2, noise0)
            loss_contras1 = instance_point_wise_contrastive_loss(contra1, contra0, contra2, noise1)
            loss_contras2 = instance_point_wise_contrastive_loss(contra2, contra0, contra1, noise2)
           
    

            loss_contras = (loss_contras0 + loss_contras1 + loss_contras2 ) / 3
            
    
            mask_loss = (loss0 + loss1 + loss2 )/3

            con_loss = loss_contras

            loss = 0.6 * mask_loss + 0.4 * con_loss

            loss_1.append(loss.item())
        self.model.train()
        return np.average(loss_1), np.average(loss_2)


    def train(self):

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.data_path)
         

        criterion = self._select_criterion()


        for epoch in range(self.num_epochs):
            iter_count = 0
            train_loss = []                 
            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                
                input = input_data.float().to(self.device)

                output0, output1, output2, outputs0_not, outputs1_not, outputs2_not,  contra0, contra1, contra2, noise0, noise1, noise2 = self.model(input) 

                
                loss0 = criterion(output0, input)       
                loss1 = criterion(output1, input)
                loss2 = criterion(output2, input)
     

                loss_contras0 = instance_point_wise_contrastive_loss(contra0, contra1, contra2, noise0)
                loss_contras1 = instance_point_wise_contrastive_loss(contra1, contra0, contra2, noise1)
                loss_contras2 = instance_point_wise_contrastive_loss(contra2, contra0, contra1, noise2)
                

                loss_contras = (loss_contras0 + loss_contras1 + loss_contras2) / 3

                mask_loss = (loss0 + loss1 + loss2)/3

                con_loss = loss_contras

                loss = 0.6 * mask_loss + 0.4 * con_loss

                train_loss.append(loss.item())

                
                if (i + 1) % 100 == 0:

                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))       
           
                    iter_count = 0
                    time_now = time.time()
 
                loss.backward()
                self.optimizer.step()

            
            
            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))

            early_stopping(vali_loss1, self.model, path)        
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
        

            
    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.data_path) + '_checkpoint_detection.pth')))
        self.model.eval()
        temperature = 50

        # (1) stastic on the train set  
        attens_energy = []
        self.anomaly_criterion = nn.L1Loss(reduce=False)

        for i, (input_data, labels) in enumerate(self.train_loader):

            input = input_data.float().to(self.device)

            output0, output1, output2, outputs0_not, outputs1_not, outputs2_not, contra0, contra1, contra2, noise0, noise1, noise2 = self.model(input) 
           
            score0 = torch.mean(self.anomaly_criterion(outputs0_not, input), dim=-1)

            score0 = score0.detach().cpu().numpy()
            score1 = torch.mean(self.anomaly_criterion(outputs1_not, input), dim=-1)    
            score1 = score1.detach().cpu().numpy()
            score2 = torch.mean(self.anomaly_criterion(outputs2_not, input), dim=-1)    
            score2 = score2.detach().cpu().numpy()

            
            mask_score = score0 + score1 + score2  
            
            
         
            distances01d = torch.sqrt(((contra0 - contra1) ** 2).sum(dim=-1)) 
            distances01 = distances01d.detach().cpu().numpy()

            distances02d = torch.sqrt(((contra0 - contra2) ** 2).sum(dim=-1))
            distances02 = distances02d.detach().cpu().numpy()

            distances12d = torch.sqrt(((contra1 - contra2) ** 2).sum(dim=-1))
            distances12 = distances12d.detach().cpu().numpy()


            distances = distances01 + distances02 + distances12 

            score = mask_score + distances

            attens_energy.append(score)   
         
     

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)       
        train_energy = np.array(attens_energy)

        # (2) find the threshold  
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            metric_sum = 0
            input = input_data.float().to(self.device)
            
   
            output0, output1, output2, outputs0_not, outputs1_not, outputs2_not, contra0, contra1, contra2, noise0, noise1, noise2 = self.model(input) 
           
            score0 = torch.mean(self.anomaly_criterion(outputs0_not, input), dim=-1)
            score0 = score0.detach().cpu().numpy()
            score1 = torch.mean(self.anomaly_criterion(outputs1_not, input), dim=-1)    
            score1 = score1.detach().cpu().numpy()
            score2 = torch.mean(self.anomaly_criterion(outputs2_not, input), dim=-1)   
            score2 = score2.detach().cpu().numpy()

            
            mask_score = score0 + score1 + score2 
            
            

            distances01d = torch.sqrt(((contra0 - contra1) ** 2).sum(dim=-1))  
            distances01 = distances01d.detach().cpu().numpy()

            distances02d = torch.sqrt(((contra0 - contra2) ** 2).sum(dim=-1))
            distances02 = distances02d.detach().cpu().numpy()

            distances12d = torch.sqrt(((contra1 - contra2) ** 2).sum(dim=-1))
            distances12 = distances12d.detach().cpu().numpy()



            distances = distances01 + distances02 + distances12 


            score = mask_score + distances
            

            attens_energy.append(score)   
                        

            
    
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)



        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
   
            
            output0, output1, output2, outputs0_not, outputs1_not, outputs2_not, contra0, contra1, contra2, noise0, noise1, noise2 = self.model(input) 
           
           
            score0 = torch.mean(self.anomaly_criterion(outputs0_not, input), dim=-1)
            score0 = score0.detach().cpu().numpy()
            score1 = torch.mean(self.anomaly_criterion(outputs1_not, input), dim=-1)   
            score1 = score1.detach().cpu().numpy()
            score2 = torch.mean(self.anomaly_criterion(outputs2_not, input), dim=-1)   
            score2 = score2.detach().cpu().numpy()
          
            mask_score = score0 + score1 + score2  
            

            distances01d = torch.sqrt(((contra0 - contra1) ** 2).sum(dim=-1)) 
            distances01 = distances01d.detach().cpu().numpy()

            distances02d = torch.sqrt(((contra0 - contra2) ** 2).sum(dim=-1))
            distances02 = distances02d.detach().cpu().numpy()

            distances12d = torch.sqrt(((contra1 - contra2) ** 2).sum(dim=-1))
            distances12 = distances12d.detach().cpu().numpy()


            distances = distances01 + distances02 + distances12 


            score = mask_score + distances
            attens_energy.append(score)   
            
            
          
            test_labels.append(labels)      
            
            
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        
        hostname = socket.gethostname()

        # Get current timestamp (seconds since Unix era)
        timestamp = int(time.time())

        # Obtain the PID (process identifier) of the process
        pid = os.getpid()
        # Build file name
        log_filename = f"{timestamp}.{hostname}.{pid}.log"
        if not os.path.exists(os.path.join('/root/TMino4/result/detection_64_ratio', self.dataset)):
            os.makedirs(os.path.join('/root/TMino4/result/detection_64_ratio', self.dataset), exist_ok=True)


        with open(os.path.join('/root/TMino4/result/detection_64_ratio', self.dataset, log_filename), 'a') as file:
            file.write("dataset: {}\n".format(self.data_path))
            file.write(str(self.config))

            for q in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085]:
                thresh = getThreshold(train_energy, test_energy, q)
                print('q: ', q)
                file.write("q: {}\n".format(q))
                file.write("thresh: {}\n".format(thresh))

                pred = (test_energy > thresh).astype(int)
                gt = test_labels.astype(int)
                matrix = [self.index]

                scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)           
                for key, value in scores_simple.items():
                    matrix.append(value)
                    print('{0:21} : {1:0.4f}'.format(key, value))
                    file.write("{0:21} : {1:0.4f}\n".format(key, value))


                pred = np.array(pred)
                gt = np.array(gt)


                from sklearn.metrics import precision_recall_fscore_support
                from sklearn.metrics import accuracy_score
                from sklearn.metrics import roc_auc_score

                accuracy = accuracy_score(gt, pred)
                precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
                auc = roc_auc_score(gt, test_energy)
                print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f},  AUC : {:0.4f} ".format(accuracy, precision, recall, f_score, auc),file=file)



                file.write(
                    "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f}\n".format(
                        accuracy, precision, recall, f_score, auc)
                )



        return accuracy, precision, recall, f_score, auc
