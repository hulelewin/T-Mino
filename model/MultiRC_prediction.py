import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .attn import AttentionLayer, FullAttention    
from .embed import DataEmbedding, TokenEmbedding, PatchEmbedding
from .RevIN import RevIN
import numpy as np
import math
from torch import Tensor
from .attention import *
from .basics import *
from typing import Callable, Optional
import sys, math, random, copy
from scipy.ndimage import gaussian_filter1d
from typing import Tuple
import scipy.stats
from scipy import interpolate
import numpy.fft as fft

def random_masking_old(xb, mask_ratio):
    bs, nvars, L, D = xb.shape    
    x = torch.movedim(xb, 1, 2)   
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, nvars,device=xb.device)  
    ids_shuffle = torch.argsort(noise, dim=1)  
    ids_restore = torch.argsort(ids_shuffle, dim=1)                  
   
    ids_keep = ids_shuffle[:, :len_keep, :]                                             
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))    

    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)                 
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)                                 
    mask[:, :len_keep, :] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)                                 
    return x_masked, x_kept, mask, ids_restore


def random_masking(xb, representative_periods):
    bs, L, C = xb.shape 
    mask_ranges = []
    for channel in range(C):
        period = representative_periods[channel].item()  

        if period <= 2:
            mask_range = 2 
        elif 2 < period <= 21:
            mask_range = 4  
        else:  
            mask_range = 8  
        
        mask_ranges.append(mask_range)


    x_masked_list = []
    x_kept_list = []
    masks_list = []

    for channel in range(C):
        mask_range = mask_ranges[channel]

        noise = torch.rand(bs, L, device=xb.device) 

        for b in range(bs):
      
            start_idx = torch.randint(0, L - mask_range + 1, (1,)).item()
            xb[b, start_idx:start_idx + mask_range, channel] = 0  
        x_masked_list.append(xb[:, :, channel].unsqueeze(-1))  

    x_masked = torch.cat(x_masked_list, dim=-1)  
    
    mask = (x_masked == 0).float()  

    return x_masked, mask


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1) 
    frequency_list = abs(xf)  
    frequency_list[:, 0, :] = 0  
    period_BKC = frequency_list.permute(2, 0, 1)  
    period_CKB = frequency_list.permute(2, 1, 0)  

    similarity_matrix = torch.matmul(period_BKC, period_CKB)  
    norm_BKC = torch.norm(period_BKC, p=2, dim=-1, keepdim=True)  
    norm_CKB = torch.norm(period_CKB, p=2, dim=-1, keepdim=True)  

    topk_similarities, topk_indices = torch.topk(similarity_matrix, k=k, dim=-1)

  
    T = x.shape[1]  
    periods = T // topk_indices.detach().cpu().numpy() 
    periods = torch.tensor(periods)  
    representative_periods = periods.max(dim=1).values  
    return representative_periods, topk_similarities


def shift(x, p=0.5, sigma=1.5):
    # print("p---------------:", p)
    device = x.device
    noise_mask = torch.rand(x.size(-1), device=device) < p
    noise = torch.randn(x.size(-1), device=device) * sigma
    x = x + noise_mask * noise
    return x


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


    
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn
    
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class TMino(nn.Module):
    def __init__(self, win_size, mask_ratio, noise_ratio, stride, enc_in, c_out, n_heads=1,
                 d_model=256, e_layers=3, patch_size=[2,4,8],                
                 channel=55, d_ff=512, dropout=0.0, activation='gelu', 
                 output_attention=True, mlp=False):  
               
        super(TMino, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size       
        self.channel = channel
        self.win_size = win_size
        self.mask_ratio = mask_ratio            
        self.noise_ratio = noise_ratio
        self.stride = stride
        padding = self.stride
        self.d_model = d_model

      
        for i, patchsize in enumerate(self.patch_size):
            if i==0:                
                self.embedding_patch_size0 = DataEmbedding(patchsize, d_model, dropout)
                self.embedding_patch_num0 = DataEmbedding(self.win_size//patchsize, d_model, dropout)
                self.projection0 = nn.Linear(d_model, patchsize, bias=True)     
            elif i==1:                  
                self.embedding_patch_size1 = DataEmbedding(patchsize, d_model, dropout)
                self.embedding_patch_num1 = DataEmbedding(self.win_size//patchsize, d_model, dropout)
                self.projection1 = nn.Linear(d_model, patchsize, bias=True)      
            else:
                self.embedding_patch_size2 = DataEmbedding(patchsize, d_model, dropout)
                self.embedding_patch_num2 = DataEmbedding(self.win_size//patchsize, d_model, dropout)
                self.projection2 = nn.Linear(d_model, patchsize, bias=True)      
                
   
      
        self.encoder0 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        
        self.encoderk0 = copy.deepcopy(self.encoder0)
        
        self.encoder1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        
        self.encoderk1 = copy.deepcopy(self.encoder1)
        
        self.encoder2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        self.encoderk2 = copy.deepcopy(self.encoder2)
      
        self.flatten0 = nn.Flatten(start_dim=-2)
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.flatten2 = nn.Flatten(start_dim=-2)
        
        # create the encoders
        self.head_q = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.head_k = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        for param_q0, param_k0 in zip(self.encoder0.parameters(), self.encoderk0.parameters()):
            param_k0.data.copy_(param_q0.data)  # initialize
            param_k0.requires_grad = False  # not update by gradient
        for param_q0, param_k0 in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k0.data.copy_(param_q0.data)  # initialize
            param_k0.requires_grad = False  # not update by gradient
            
        for param_q1, param_k1 in zip(self.encoder1.parameters(), self.encoderk1.parameters()):
            param_k1.data.copy_(param_q1.data)  # initialize
            param_k1.requires_grad = False  # not update by gradient
        for param_q1, param_k1 in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k1.data.copy_(param_q1.data)  # initialize
            param_k1.requires_grad = False  # not update by gradient
            
        for param_q2, param_k2 in zip(self.encoder2.parameters(), self.encoderk2.parameters()):
            param_k2.data.copy_(param_q2.data)  # initialize
            param_k2.requires_grad = False  # not update by gradient
        for param_q2, param_k2 in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k2.data.copy_(param_q2.data)
            param_k2.requires_grad = False 
        
        
        if mlp:  
            dim_mlp = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.fc
            )

   
        
        self.register_buffer('queue', F.normalize(torch.randn(d_model, self.win_size), dim=0))     
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update for key encoder
        """
        self.m = 0.999
        for param_q0, param_k0 in zip(self.encoder0.parameters(), self.encoderk0.parameters()):
            param_k0.data = param_k0.data * self.m + param_q0.data * (1 - self.m)    
        for param_q0, param_k0 in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k0.data = param_k0.data * self.m + param_q0.data * (1 - self.m)
            
        for param_q1, param_k1 in zip(self.encoder1.parameters(), self.encoderk1.parameters()):
            param_k1.data = param_k1.data * self.m + param_q1.data * (1 - self.m)
        for param_q1, param_k1 in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k1.data = param_k1.data * self.m + param_q1.data * (1 - self.m)
            
        for param_q2, param_k2 in zip(self.encoder2.parameters(), self.encoderk2.parameters()):
            param_k2.data = param_k2.data * self.m + param_q2.data * (1 - self.m)
        for param_q2, param_k2 in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k2.data = param_k2.data * self.m + param_q2.data * (1 - self.m)


    def forward(self, x):
        B, L, M = x.shape 
        device = x.device
        revin_layer = RevIN(num_features=M, device=device)

        x = revin_layer(x, 'norm')


        representative_periods, topk_similarities = FFT_for_Period(x, k=1)   
        xb_mask, mask = random_masking(x, representative_periods)  

        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l') 
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l') 
           
            if patch_index==0:
              

                x_patch_size0 = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p=patchsize)  
                x_patch_size0 = rearrange(x_patch_size0, '(b m) n p -> b m n p', b = B, m = M) 
                enc_mask = x_patch_size0   
                x_patch_size_mask, x_patch_num_mask = xb_mask, xb_mask
                x_patch_size_mask = rearrange(x_patch_size_mask, 'b l m -> b m l')  
                x_patch_num_mask = rearrange(x_patch_num_mask, 'b l m -> b m l')  
                xb_mask0 = rearrange(x_patch_size_mask, 'b m (n p) -> (b m) n p', p=patchsize)


                last_size = int(enc_mask.shape[2] // 2)
             
                xb_notmask0 = torch.reshape(enc_mask, (-1, enc_mask.shape[-2], enc_mask.shape[-1])) 
                x_first_noise, x_last_noise0 = torch.split(enc_mask, last_size, dim=-2)       
            
                x_last_noise = shift(x_last_noise0, p=self.noise_ratio).to(device)

            
                
                x_first_noise = rearrange(x_first_noise, 'b m n p -> b n m p')
                x_last_noise = rearrange(x_last_noise, 'b m n p -> b n m p')
               
                noise = torch.cat((x_first_noise, x_last_noise), dim=1) 
                
                
                x_noise = torch.reshape(noise, (-1, noise.shape[1], noise.shape[-1]))  
   
                x_patch_mask0 = self.embedding_patch_size0(xb_mask0)   
                rand_idx0 = np.random.randint(0, x_patch_mask0.shape[1])
                series0, attns = self.encoder0(x_patch_mask0)  
                window_size = self.win_size  
                transposed0 = series0.transpose(1, 2)  
                upsampled0 = F.interpolate(transposed0, size=window_size, mode='nearest')
                contra0 = upsampled0.transpose(1, 2)  

                contra0 = torch.mean(contra0, dim=-1)  
                contra0 = rearrange(contra0, '(b m) w -> b w m', m=M)


                x_patch_notmask0 = self.embedding_patch_size0(xb_notmask0)   
                series0_notmask, attns = self.encoder0(x_patch_notmask0) 
           
                noise0 = self.embedding_patch_size0(x_noise)  
                noise0_contras, attns = self.encoder0(noise0)                
        
                noise_transposed0 = noise0_contras.transpose(1, 2) 
                noise_upsampled0 = F.interpolate(noise_transposed0, size=window_size, mode='nearest')
                negative0 = noise_upsampled0.transpose(1, 2)

                negative0 = torch.mean(negative0, dim=-1)  
                negative0 = rearrange(negative0, '(b m) w -> b w m', m=M)
            
            
                dec_out0 = self.projection0(series0)  
                dec_1_0 = rearrange(dec_out0, '(b m) n p -> b (n p) m', b=B, p=patchsize)  
            
                dec_out0_notmask = self.projection0(series0_notmask)   
                dec_1_0_notmask = rearrange(dec_out0_notmask, '(b m) n p -> b (n p) m', b=B, p=patchsize)  
            
            elif patch_index==1:

                x_patch_size1 = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p=patchsize) 
                x_patch_size1 = rearrange(x_patch_size1, '(b m) n p -> b m n p', b = B, m = M) 
                enc_mask1 = x_patch_size1  
                x_patch_size_mask, x_patch_num_mask = xb_mask, xb_mask
                x_patch_size_mask = rearrange(x_patch_size_mask, 'b l m -> b m l')  
                x_patch_num_mask = rearrange(x_patch_num_mask, 'b l m -> b m l')  
                xb_mask1 = rearrange(x_patch_size_mask, 'b m (n p) -> (b m) n p', p=patchsize) 


                last_size = int(enc_mask1.shape[2] // 2)
              
                xb_notmask1 = torch.reshape(enc_mask1, (-1, enc_mask1.shape[-2], enc_mask1.shape[-1])) 
                x_first_noise, x_last_noise0 = torch.split(enc_mask1, last_size, dim=-2)     
                x_last_noise = shift(x_last_noise0, p=self.noise_ratio).to(device)

                
                x_first_noise = rearrange(x_first_noise, 'b m n p -> b n m p')
                x_last_noise = rearrange(x_last_noise, 'b m n p -> b n m p')
                noise = torch.cat((x_first_noise, x_last_noise ), dim=1)     
                x_noise = torch.reshape(noise, (-1, noise.shape[1], noise.shape[-1]))  
              
      
                x_patch_mask1 = self.embedding_patch_size1(xb_mask1) 
                rand_idx1 = np.random.randint(0, x_patch_mask1.shape[1])
                series1, attns = self.encoder1(x_patch_mask1)

             
                transposed1 = series1.transpose(1, 2) 
                upsampled1 = F.interpolate(transposed1, size=window_size, mode='nearest')
                contra1 = upsampled1.transpose(1, 2) 
                contra1 = torch.mean(contra1, dim=-1)  

                contra1 = rearrange(contra1, '(b m) w -> b w m', m=M)

                x_patch_notmask1 = self.embedding_patch_size1(xb_notmask1)    
                series1_notmask, attns = self.encoder1(x_patch_notmask1) 

                noise1 = self.embedding_patch_size1(x_noise)  
                noise1_contras, attns = self.encoder1(noise1)                   
                noise_transposed1 = noise1_contras.transpose(1, 2) 
                noise_upsampled1 = F.interpolate(noise_transposed1, size=window_size, mode='nearest')
                negative1 = noise_upsampled1.transpose(1, 2) 
                negative1 = torch.mean(negative1, dim=-1)  
                negative1 = rearrange(negative1, '(b m) w -> b w m', m=M)


                
                dec_out1 = self.projection1(series1)           
                dec_1_1 = rearrange(dec_out1, '(b m) n p -> b (n p) m', b=B, p=patchsize)  
        
                dec_out1_notmask = self.projection1(series1_notmask)   
                dec_1_1_notmask = rearrange(dec_out1_notmask, '(b m) n p -> b (n p) m', b=B, p=patchsize)  
        
   
                
                
            else:
              
                x_patch_size2 = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p=patchsize)  
                x_patch_size2 = rearrange(x_patch_size2, '(b m) n p -> b m n p', b = B, m = M) 
                enc_mask2 = x_patch_size2  
                x_patch_size_mask, x_patch_num_mask = xb_mask, xb_mask
                x_patch_size_mask = rearrange(x_patch_size_mask, 'b l m -> b m l')  
                x_patch_num_mask = rearrange(x_patch_num_mask, 'b l m -> b m l') 
                xb_mask2 = rearrange(x_patch_size_mask, 'b m (n p) -> (b m) n p', p=patchsize)   

                last_size = int(enc_mask2.shape[2] // 2)
    
                xb_notmask2 = torch.reshape(enc_mask2, (-1, enc_mask2.shape[-2], enc_mask2.shape[-1])) 

                x_first_noise, x_last_noise0 = torch.split(enc_mask2, last_size, dim=-2)
               
                x_last_noise = shift(x_last_noise0, p=self.noise_ratio).to(device)       

                x_first_noise = rearrange(x_first_noise, 'b m n p -> b n m p')
                x_last_noise = rearrange(x_last_noise, 'b m n p -> b n m p')
                noise = torch.cat((x_first_noise, x_last_noise ), dim=1)     

                x_noise = torch.reshape(noise, (-1, noise.shape[1], noise.shape[-1])) 
            
                x_patch_mask2 = self.embedding_patch_size2(xb_mask2) 
                rand_idx2 = np.random.randint(0, x_patch_mask2.shape[1])
                series2, attns = self.encoder2(x_patch_mask2)

        
                transposed2 = series2.transpose(1, 2)  
                upsampled2 = F.interpolate(transposed2, size=window_size, mode='nearest')
                contra2 = upsampled2.transpose(1, 2)
                contra2 = torch.mean(contra2, dim=-1)  

                contra2 = rearrange(contra2, '(b m) w -> b w m', m=M)

         
                x_patch_notmask2 = self.embedding_patch_size2(xb_notmask2)    
                series2_notmask, attns = self.encoder2(x_patch_notmask2) 
    

          
                noise2 = self.embedding_patch_size2(x_noise)  
                noise2_contras, attns = self.encoder2(noise2)                           
             
                noise_transposed2 = noise2_contras.transpose(1, 2)  
                noise_upsampled2 = F.interpolate(noise_transposed2, size=window_size, mode='nearest')
                negative2 = noise_upsampled2.transpose(1, 2)  
                negative2 = torch.mean(negative2, dim=-1)  
                negative2 = rearrange(negative2, '(b m) w -> b w m', m=M)

           
                dec_out2 = self.projection2(series2)            
                dec_1_2 = rearrange(dec_out2, '(b m) n p -> b (n p) m', b=B, p=patchsize)  
  
                
                dec_out2_notmask = self.projection2(series2_notmask)   
                dec_1_2_notmask = rearrange(dec_out2_notmask, '(b m) n p -> b (n p) m', b=B, p=patchsize)  

                
        if self.output_attention:
            dec_1_0 = revin_layer(dec_1_0, 'denorm')
            dec_1_1 = revin_layer(dec_1_1, 'denorm')
            dec_1_2 = revin_layer(dec_1_2, 'denorm')
            dec_1_0_notmask = revin_layer(dec_1_0_notmask, 'denorm')
            dec_1_1_notmask = revin_layer(dec_1_1_notmask, 'denorm')
            dec_1_2_notmask = revin_layer(dec_1_2_notmask, 'denorm')

            return dec_1_0, dec_1_1, dec_1_2, dec_1_0_notmask, dec_1_1_notmask, dec_1_2_notmask, contra0, contra1, contra2, negative0, negative1, negative2
        else:
            return None

 
