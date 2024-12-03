import torch
import torch.nn.functional as F
from models.timelags import *



def inst_CL_hard(z1, z2): 
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C     T-win_size
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B   
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]   
    logits = -F.log_softmax(logits, dim=-1)         
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def instance_contrastive_loss(z1, z2, zneg1, zneg2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    
   
    z_pos = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z_pos = z_pos.transpose(0, 1)  # T x 2B x C
    sim_pos = torch.matmul(z_pos, z_pos.transpose(1, 2))  # T x 2B x 2B
    
    
    sim_neg1 = torch.matmul(z1, zneg1.transpose(1, 2))  # T x B x B
    sim_neg2 = torch.matmul(z2, zneg2.transpose(1, 2))  # T x B x B
    
  
    logits_pos = torch.tril(sim_pos, diagonal=-1)
    logits_neg1 = sim_neg1.diagonal(dim1=-2, dim2=-1)
    logits_neg2 = sim_neg2.diagonal(dim1=-2, dim2=-1)
    
   
    loss_pos = -F.log_softmax(logits_pos, dim=-1).mean()
    loss_neg1 = -F.log_softmax(logits_neg1, dim=-1).mean()
    loss_neg2 = -F.log_softmax(logits_neg2, dim=-1).mean()
    
   
    loss = (loss_pos + (loss_neg1 + loss_neg2) / 2) / 2
    return loss


def hierarchical_contrastive_loss(z1, z2, zneg1, zneg2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, zneg1, zneg2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d



def inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    loss = torch.sum(logits[:,i]*soft_labels_L)
    loss += torch.sum(logits[:,B + i]*soft_labels_R)
    loss /= (2*B*T)
    return loss

def hier_CL_soft(z1, z2, soft_labels, tau_temp=2, lambda_=0.5, temporal_unit=0, 
                 soft_temporal=False, soft_instance=False, temporal_hierarchy=True):
    
    if soft_labels is not None:
        soft_labels = torch.tensor(soft_labels, device=z1.device)
        soft_labels_L, soft_labels_R = dup_matrix(soft_labels)      ## 
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d



def temporal_contrastive_loss(z1, z2, zneg1, zneg2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
   
    z_pos = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z_pos = z_pos.transpose(0, 1)  # T x 2B x C
    sim_pos = torch.matmul(z_pos, z_pos.transpose(1, 2))  # T x 2B x 2B
  
    sim_neg1 = torch.matmul(z1, zneg1.transpose(1, 2))  # T x B x B
    sim_neg2 = torch.matmul(z2, zneg2.transpose(1, 2))  # T x B x B

    logits_pos = torch.tril(sim_pos, diagonal=-1)
    logits_neg1 = sim_neg1.diagonal(dim1=-2, dim2=-1)
    logits_neg2 = sim_neg2.diagonal(dim1=-2, dim2=-1)

    loss_pos = -F.log_softmax(logits_pos, dim=-1).mean()
    loss_neg1 = -F.log_softmax(logits_neg1, dim=-1).mean()
    loss_neg2 = -F.log_softmax(logits_neg2, dim=-1).mean()
    loss = (loss_pos + (loss_neg1 + loss_neg2) / 2) / 2
    return loss
def hierarchical_contrastive_loss(z1, z2, zneg1, zneg2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2, zneg1, zneg2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    return loss / d