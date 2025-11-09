import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import models
import time
import torch

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), 'checkpoint/' + name)
        return name

    def forward(self, *input):
        pass

class ImgModule(BasicModule):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=3):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)
        #self.apply(weights_init)
        self.norm = norm
    def forward(self, x):
        out = self.fc(x).tanh()
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out
    
class TxtModule(BasicModule):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=2):
        super(TxtModule, self).__init__()
        self.module_name = "text_model"
        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)
        self.norm = norm

    def forward(self, x):
        out = self.fc(x).tanh()
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out



class MetaSimilarityImportanceAssignmentNetwork(nn.Module):
    """
    Meta Similarity Importance Assignment Network (MSIAN)
    Meta network for dynamically assigning sample weights
    """

    def __init__(self, input_dim, hidden_dim=None):
        super(MetaSimilarityImportanceAssignmentNetwork, self).__init__()
        # hidden_dim = input_dim // 2 if hidden_dim is None else hidden_dim
        hidden_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, similarity_features):
        if similarity_features.dim() > 2:
            similarity_features = similarity_features.view(similarity_features.size(0), -1)
       
        weights = self.mlp(similarity_features)
       
        return weights.view(-1)

class CombinedNet(nn.Module):
    def __init__(self, img_net, txt_net):
        super().__init__()
        self.img_net = img_net
        self.txt_net = txt_net

    def forward(self, images, tags, lengths=None):
        u = self.img_net(images)
       
        v = self.txt_net(tags) if lengths is None else self.txt_net(tags, lengths)
        return u, v
    
def cross_modal_contrastive_ctriterion_q(fea, tau=1.0, q=1.0, opt=None):
    """
    Cross-modal contrastive loss (in q-GCE form)
    fea: list of two feature tensors [B×bit, B×bit]
    tau: temperature parameter
    q:   q parameter
    opt: optional dict, internal fields not used
    """

    n_view    = 2
    batch_size = fea[0].shape[0]
   
    all_fea = torch.cat(fea, dim=0)                  
   
    sim = all_fea @ all_fea.t()                      
    sim = (sim / tau).exp()                          
    sim = sim - torch.diag(sim.diag())               
    
 
    sim_sum1 = sum(sim[:, v*batch_size:(v+1)*batch_size]
                   for v in range(n_view))        
    diag1 = torch.cat([
        sim_sum1[v*batch_size:(v+1)*batch_size].diag()
        for v in range(n_view)
    ])                                               
    p1 = diag1 / sim.sum(dim=1)                     
    loss1 = (1 - q) * (1.0 - p1**q).div(q) + q * (1 - p1)
    
    sim_sum2 = sum(sim[v*batch_size:(v+1)*batch_size, :]
                   for v in range(n_view))         
    diag2 = torch.cat([
        sim_sum2[:, v*batch_size:(v+1)*batch_size].diag()
        for v in range(n_view)
    ])                                               
    p2 = diag2 / sim.sum(dim=1)
    loss2 = (1 - q) * (1.0 - p2**q).div(q) + q * (1 - p2)
    
    return loss1.mean() + loss2.mean()

