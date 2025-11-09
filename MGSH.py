from utils.tools import *
from scipy.linalg import hadamard
from network import *
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')

def get_config(dataset, bit_len, noise_rate, Lambda,epoch,r,margin,tao):
   
    if dataset == 'flickr':
        train_size, n_class, tag_len = 9500, 24, 1386
        meta_data_path=r'/data1/tza/NRCH-master/meta/mirflickr25k-meta.h5'
    elif dataset == 'ms-coco':
        train_size, n_class, tag_len = 10000, 80, 300
        meta_data_path=r'/data1/tza/NRCH-master/meta/MS-COCO-meta.h5'
    elif dataset == 'nuswide10':
        train_size, n_class, tag_len = 10500, 10, 1000
        meta_data_path=r'/data1/tza/NRCH-master/meta/NUS-WIDE-meta.h5'
    elif dataset == 'iapr':
        train_size, n_class, tag_len = 10000, 255, 2912
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    config = {
        "optimizer":     {"type":  optim.Adam, "optim_params": {"lr": 1e-5, "weight_decay": 1e-5}},
        "txt_optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5, "weight_decay": 1e-5}},
        "info":          "[CSQ]",
        "resize_size":   256,
        "crop_size":     224,
        "batch_size":    128,
        "meta_batch_size": 64,
        "dataset":       dataset,
        "epoch":         epoch,
        "device":        torch.device("cuda:0"),
        "bit_len":       bit_len,
        "noise_type":    'symmetric',
        "noise_rate":    noise_rate,
        "random_state":  1,
        "n_class":       n_class,
        "lambda":        Lambda,
        "tag_len":       tag_len,
        "train_size":    train_size,
        "num_gradual":   epoch,
        "meta_data_path": meta_data_path,
        "meta_batch_size": 64,
        "meta_lr": 1e-5,
        "margin": margin,
        "tau": tao,
        "warmup":5,
        'threshold':0.3,
        "r":       r,
    }
    return config

class Robust_Loss(torch.nn.Module):
    def __init__(self, config, bit):
        super(Robust_Loss, self).__init__()
        self.shift  = 1.0
        self.margin = config.get("margin")
        self.tau    = config.get("tau", 1.0)
        
        K = config["n_class"]
        L = config["bit_len"]
        C = torch.randn(K, L).sign()
        C = F.normalize(C, p=2, dim=1)
        self.C = nn.Parameter(C)
        self.ca_tau = config.get("ca_tau", 1.0)
        self.ca_r   = config.get("r",     0.7)
        self.eta    = config.get("eta",   0.05)
        self.eps    = 1e-6

    def calc_neighbor(self, label1, label2):
        return (label1.float().mm(label2.float().t()) > 0).float()

    def forward(self, u, v, y, config,
                margins=None, return_vector=False, w=None):
       
        # 1) Activation & T matrix 
        u = u.tanh();  v = v.tanh()
        T = self.calc_neighbor(y, y);  T.diagonal().fill_(0)
        lam = config["lambda"]

        # 2) Construct cost matrix & loss_r
        S = u.mm(v.t());  B = S.size(0)
        d = S.diag().view(-1,1)
        d1, d2 = d.expand_as(S), d.expand_as(S).t()
        if margins is None:
            m_vec = self.margin
            m1 = m2 = self.margin
        else:
            m_vec = margins
            m1 = margins.view(-1,1).expand_as(S)
            m2 = margins.view(1,-1).expand_as(S).t()
        mask_te = (S>=d1-m1).float().detach()
        cost_te = S*mask_te + (1-mask_te)*(S-self.shift)
        mask_im = (S>=d2-m2).float().detach()
        cost_im = S*mask_im + (1-mask_im)*(S-self.shift)
        I_eye = torch.eye(B, device=S.device)
        cost_te_max = cost_te*(1-I_eye)+torch.diag_embed(cost_te.diag().clamp(min=0))
        cost_im_max = cost_im*(1-I_eye)+torch.diag_embed(cost_im.diag().clamp(min=0))
        term_te = (-cost_te.diag()
                   + self.tau*((cost_te_max/self.tau*(1-T)).exp().sum(1)).log()
                   + (m_vec if isinstance(m_vec,float) else m_vec))
        term_im = (-cost_im.diag()
                   + self.tau*((cost_im_max/self.tau*(1-T)).exp().sum(1)).log()
                   + (m_vec if isinstance(m_vec,float) else m_vec))
        loss_r = term_te + term_im  # [B]

        # 3) Quantization regularization
        d_feat = u.size(1)
        Q_u = (u.abs()-1/np.sqrt(d_feat)).pow(2).mean(1)
        Q_v = (v.abs()-1/np.sqrt(d_feat)).pow(2).mean(1)
        Q_loss = Q_u + Q_v         # [B]

        # 4) Center aggregation loss 
        loss_ca = self.cross_modal_center_aggregation(u, v, y)

        # 5) Total loss
        loss_vec = lam*loss_r + (1-lam)*(loss_ca + Q_loss)

        # 6) Update centers (use provided w if given; otherwise compute default weights internally)
        # self.update_centers(u, v, y, w)

        return loss_vec if return_vector else loss_vec.mean()

    def cross_modal_center_aggregation(self, u, v, y):
        C = self.C.to(u.device)
        B_u = u; B_v = v
        B = torch.cat([B_u, B_v], dim=0)
        B_u, B_v = B.chunk(2, dim=0)
        logits_u = B_u @ C.t().div(self.ca_tau)
        logits_v = B_v @ C.t().div(self.ca_tau)
        p_u = F.softmax(logits_u, dim=1)
        p_v = F.softmax(logits_v, dim=1)
        v_u = (y*p_u).sum(1)
        v_v = (y*p_v).sum(1)
        term_u = (1-self.ca_r)*(1-v_u.pow(self.ca_r))/self.ca_r + self.ca_r*(1-v_u)
        term_v = (1-self.ca_r)*(1-v_v.pow(self.ca_r))/self.ca_r + self.ca_r*(1-v_v)
        return (term_u+term_v).mean()

    def update_centers(self, u, v, y, w=None):
        """
        Dynamic center update: use given w if provided; otherwise compute weights internally based on features.
        u, v: [B, L], y: [B, K], w: [B] or None
        """
        B, L = u.size()
        
        if w is None:
            
            omega = torch.sigmoid(u.abs().mean(1)).to(u.device)
            omega = torch.cat([omega, torch.sigmoid(v.abs().mean(1)).to(u.device)], dim=0)
        else:
            
            omega = torch.cat([w, w], dim=0)  
        
        C_update = torch.zeros_like(self.C)
        for k in range(self.C.size(0)):
            mask = y[:,k].float().view(-1,1)           
            w_pair = omega.view(-1,1)                 
            W_u = mask * w_pair[:B]                    
            W_v = mask * w_pair[B:]                    
            num = (W_u*u + W_v*v).sum(0)              
            den = W_u.sum() + W_v.sum() + self.eps
            C_update[k] = num.div(den)
       
        newC = (1-self.eta)*self.C.data + self.eta*C_update
        self.C.data = F.normalize(newC, p=2, dim=1)

def split_prob(prob, threshld):
    pred = (prob >= threshld)
    return (pred+0)

def get_loss(net, txt_net, config, data_loader, epoch, W):
    """
    Returns:
    sorted_losses: np.ndarray, shape (N, 1)
    pred: torch.Tensor, shape (N,) — 0/1 mask, where 1 indicates clean samples
    precision: float
    """

    device = next(net.parameters()).device
    sample_losses = []
    labels_clean = []
    # 1) Iterate over all samples, computing loss and true clean labels.
    for image, tag, tlabel, label, ind in data_loader:
        image = image.to(device).float()
        tag   = tag.to(device).float()
        label = label.to(device)
        tlabel= tlabel.to(device)

        u = net(image)
        v = txt_net(tag)

        with torch.no_grad():
           
            label_signed = (label - 0.5) * 2    # [B, C]
           
            u_sims = u @ W.tanh().t()           # [B, C]
            v_sims = v @ W.tanh().t()           # [B, C]
           
            per_elem = (label_signed - u_sims)**2 + (label_signed - v_sims)**2
            loss_vec = (per_elem * label).max(dim=1)[0].cpu().numpy()  # [B]
          
            right_mask = ((tlabel == label).float().mean(dim=1) == 1).cpu().numpy()

        for lid, lval, r in zip(ind.cpu().numpy(), loss_vec, right_mask):
            sample_losses.append((lid, lval, r))

    # 2) Sort in order of the global index.
    sample_losses.sort(key=lambda x: x[0])
    losses      = np.array([x[1] for x in sample_losses], dtype=np.float32)  # [N]
    labels_clean= np.array([x[2] for x in sample_losses], dtype=np.float32)  # [N]

    # 3) Normalize to [0,1] and reshape to (N,1)
    mn, mx = losses.min(), losses.max()
    norm   = (losses - mn + 1e-8) / (mx - mn + 1e-8)
    sorted_losses = norm.reshape(-1, 1)

    # 4) Compute dynamic forgetting rate by epoch
    noise_rate  = config["noise_rate"]
    num_gradual = config["num_gradual"]
    if epoch < num_gradual:
        forget_rate = noise_rate * (epoch + 1) / num_gradual
    else:
        forget_rate = noise_rate
    remember_rate = 1 - forget_rate

    # 5) Select samples with the smallest losses based on the remember_rate
    N = len(norm)
    num_remember = int(remember_rate * N)
    sorted_idx   = np.argsort(norm)            
    clean_idx    = sorted_idx[:num_remember]   
    pred         = np.zeros(N, dtype=np.float32)
    pred[clean_idx] = 1.0
    pred = torch.from_numpy(pred)

    # 6) Calculate precision
    clean_set = set(np.where(labels_clean == 1)[0])
    pred_set  = set(np.where(pred.numpy() == 1)[0])
    tp = len(clean_set & pred_set)
    fp = len(pred_set - clean_set)
    precision = tp / (tp + fp + 1e-12)

    return sorted_losses, pred, precision

def l2m(img_net, txt_net, meta_net,
         opt_main, opt_meta,
         meta_loader, meta_iter,
         images, tags, labels,
         W, config):
    """
    L2m meta-learning using Hamming distance to replace dot product:
        1) Update the main network (opt_main) using clean meta-data
        2) Compute the cleanliness weight w_det and adaptive margin γ̂ for the current training batch
        3) Update the meta network (opt_meta) using w_det from the training batch

    Returns:
        w_meta: Weights of training samples output by the updated meta network, shape=[B]
        gamma_hat: Adaptive margin vector, shape=[B]
        meta_iter: Updated iterator
    """

    device = next(img_net.parameters()).device
    eps = 1e-8

    # 1) Update the main network using meta-data

    try:
        meta_imgs, meta_tags, _, _, _ = next(meta_iter)
    except StopIteration:
        meta_iter = iter(meta_loader)
        meta_imgs, meta_tags, _, _, _ = next(meta_iter)
    meta_imgs = meta_imgs.to(device)
    meta_tags = meta_tags.to(device)

    img_net.train(); txt_net.train()
    u_meta = img_net(meta_imgs)                 # [B_meta, d]
    v_meta = txt_net(meta_tags)                 # [B_meta, d]
    labels_meta = torch.ones(u_meta.size(0), config['n_class'], device=device)
    fixed_margin = torch.full((u_meta.size(0),), config['margin'], device=device)
    loss_main_meta = Robust_Loss(config, config['bit_len'])(
        u_meta, v_meta,
        labels_meta, config,
        margins=fixed_margin,
        return_vector=False
    )
    opt_main.zero_grad()
    loss_main_meta.backward()
    opt_main.step()

    # 2) Compute w_det (cleanliness) and γ̂ for the training batch
    img_net.eval(); txt_net.eval()
    with torch.no_grad():
        u_train_nd = img_net(images)             # [B_train, d]
        v_train_nd = txt_net(tags)               # [B_train, d]
        B_u = (u_train_nd > 0).float()           # [B, d]
        B_v = (v_train_nd > 0).float()           # [B, d]
        pos_dist = CalcHammingDist(B_u, B_v).diag()  # [B]
        idx_perm = torch.randperm(B_u.size(0), device=device)
        B_v_neg = (txt_net(tags[idx_perm]) > 0).float()
        neg_dist = CalcHammingDist(B_u, B_v_neg).diag()  # [B]
    temp = config.get('temperature', 1.0)
    w_det = torch.sigmoid((neg_dist - pos_dist) / (temp + eps))  # [B]

    m = config['margin']; tau = config.get('tau', 1.0)
    gamma_hat = m / (1 + ((1.0/(w_det + eps) - 1) ** tau))       # [B]

    # 3) Update the meta network using w_det from the training batch
    meta_net.train()
    s_train = (u_train_nd * v_train_nd)       # [B, d]

    w_meta = meta_net(s_train)               # [B]
    loss_meta = F.binary_cross_entropy(w_meta, w_det.detach())
    opt_meta.zero_grad()
    loss_meta.backward()
    opt_meta.step()
    w_meta = meta_net(s_train)

    return w_meta.detach(), gamma_hat.detach(), meta_iter

def train_with_meta(config, bit,save_path, model_path):
    """
    Integrate l2b's Meta-Guided cross-modal hashing training, supporting warmup pre-training.
    config should include:
    - "warmup": number of warmup epochs
    - "threshold": fixed threshold for GMM binarization
    - "dataset", "noise_rate": used for visualization file naming
    """

    device    = config["device"]
    tag_len   = config["tag_len"]
    n_class   = config["n_class"]
    bit_len   = config["bit_len"]
    warmup    = config.get("warmup", 0)
    T = config["epoch"]

    # ---------- Data Loading ----------
    train_loader, eval_loader, test_loader, database_loader, meta_loader, \
    num_train, num_test, num_database, num_meta = get_data(config)
    config["num_train"], config["num_meta"] = num_train, num_meta


    meta_iter = iter(meta_loader)

    # ---------- Model Initialization ----------
    img_net  = ImgModule(y_dim=4096, bit=bit, hiden_layer=3).to(device)
    txt_net  = TxtModule(y_dim=tag_len,  bit=bit, hiden_layer=2).to(device)
    W_param  = torch.nn.Parameter(torch.empty(n_class, bit_len, device=device))
    torch.nn.init.orthogonal_(W_param, gain=1)
    img_net.register_parameter('W', W_param)
    meta_net = MetaSimilarityImportanceAssignmentNetwork(input_dim=bit).to(device)

    # ---------- Optimizer and Loss ----------
    opt_main  = config["optimizer"]["type"](
        list(img_net.parameters()) + list(txt_net.parameters()),
        **config["optimizer"]["optim_params"]
    )
    opt_meta  = torch.optim.Adam(meta_net.parameters(), lr=config["meta_lr"])


    criterion = Robust_Loss(config, bit)

    # ---------- Number of Training Epochs & Checkpoint ----------
    best_sum = 0.0
    eps      = 1e-8
    os.makedirs('./checkpoint', exist_ok=True)

    epoch_list, i2t_list, t2i_list = [], [], []

    for epoch in range(T):
        img_net.train(); txt_net.train()
        w_loss_batches = []
       
        sorted_losses, pred_mask, precision = get_loss(
            img_net, txt_net, config, eval_loader,
            epoch, W_param
        )

        epoch_rob_loss, epoch_q_loss = 0.0, 0.0
        warmup_batches, valid_batches = 0, 0
        w_means, gamma_means = [], []

        for imgs, tags, _, labels, idx in train_loader:
            imgs, tags, labels = imgs.to(device), tags.to(device), labels.to(device)
        
            if epoch < warmup:
                batch_size = imgs.size(0)
                loss = criterion(
                    img_net(imgs), txt_net(tags),
                    labels.float(), config,
                    margins=None,
                    return_vector=False,
                    w=None
                )
                opt_main.zero_grad(); loss.backward(); opt_main.step()
                epoch_rob_loss  += loss.item()
                warmup_batches  += 1
            else:
                keep = pred_mask[idx.cpu().numpy()] == 1
                if keep.sum() == 0:
                    continue
                imgs, tags, labels = imgs[keep].to(device), tags[keep].to(device), labels[keep].to(device)
                idx = idx[keep]

                # 1) Compute meta weights w and adaptive margin γ̂
                w, gamma_hat, meta_iter = l2m(
                    img_net, txt_net, meta_net,
                    opt_main, opt_meta,
                    meta_loader, meta_iter,
                    imgs, tags, labels,
                    W_param, config
                )

                w_means.append(w.mean().item())
                gamma_means.append(gamma_hat.mean().item())

                # 2) Compute the robust loss vector for each sample
                loss_vec = criterion(
                    img_net(imgs), txt_net(tags),
                    labels.float(), config,
                    margins=None,
                    # margins=gamma_hat.detach(),
                    return_vector=True,
                    w=w.detach()
                )
                w_loss_batches.append((
                    w.cpu().detach(), 
                    loss_vec.cpu().detach()
                ))
                # 3) Compute the category consistency loss vector for each sample
                u, v     = img_net(imgs), txt_net(tags)
                y_sgn    = (labels - 0.5) * 2  # {-1, +1}
                u_sim    = u @ W_param.tanh().T
                v_sim    = v @ W_param.tanh().T
                q_loss_vec = ((y_sgn - u_sim)**2 + (y_sgn - v_sim)**2).mean(dim=1)

                w_adjusted = w
                total_vec     = loss_vec      # [batch]
                weighted_loss = (w_adjusted * total_vec).sum() / (w_adjusted.sum() + eps)

                opt_main.zero_grad()
                weighted_loss.backward()
                opt_main.step()

                epoch_rob_loss += loss_vec.mean().item()
                epoch_q_loss   += q_loss_vec.mean().item()
                valid_batches  += 1
 
        if epoch < warmup:
            avg_warm = epoch_rob_loss/ warmup_batches if warmup_batches else 0.0
            print(f"[Epoch {epoch+1}/{T}] warmup_loss={avg_warm:.4f}")
        else:
            avg_L_rob  = epoch_rob_loss / valid_batches if valid_batches else 0.0
            avg_L_q    = epoch_q_loss   / valid_batches if valid_batches else 0.0
            print(
                f"[Epoch {epoch+1}/{T}] "
                f"precision={precision:.4f}  "
                f"avg_w={np.mean(w_means):.4f}  "
                f"avg_gamma={np.mean(gamma_means):.4f}  "
                f"avg_L_rob={avg_L_rob:.4f}  "
                f"avg_L_q={avg_L_q:.4f}"
                f"noise_rate={config['noise_rate']:.1f}"
            )

        # —— Periodically evaluate mAP and save ——
        if (epoch + 1) % 5 == 0:
            img_net.eval(); txt_net.eval()
            with torch.no_grad():
                img_tb, img_tl = compute_img_result(test_loader,    img_net, device=device)
                img_db, img_dl = compute_img_result(database_loader, img_net, device=device)
                txt_tb, txt_tl = compute_tag_result(test_loader,    txt_net, device=device)
                txt_db, txt_dl = compute_tag_result(database_loader, txt_net, device=device)

            t2i = calc_map_k(img_db.numpy(),  txt_tb.numpy(), img_dl.numpy(), txt_tl.numpy(),device=config["device"])
            i2t = calc_map_k(txt_db.numpy(),  img_tb.numpy(), txt_dl.numpy(), img_tl.numpy(),device=config["device"])
            epoch_list.append(epoch+1)
            i2t_list.append(i2t)
            t2i_list.append(t2i)
            t2i_val = t2i
            i2t_val = i2t
            score   = t2i_val + i2t_val

            if  (t2i + i2t) > best_sum:
                best_sum = t2i + i2t
                torch.save({
                    'net_state_dict':     img_net.state_dict(),
                    'txt_net_state_dict': txt_net.state_dict(),
                    'meta_net':           meta_net.state_dict(),
                }, model_path)
                print(f"Save the best model: t2i={t2i_val:.3f}, i2t={i2t_val:.3f}, sum={score:.3f}")
            print(f"   eval: t2i={t2i:.3f}, i2t={i2t:.3f}")
            
            if w_loss_batches:
                
                all_w    = np.concatenate([w.numpy() for w, _ in w_loss_batches])
                all_loss = np.concatenate([l.numpy() for _, l in w_loss_batches])

                os.makedirs('figure', exist_ok=True)
                # # 1) MetaWeight distribution histogram
                # plt.figure(figsize=(6, 4))
                # plt.hist(all_w, bins=20)
                # plt.xlabel('MetaWeight (w)')
                # plt.ylabel('Number of samples')
                # plt.title(f'MetaWeight distribution of Epoch {epoch+1}')
                # plt.grid(linestyle='--', alpha=0.3)
                # hist_path = f'figure/epoch_{epoch+1}_w_distribution.png'
                # plt.tight_layout()
                # plt.savefig(hist_path, dpi=400)
                # plt.close()
                # print(f"[Epoch {epoch+1}] MetaWeight distribution saved → {hist_path}")

                # 2) MetaWeight vs Loss Heatmap Scatter Plot
                # plot_metaweight_vs_loss(
                #     all_w,
                #     all_loss,
                #     epoch=epoch,
                # )

    print("Training finished.")
     
def test(config, bit, model_path):
    device   = config["device"]
    tag_len  = config["tag_len"]
    n_class  = config["n_class"]
    bit_len  = config["bit_len"]
    _, _, test_loader, dataset_loader, *rest = get_data(config)

    # —— Model Loading ——
    net = ImgModule(y_dim=4096, bit=bit, hiden_layer=3).to(device)
    txt_net = TxtModule(y_dim=tag_len, bit=bit, hiden_layer=2).to(device)
    W = torch.nn.Parameter(torch.nn.init.orthogonal_(torch.empty(n_class, bit_len, device=device), gain=1))
    net.register_parameter('W', W)

    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    txt_net.load_state_dict(checkpoint['txt_net_state_dict'])
    net.eval(); txt_net.eval()

    # —— Extract Binary Codes ——
    img_tst_binary, img_tst_label = compute_img_result(test_loader, net, device=device)
    img_trn_binary, img_trn_label = compute_img_result(dataset_loader, net, device=device)
    txt_tst_binary, txt_tst_label = compute_tag_result(test_loader, txt_net, device=device)
    txt_trn_binary, txt_trn_label = compute_tag_result(dataset_loader, txt_net, device=device)

    # —— Calculate mAP ——
    print("calculating mAP...")
    t2i_mAP = calc_map_k(img_trn_binary.numpy(), txt_tst_binary.numpy(),
                         img_trn_label.numpy(), txt_tst_label.numpy())
    i2t_mAP = calc_map_k(txt_trn_binary.numpy(), img_tst_binary.numpy(),
                         txt_trn_label.numpy(), img_tst_label.numpy())
    print(f"Test Results: t2i_mAP: {t2i_mAP:.3f}, i2t_mAP: {i2t_mAP:.3f}")

    # —— Calculate PR Curve ——
    R_t2i, P_t2i = pr_curve1(
        rB=img_trn_binary.numpy(), qB=txt_tst_binary.numpy(),
        retrieval_L=img_trn_label.numpy(), query_L=txt_tst_label.numpy()
    )
    R_i2t, P_i2t = pr_curve1(
        rB=txt_trn_binary.numpy(), qB=img_tst_binary.numpy(),
        retrieval_L=txt_trn_label.numpy(), query_L=img_tst_label.numpy()
    )


    R_t2i_arr = np.array(R_t2i)
    P_t2i_arr = np.array(P_t2i)
    R_i2t_arr = np.array(R_i2t)
    P_i2t_arr = np.array(P_i2t)

   
    print("\n--- T2I PR curve data ---")
    print("\n--- T2I PR curve Recall ---")
    for r in R_t2i_arr:
        print(f"Recall={r:.4f}")

    print("\n--- T2I PR curve Precision ---")
    for p in P_t2i_arr:
        print(f"Precision={p:.4f}")

    print("\n--- I2T PR curve Recall ---")
    for r in R_i2t_arr:
        print(f"Recall={r:.4f}")

    print("\n--- I2T PR curve Precision ---")
    for p in P_i2t_arr:
        print(f"Precision={p:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CSQ training and testing')
    parser.add_argument('--gpus',        type=str,   default='0')
    parser.add_argument('--hash_dim',    type=int,   default=128, help='Hash code length')
    parser.add_argument('--epoch',    type=int,   default=60, help='Epochs for training')
    parser.add_argument('--noise_rate',  type=float, default=0.5, help='Noise rate')
    parser.add_argument('--dataset',     type=str,   default='flickr',
                        choices=['flickr','nuswide10','ms-coco'], help='Dataset name')
    parser.add_argument('--Lambda',      type=float, default=0.5, help='Lambda weight in the loss function')
    parser.add_argument('--r',      type=float, default=0.7, help='Weight in the category center clustering loss')
    parser.add_argument('--margin',      type=float, default=0.5, help='Initial distance')
    parser.add_argument('--tao',      type=float, default=0.5)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    
    config = get_config(
        dataset    = args.dataset,
        bit_len    = args.hash_dim,
        noise_rate = args.noise_rate,
        Lambda     = args.Lambda,
        epoch=args.epoch,
        r=args.r,
        margin=args.margin,
        tao=args.tao
    )
    save_path = f"map_curve_{config['dataset']}_noise{config['noise_rate']}.png"
    model_path = f"./checkpoint/best_model_{config['dataset']}_noise{config['noise_rate']:.2f}_{args.hash_dim}.pth"

    print("Current configuration:", config)
    train_with_meta(config, args.hash_dim,save_path,model_path)
    print("Current configuration:", config)
    test(config,  args.hash_dim,model_path)

