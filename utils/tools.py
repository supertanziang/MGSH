import numpy as np
import h5py
import pdb 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import seaborn as sns
class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

def get_clean_and_noisy_index(dataset,noise_rate):
    if dataset == 'nuswide10':
        noise = h5py.File('./noise/nus-wide-tc10-lall-noise_{}.h5'.format(noise_rate))
    elif dataset == 'flickr':
        noise = h5py.File('./noise/mirflickr25k-lall-noise_{}.h5'.format(noise_rate))
    elif dataset == 'ms-coco':
        noise = h5py.File('./noiseMSCOCO-lall-noise_{}.h5'.format(noise_rate))
    elif dataset == 'iapr':
        noise = h5py.File('./noiseIAPR-lall-noise_{}.h5'.format(noise_rate))
    fl = list(noise['True'])
    ffl = list(noise['noisy'])
    clean_index = []
    noisy_index = []

    for i in range(len(fl)):
        equal = True
        #pdb.set_trace()
        for j in range(len(fl[i])):
            if fl[i][j] != ffl[i][j]:
                equal=False
        if equal:
            clean_index.append(i)
            
        else:
            noisy_index.append(i)

    #pdb.set_trace()
    return clean_index, noisy_index

class DataList(Dataset):
    """
        Unified loading of train/test/database/meta data for MS-COCO, NUS-WIDE10, and MIRFlickr,
        compatible with three noise script output formats:
        - MIRFlickr: noise H5 contains 'clean' (train_clean matrix) & 'noisy' (noisy matrix)
        - NUS-WIDE10: noise H5 contains 'noisy' (noisy_matrix) & 'True' (clean_matrix)
        - MS-COCO: noise H5 contains 'noisy' (sample indices), 'noisy_labels', 'true_labels'
    """

    def __init__(self,
                 dataset: str,
                 data_type: str,      # 'train','test','database','meta'
                 transform,
                 noise_type,
                 noise_rate: float,
                 random_state: int,
                 meta_path: str = None):
        self.data_type    = data_type
        self.transform    = transform
        self.noise_type   = noise_type
        self.noise_rate   = noise_rate
        self.random_state = random_state

        # 1) Meta split 
        if data_type == 'meta':
            if meta_path is None:
                raise ValueError("meta_path must be provided for data_type='meta'")
            f_meta = h5py.File(meta_path, 'r')
            self.imgs = np.array(f_meta['ImgMeta'], dtype=np.float32)
            self.tags = np.array(f_meta['TagMeta'], dtype=np.float32)
            self.labs = np.array(f_meta['LabMeta'], dtype=int)
            self.tlabs = self.labs.copy()
            f_meta.close()
            return

        # 2) Open original data H5 and noise H5 files
        if dataset == 'nuswide10':
            data_f  = h5py.File('./data/NUS-WIDE.h5', 'r')
            noise_f = h5py.File(f'./noise/nus-wide-tc10-lall-noise_{noise_rate:.1f}.h5', 'r')
        elif dataset == 'flickr':
            data_f  = h5py.File('./data/MIRFlickr.h5', 'r')
            noise_f = h5py.File(f'./noise/mirflickr25k-lall-noise_{noise_rate:.1f}.h5', 'r')
        elif dataset == 'ms-coco':
            data_f  = h5py.File('./data/MS-COCO_rand_combined.h5', 'r')
            noise_f = h5py.File(f'./noise/ms-coco-lall-noise_{noise_rate:.1f}.h5', 'r')
        else:
            raise ValueError(f"Unknown dataset {dataset}")

        # 3) Load features 
        if data_type == 'train':
            imgs_all = np.array(data_f['ImgTrain'],    dtype=np.float32)
            tags_all = np.array(data_f['TagTrain'],    dtype=np.float32)
            clean_labels = np.array(data_f['LabTrain'],dtype=int)
        elif data_type == 'test':
            imgs_all = np.array(data_f['ImgQuery'],    dtype=np.float32)
            tags_all = np.array(data_f['TagQuery'],    dtype=np.float32)
            clean_labels = np.array(data_f['LabQuery'],dtype=int)
        elif data_type == 'database':
            imgs_all = np.array(data_f['ImgDataBase'], dtype=np.float32)
            tags_all = np.array(data_f['TagDataBase'], dtype=np.float32)
            clean_labels = np.array(data_f['LabDataBase'],dtype=int)
        else:
            raise ValueError(f"Unsupported data_type {data_type}")

        # Default: all samples with clean labels
        self.imgs = imgs_all
        self.tags = tags_all
        self.labs = clean_labels.copy()   # Training labels
        self.tlabs= clean_labels.copy()   # Evaluation (true) labels

        # 4) Replace labels only when using the train split and noise is specified
        if data_type == 'train' and noise_type is not None and noise_rate > 0:
           
            if 'clean' in noise_f and 'noisy' in noise_f:
                clean_mat = noise_f['clean'][:]   # shape=(N_train, C)
                noisy_mat = noise_f['noisy'][:]   # shape=(N_train, C)
                L = clean_mat.shape[0]
               
                self.imgs = imgs_all[:L]
                self.tags = tags_all[:L]
                self.labs = noisy_mat
                self.tlabs= clean_mat

            # MS-COCO script output: 'noisy' (indices), 'noisy_labels', 'true_labels'
            elif 'noisy_labels' in noise_f and 'true_labels' in noise_f:
                noisy_mat = noise_f['noisy_labels'][:]
                true_mat  = noise_f['true_labels'][:]
               
                self.labs  = noisy_mat
                self.tlabs = true_mat

            # NUS-WIDE script output old format: 'noisy' & 'True' matrices
            elif 'noisy' in noise_f and 'True' in noise_f \
                 and noise_f['noisy'].ndim == 2:
                noisy_mat = noise_f['noisy'][:]
                true_mat  = noise_f['True'][:]
                self.labs  = noisy_mat
                self.tlabs = true_mat

            else:
                # No matrices; treat as index filters.
                if 'noisy' in noise_f and noise_f['noisy'].ndim == 1:
                    bad_idx = noise_f['noisy'][:].astype(int)
                    keep_idx = np.setdiff1d(np.arange(len(imgs_all)), bad_idx)
                    self.imgs = imgs_all[keep_idx]
                    self.tags = tags_all[keep_idx]
                    self.labs = clean_labels[keep_idx]
                    self.tlabs= clean_labels[keep_idx]

        data_f.close()
        noise_f.close()

    def __getitem__(self, index):
        img = self.imgs[index]
        tag = self.tags[index]
        img_tensor = torch.from_numpy(img) if img.ndim == 1 else self.transform(img)
        tag_tensor = torch.from_numpy(tag)
       
        return img_tensor, tag_tensor, self.tlabs[index], self.labs[index], index

    def __len__(self):
        return len(self.imgs)


def SaveH5File_F(resize_size):
    train_size = 9500
    query_size = 1900
    root = './data/'
    fi = sio.loadmat(os.path.join(root, 'mirflickr25k-iall-vgg.mat'))['XAll']
    fl = sio.loadmat(os.path.join(root, 'mirflickr25k-lall.mat'))['LAll']
    ft = sio.loadmat(os.path.join(root, 'mirflickr25k-yall.mat'))['YAll']
    imgs = list(fi[query_size: query_size + train_size])
    labs = list(fl[query_size: query_size + train_size])
    tags = list(ft[query_size: query_size + train_size])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,24])
    Tag = np.zeros([n,1386])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf = h5py.File('./data/MIRFlickr.h5','w')
    hf.create_dataset('ImgTrain', data = Img)
    hf.create_dataset('TagTrain', data = Tag)
    hf.create_dataset('LabTrain', data = Lab)

    imgs = list(fi[0: query_size])
    labs = list(fl[0: query_size])
    tags = list(ft[0: query_size])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,24])
    Tag = np.zeros([n,1386])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgQuery', data = Img)
    hf.create_dataset('TagQuery', data = Tag)
    hf.create_dataset('LabQuery', data = Lab)

    imgs = list(fi[query_size::])
    labs = list(fl[query_size::])
    tags = list(ft[query_size::])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,24])
    Tag = np.zeros([n,1386])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgDataBase', data = Img)
    hf.create_dataset('TagDataBase', data = Tag)
    hf.create_dataset('LabDataBase', data = Lab)
    hf.close()


def SaveH5File_C(resize_size):
    # Split parameters
    train_size = 10000
    query_size = 5000

    # Path configuration 
    root       = '/data1/tza/NRCH-master/data'
    in_path    = os.path.join(root, 'MSCOCO_deep_doc2vec_data_rand.h5py')
    out_path   = os.path.join(root, 'MS-COCO_rand_combined.h5')

    # Load original data
    with h5py.File(in_path, 'r') as f_in:
        # LAll: (N, C_label), XAll: (N, 4096), YAll: (N, C_tag)
        LAll = f_in['LAll'][:]  
        XAll = f_in['XAll'][:]  
        YAll = f_in['YAll'][:]  

    # Verify total number of samples
    N = XAll.shape[0]
    assert LAll.shape[0] == N and YAll.shape[0] == N, "三个数组样本数不一致！"

    # Split subset
    ImgQuery    = XAll[0:query_size]
    LabQuery    = LAll[0:query_size]
    TagQuery    = YAll[0:query_size]

    start_trn   = query_size
    end_trn     = query_size + train_size
    ImgTrain    = XAll[start_trn:end_trn]
    LabTrain    = LAll[start_trn:end_trn]
    TagTrain    = YAll[start_trn:end_trn]
 
    ImgDataBase = XAll[end_trn:]
    LabDataBase = LAll[end_trn:]
    TagDataBase = YAll[end_trn:]

    with h5py.File(out_path, 'w') as f_out:
       
        f_out.create_dataset('ImgTrain',    data=ImgTrain)
        f_out.create_dataset('LabTrain',    data=LabTrain)
        f_out.create_dataset('TagTrain',    data=TagTrain)
  
        f_out.create_dataset('ImgQuery',    data=ImgQuery)
        f_out.create_dataset('LabQuery',    data=LabQuery)
        f_out.create_dataset('TagQuery',    data=TagQuery)
   
        f_out.create_dataset('ImgDataBase', data=ImgDataBase)
        f_out.create_dataset('LabDataBase', data=LabDataBase)
        f_out.create_dataset('TagDataBase', data=TagDataBase)

    print(f"Merged file created: {out_path}")  

def load_mat_data(path, key):
    try:
        with h5py.File(path, 'r') as f:
            return f[key][:]
    except (OSError, IOError):
        return sio.loadmat(path)[key]
      
def SaveH5File_N(resize_size):
    train_size = 10500
    query_size = 2100
    root = './data/'

    fi = load_mat_data(root + 'nus-wide-tc10-xall-vgg-clean.mat', 'IAll')
    fl = load_mat_data(root + 'nus-wide-tc10-lall.mat',           'LAll')
    if fl.shape[0] != fi.shape[0]:
        fl = fl.T
    ft = load_mat_data(root + 'nus-wide-tc10-yall.mat',           'YAll')
    if ft.shape[0] != fi.shape[0]:
        ft = ft.T

    def write_split(hf, name, data_list, dim1, dim2):
        n = len(data_list)
        D = np.zeros([n, dim2])
        for i in tqdm(range(n), desc=name):
            arr = np.asarray(data_list[i])
            D[i, :] = arr
        hf.create_dataset(name, data=D)

    hf_out = h5py.File(root + 'NUS-WIDE.h5', 'w')

    # Train
    imgs = fi[query_size: query_size + train_size]
    labs = fl[query_size: query_size + train_size]
    tags = ft[query_size: query_size + train_size]
    write_split(hf_out, 'ImgTrain', imgs,   train_size, 4096)
    write_split(hf_out, 'LabTrain', labs,   train_size, 10)
    write_split(hf_out, 'TagTrain', tags,   train_size, 1000)

    # Query
    imgs = fi[0: query_size]
    labs = fl[0: query_size]
    tags = ft[0: query_size]
    write_split(hf_out, 'ImgQuery', imgs,   query_size, 4096)
    write_split(hf_out, 'LabQuery', labs,   query_size, 10)
    write_split(hf_out, 'TagQuery', tags,   query_size, 1000)

    # Database
    imgs = fi[query_size:]
    labs = fl[query_size:]
    tags = ft[query_size:]
    db_size = imgs.shape[0]
    write_split(hf_out, 'ImgDataBase', imgs,   db_size, 4096)
    write_split(hf_out, 'LabDataBase', labs,   db_size, 10)
    write_split(hf_out, 'TagDataBase', tags,   db_size, 1000)

    hf_out.close()

def SaveH5File_I(resize_size):
    root = '../data/'
    file_path = os.path.join(root, 'iapr-tc12-rand.mat')
    data = sio.loadmat(file_path)
    valid_img = data['VDatabase'].astype('float32')
    valid_txt = data['YDatabase'].astype('float32')
    valid_labels = data['databaseL']
    test_img = data['VTest'].astype('float32')
    test_txt = data['YTest'].astype('float32')
    test_labels = data['testL']
    fi, ft, fl = np.concatenate([valid_img, test_img]), np.concatenate([valid_txt, test_txt]), np.concatenate([valid_labels, test_labels])
    query_size = 2000
    train_size = 10000
    imgs = list(fi[query_size: query_size + train_size])
    labs = list(fl[query_size: query_size + train_size])
    tags = list(ft[query_size: query_size + train_size])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,255])
    Tag = np.zeros([n,2912])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf = h5py.File('../data/IAPR.h5','w')
    hf.create_dataset('ImgTrain', data = Img)
    hf.create_dataset('TagTrain', data = Tag)
    hf.create_dataset('LabTrain', data = Lab)

    imgs = list(fi[0: query_size])
    labs = list(fl[0: query_size])
    tags = list(ft[0: query_size])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,255])
    Tag = np.zeros([n,2912])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgQuery', data = Img)
    hf.create_dataset('TagQuery', data = Tag)
    hf.create_dataset('LabQuery', data = Lab)

    imgs = list(fi[query_size::])
    labs = list(fl[query_size::])
    tags = list(ft[query_size::])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,255])
    Tag = np.zeros([n,2912])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgDataBase', data = Img)
    hf.create_dataset('TagDataBase', data = Tag)
    hf.create_dataset('LabDataBase', data = Lab)
    hf.close()     

def get_data(config):
    """
        Returns:
            train_loader,   # for training (shuffle, drop_last)
            eval_loader,    # for get_loss (no shuffle, keep all)
            test_loader,
            database_loader,
            meta_loader,
            num_train, num_test, num_database, num_meta
    """

    # Data augmentation for training
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.5,                    
            scale=(0.02, 0.4),       
            ratio=(0.3, 3.3),        
            value='random'            
        )
    ])
    # eval / test / database / meta only apply ToTensor transformations 
    default_transform = transforms.ToTensor()

    # train & eval loader on same dataset 
    train_ds = DataList(
        config["dataset"], "train",
        default_transform,         
        config["noise_type"], config["noise_rate"], config["random_state"]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=False
    )
    
    print("train:", len(train_ds))
    eval_loader = DataLoader(
        DataList(
            config["dataset"], "train",
            default_transform,     
            config["noise_type"], config["noise_rate"], config["random_state"]
        ),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    # test loader 
    test_loader = DataLoader(
        DataList(
            config["dataset"], "test",
            default_transform,
            config["noise_type"], config["noise_rate"], config["random_state"]
        ),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    # database loader 
    database_loader = DataLoader(
        DataList(
            config["dataset"], "database",
            default_transform,
            config["noise_type"], config["noise_rate"], config["random_state"]
        ),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    # meta loader  
    meta_loader = DataLoader(
        DataList(
            config["dataset"], "meta",
            default_transform,
            noise_type=None, noise_rate=0.0,
            random_state=config["random_state"],
            meta_path=config["meta_data_path"]
        ),
        batch_size=config["meta_batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    return (
        train_loader,
        eval_loader,
        test_loader,
        database_loader,
        meta_loader,
        len(train_ds),
        len(test_loader.dataset),
        len(database_loader.dataset),
        len(meta_loader.dataset)
    )

def compute_img_result(dataloader, net, device):
    bs, tclses, clses = [], [], []
    net.eval()
    for img, tag, tcls, cls, _ in tqdm(dataloader):
        tclses.append(tcls)
        clses.append(cls)
        bs.append((net(img.to('cuda'))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_tag_result(dataloader, net, device):
    bs,tclses, clses = [], [], []
    net.eval()
    for img, tag, tcls, cls, _ in tqdm(dataloader):
        tclses.append(tcls)
        clses.append(cls)
        tag = tag.float()
        bs.append((net(tag.to('cuda'))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def CalcHammingDist(B1, B2):
    #B1=B1.cpu()
    #B2=B2.cpu()
    q = B2.shape[1]
    distH = 0.5 * (q - torch.matmul(B1, B2.transpose(0,1)))

    return distH

def calc_map_k(
    rB,             
    qB,              
    retrieval_label, 
    query_label,    
    k=None,         
    device=None,    
    q_chunk=64       
):
    # 1. Device & Tensor Conversion 
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    def _to_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(device)
    qB = _to_tensor(qB).float()              # [m, d], float32
    rB = _to_tensor(rB).float()              # [n, d]
    query_label    = _to_tensor(query_label).float()    # [m, C]
    retrieval_label= _to_tensor(retrieval_label).float()# [n, C]

    m, d = qB.shape
    n, _ = rB.shape
    k = k or n

    total_AP = []
    
    for start in range(0, m, q_chunk):
        end = min(m, start + q_chunk)
        qB_chunk = qB[start:end]                   # [bs, d]
        ql_chunk = query_label[start:end]          # [bs, C]
        bs = end - start

        # 2. Compute similarity & top-k
        sim = qB_chunk @ rB.t()                    # [bs, n]
        sim_k, idx_k = torch.topk(sim, k, largest=True)  # [bs, k]

        # 3. Build ground truth & select top-k labels
        GND_chunk = (ql_chunk @ retrieval_label.t()) > 0  # bool [bs, n]
        gnd_k = GND_chunk.float().gather(1, idx_k)        # [bs, k]

        # 4. Parallel computation of AP for each query
      
        positions = torch.arange(1, k+1, device=device).view(1, k).expand(bs, k).float()
        tp_cum = torch.cumsum(gnd_k, dim=1)               # [bs, k]
        prec_i = tp_cum / positions                       # [bs, k]
        denom = gnd_k.sum(dim=1).clamp(min=1.0)            # [bs]
        AP_chunk = (prec_i * gnd_k).sum(dim=1) / denom     # [bs]

        total_AP.append(AP_chunk)

    # 5. Aggregate MAP@K 
    AP_all = torch.cat(total_AP, dim=0)                 
    mapk = AP_all.mean().item()                         
    return mapk

def pr_curve(rB, qB, retrieval_L, query_L, num_points=15):
    if isinstance(qB, np.ndarray):
        qB = torch.from_numpy(qB)
    # check if rB is a numpy array and convert it to a torch tensor if it is
    if isinstance(rB, np.ndarray):
        rB = torch.from_numpy(rB)
    # check if query_label is a numpy array and convert it to a torch tensor if it is
    if isinstance(query_L, np.ndarray):
        query_L = torch.from_numpy(query_L)
    # check if retrieval_label is a numpy array and convert it to a torch tensor if it is
    if isinstance(retrieval_L, np.ndarray):
        retrieval_L = torch.from_numpy(retrieval_L)
    qB, rB, query_L, retrieval_L = [i.cuda().to(dtype=torch.float64) for i in [qB, rB, query_L, retrieval_L]]
    num_query = query_L.shape[0]
    topK = retrieval_L.shape[0]
    query_L = query_L.float().cuda()
    retrieval_L = retrieval_L.float().cuda()
    qB = qB.float().cuda()
    rB = rB.float().cuda()
    GND = (query_L.mm(retrieval_L.t()) > 0).float()

    P, R = [], []
    # Uniformly sample num_points recall thresholds in [0, 1]
    for r_thresh in np.linspace(0, 1, num_points):
        K = max(1, int(r_thresh * topK))
        p = torch.zeros(num_query, device=qB.device)
        r = torch.zeros(num_query, device=qB.device)

        for i in range(num_query):
            dists = CalcHammingDist(qB[i], rB)
            _, inds = torch.sort(dists)
            
            correct = GND[i][inds[:K]].sum()
            p[i] = correct / K
            
            total_pos = GND[i].sum()
            r[i] = correct / total_pos if total_pos > 0 else 0.0

        P.append(p.mean().item())
        R.append(r.mean().item())

    return R, P

def pr_curve1(rB, qB, retrieval_L, query_L, num_points=15):
    import numpy as np
    if isinstance(qB, np.ndarray):
        qB = torch.from_numpy(qB)
    if isinstance(rB, np.ndarray):
        rB = torch.from_numpy(rB)
    if isinstance(query_L, np.ndarray):
        query_L = torch.from_numpy(query_L)
    if isinstance(retrieval_L, np.ndarray):
        retrieval_L = torch.from_numpy(retrieval_L)
    qB, rB, query_L, retrieval_L = [i.cuda().to(dtype=torch.float64) for i in [qB, rB, query_L, retrieval_L]]
    num_query = query_L.shape[0]
    topK = retrieval_L.shape[0]
    GND = (query_L.mm(retrieval_L.t()) > 0).float()

    P, R = [], []

    # Using a fixed list of recall thresholds
    fixed_recalls = [0.0001, 0.1088, 0.2097, 0.3048, 0.3946, 0.4782,
                     0.5562, 0.6296, 0.6975, 0.7611, 0.8193, 0.8722,
                     0.9199, 0.9631, 1.0]

    for r_thresh in fixed_recalls:
        K = max(1, int(r_thresh * topK))
        p = torch.zeros(num_query, device=qB.device)
        r = torch.zeros(num_query, device=qB.device)

        for i in range(num_query):
            dists = CalcHammingDist(qB[i], rB)
            _, inds = torch.sort(dists)
            correct = GND[i][inds[:K]].sum()
            p[i] = correct / K
            total_pos = GND[i].sum()
            r[i] = correct / total_pos if total_pos > 0 else 0.0

        P.append(p.mean().item())
        R.append(r.mean().item())

    return R, P

def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def TCalcTopMap(rB, qB, retrievalL, queryL, topk, tretrievalL, tqueryL):
    num_query = queryL.shape[0]
    topkmap = 0
    temp_ind = 0
    for iter in tqdm(range(num_query)):
        if np.dot(tqueryL[iter,:], queryL[iter,:].transpose()) > 0:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
            
            hamm = CalcHammingDist(qB[iter, :], rB)
            ind = np.argsort(hamm)
            gnd = gnd[ind]

            tgnd = gnd[0:topk]

            tsum = np.sum(tgnd).astype(int)
            if tsum == 0:
                continue
            count = np.linspace(1, tsum, tsum)

            tindex = np.asarray(np.where(tgnd == 1)) + 1.0
            topkmap_ = np.mean(count / (tindex))
            topkmap = topkmap + topkmap_
            temp_ind += 1
    cor_topkmap = topkmap / temp_ind

    topkmap = 0
    temp_ind = 0
    for iter in tqdm(range(num_query)):
        if np.dot(tqueryL[iter,:], queryL[iter,:].transpose()) == 0:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
            hamm = CalcHammingDist(qB[iter, :], rB)
            ind = np.argsort(hamm)
            gnd = gnd[ind]

            tgnd = gnd[0:topk]
            tsum = np.sum(tgnd).astype(int)
            if tsum == 0:
                continue
            count = np.linspace(1, tsum, tsum)

            tindex = np.asarray(np.where(tgnd == 1)) + 1.0
            topkmap_ = np.mean(count / (tindex))
            topkmap = topkmap + topkmap_
            temp_ind += 1
    oth_topkmap = topkmap / (temp_ind +0.0001)
    return cor_topkmap, oth_topkmap

def plot_gmm(gmm, X, clean_index, noisy_index, save_path='', plot_pdf=True):
    plt.figure(figsize=(7, 5))
    plt.clf()
    ax = plt.gca()

    # Compute PDF of whole mixture
    x = np.linspace(0, 1, 1000)
    logprob = gmm.score_samples(x.reshape(-1, 1))
    pdf = np.exp(logprob)

    # Compute PDF for each component
    responsibilities = gmm.predict_proba(x.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    # Plot data histogram
    ax.hist(X[clean_index], bins=100, density=True, histtype='stepfilled', color='green', alpha=0.4, label='Clean Pairs')
    ax.hist(X[noisy_index], bins=100, density=True, histtype='stepfilled', color='red', alpha=0.4, label='Noisy Pairs')

    # Plot PDF of whole model
    if plot_pdf:
        # Plot PDF of each component
        ax.plot(x, pdf_individual[:,  gmm.means_.argmin()], '--', label='Component A', color='green')
        ax.plot(x, pdf_individual[:,  gmm.means_.argmax()], '--', label='Component B', color='red')
        ax.plot(x, pdf, '-k', label='Mixture PDF')
    plt.yticks(size=15)
    plt.xticks(size=15)
    plt.xlabel('Per-sample loss', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.legend(loc='upper right', fontsize=12,frameon=True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_map_curve(epoch_list, i2t_mAP_list, t2i_mAP_list, save_path=None):
    """
    Plot mAP curves for I2T and T2I
        :param epoch_list: list of int, epochs at evaluation
        :param i2t_mAP_list: list of float, I2T mAP values
        :param t2i_mAP_list: list of float, T2I mAP values
        :param save_path: str or None, save the plot if a path is provided
    """

    plt.figure()
    plt.plot(epoch_list, i2t_mAP_list, marker='o', label='I2T mAP')
    plt.plot(epoch_list, t2i_mAP_list, marker='s', label='T2I mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Cross-modal retrieval mAP curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metaweight_vs_loss(
    metaweight, loss_array, epoch=None,
    output_dir='figure',
    figsize=(6, 5),
    cmap_name='jet',
    levels=100,        
    thresh=0.01,        
    scatter_size=5,
    scatter_alpha=0.5,
    heat_alpha=0.8
):
    """
    Draw a smooth heatmap with KDE plus scatter plot, and add a proper colorbar.
    Ensure the heatmap is above the scatter plot by explicitly setting zorder.
    """

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)

    # 1) Smooth heatmap base (KDE), zorder=2 ensures it appears above the scatter points.
    sns.kdeplot(
        x=metaweight, y=loss_array,
        fill=True,
        levels=levels,
        thresh=thresh,
        cmap=plt.get_cmap(cmap_name),
        alpha=heat_alpha,
        ax=ax,
        zorder=2
    )

    # 2) Extract the first QuadContourSet from the Axes as the mappable.
    quad = ax.collections[0]

    # 3) Add a colorbar.
    cbar = fig.colorbar(quad, ax=ax, label='Density')
    cbar.ax.tick_params(labelsize=8)

    # 4) Overlay light gray scatter points with zorder=1 to ensure they appear beneath the heatmap.
    ax.scatter(
        metaweight, loss_array,
        s=scatter_size,
        alpha=scatter_alpha,
        color='blue',
        linewidths=0,
        zorder=1
    )

    # 5) Beautify the plot.
    ax.grid(linestyle='--', alpha=0.3)
    ax.set_xlabel("MetaWeight")
    ax.set_ylabel("Loss")

    plt.tight_layout()
    fname = f'{epoch}MetaLoss_smooth_bar.png' if epoch is not None else 'MetaLoss_smooth_bar.png'
    save_path = os.path.join(output_dir, fname)
    plt.savefig(save_path, dpi=400)
    plt.close(fig)

    print(f"Saved → {save_path}")
    return save_path

# Define function to plot I2T and T2I PR curves
def plot_pr_curve(R, P, method_name, save_path, task):
    """
    Plot the PR curve for a single method.
    Args:
    - R, P: arrays of recall and precision;
    - method_name: name of the method (for the plot title);
    - save_path: prefix path to save the plot (without file extension);
    - task: "I2T" or "T2I".
    """

    plt.figure(figsize=(6, 5))
    plt.plot(R, P, marker='o')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{task} PR Curve - {method_name}")
    plt.grid(True)
    plt.tight_layout()
    out_file = f"{save_path}_PR_{task}.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {task} PR curve plot → {out_file}"
)

def record_pr_curve(R, P, method_name, save_dir, task):
    """
    Save PR curve data for a method as a .npz file.

    Args:
    - R: Recall array (list or numpy.ndarray)
    - P: Precision array (list or numpy.ndarray)
    - method_name: method name string, e.g., "MGHM(ours)"
    - save_dir: directory to save, e.g., "./pr_results"
    - task: task type string "I2T" or "T2I"
    """

    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{task}_{method_name}.npz"
    save_path = os.path.join(save_dir, file_name)
    np.savez(save_path, recall=np.array(R), precision=np.array(P))
    print(f"✅ PR curve data saved → {save_path}")

if __name__ == "__main__":
    # SaveH5File_I(256)
    # SaveH5File_F(256)
    # SaveH5File_C(256)
    SaveH5File_N(256)
    # file_path = "/data1/tza/NRCH-master/data/MSCOCO_deep_doc2vec_data_rand.h5py"
    # visualize_shapes(file_path)