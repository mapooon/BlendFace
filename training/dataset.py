import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn
from tqdm import tqdm
import pickle
import cv2
from PIL import Image

def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    seed = 2048,
    num_workers = 2,
    p_augment=None
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    train_set = AugmentedDataset(root_dir=root_dir, local_rank=local_rank, p_augment=p_augment)

    
    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

class AugmentedDataset(Dataset):
    def __init__(self, root_dir, local_rank, p_augment):
        super(AugmentedDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.local_rank=local_rank
        self.root_dir = root_dir
        self.p_augment=p_augment
        
        self.lmk_points=list(range(31))+list(range(36,49))
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
        
        path_idx2label=os.path.join(root_dir, 'idx2label.json')
        path_idx2id=os.path.join(root_dir, 'idx2id.json')
        if os.path.isfile(path_idx2label) and os.path.isfile(path_idx2id):
            print('loading dicts')
            self.idx2label=pickle.load(open(path_idx2label,'rb'))
            self.idx2id=pickle.load(open(path_idx2id,'rb'))
        else:
            idx2label={}
            idx2id={}
            print('constructing dicts')
            for index in tqdm(range(len(self.imgidx))):
                idx = self.imgidx[index]
                s = self.imgrec.read_idx(idx)
                header, img = mx.recordio.unpack(s)
                label = header.label
                if not isinstance(label, numbers.Number):
                    label = label[0]
                identity = header.id
                idx2label[idx]=int(label)
                idx2id[idx]=int(identity)
            pickle.dump(idx2label,open(path_idx2label,'wb'))
            pickle.dump(idx2id,open(path_idx2id,'wb'))
            self.idx2label=idx2label
            self.idx2id=idx2id
        print('done')

        path_idx2nearestidxs=os.path.join(root_dir, 'idx2nearestidxs.pkl')
        self.idx2nearestidxs=pickle.load(open(path_idx2nearestidxs,'rb'))
            

    def logical_or_masks(self,mask_list):
        mask_all = np.zeros_like(mask_list[0], dtype=bool)
        for mask in mask_list:
            mask_all = np.logical_or(mask_all, mask)
        mask_all=1.0*mask_all
        return mask_all

    def encode_segmentation_rgb(self,parse, no_neck=True):

        face_part_ids = [1, 2, 3, 4, 5, 6, 10, 12, 13] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
        mouth_id = 11
        hair_id = 17
        face_map = np.zeros([parse.shape[0], parse.shape[1]])
        mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
        hair_map = np.zeros([parse.shape[0], parse.shape[1]])

        for valid_id in face_part_ids:
            valid_index = np.where(parse==valid_id)
            face_map[valid_index] = 1
        valid_index = np.where(parse==mouth_id)
        mouth_map[valid_index] = 1
        valid_index = np.where(parse==hair_id)
        hair_map[valid_index] = 1

        return self.logical_or_masks([face_map, mouth_map])


    def color_transfer(self,s,t,sm=None,tm=None):
        s=cv2.cvtColor(cv2.cvtColor(s,cv2.COLOR_RGB2BGR).astype('float32')/255,cv2.COLOR_BGR2LAB).astype('float32')
        t=cv2.cvtColor(cv2.cvtColor(t,cv2.COLOR_RGB2BGR).astype('float32')/255,cv2.COLOR_BGR2LAB).astype('float32')
        if sm is not None:
            s_=s[sm]
        else:
            s_=s.reshape((-1,3))
        if tm is not None:
            t_=t[tm]
        else:
            t_=t.reshape((-1,3))
        s_mean=s_.mean(0)
        t_mean=t_.mean(0)
        s_std=s_.std(0)
        t_std=t_.std(0)
        assert (0==s_std).sum()==0
        assert (0==t_std).sum()==0
        s_mean=s_mean.reshape((1,1,3))
        t_mean=t_mean.reshape((1,1,3))
        s_std=s_std.reshape((1,1,3))
        t_std=t_std.reshape((1,1,3))
        t=(t-t_mean)*(s_std/t_std)+s_mean
        t = cv2.cvtColor(cv2.cvtColor(t,cv2.COLOR_LAB2BGR)*255,cv2.COLOR_BGR2RGB)
        t=t.clip(0,255)
        return t

    def swap_attr(self,img_t,img_s, mask_t,mask_s):
        mask_t=self.encode_segmentation_rgb(mask_t)
        mask_s=self.encode_segmentation_rgb(mask_s)
        img_s=self.color_transfer(s=img_t,t=img_s,sm=mask_t==1,tm=mask_s==1)
        mask=mask_s*mask_t
        
        assert mask.sum()>max(mask_t.sum(),mask_s.sum())*0.5
        mask=cv2.GaussianBlur(mask,(11,11),0)
        mask/=mask.max()
        mask[mask!=1]=0
        mask=cv2.GaussianBlur(mask,(11,11),0)
        mask/=mask.max()
        mask=mask.reshape((112,112,1))
        swapped=img_s*mask+img_t*(1-mask)
        
        return swapped.astype(np.uint8)


    def get_mask_path(self,idx):
        return os.path.join(self.root_dir,f'masks/{self.idx2label[idx]}/{self.idx2id[idx]}.npy')

    def __getitem__(self, index):
        flag=True
        do_aug=self.p_augment>torch.rand(1).item()
        while flag:
            try:
                idx = self.imgidx[index]
                s = self.imgrec.read_idx(idx)
                header, img = mx.recordio.unpack(s)
                label = self.idx2label[idx]
                identity = self.idx2id[idx]
                sample = mx.image.imdecode(img).asnumpy()
                mask_path=self.get_mask_path(idx)
                if do_aug:
                    try:
                        idx_candidate=self.idx2nearestidxs[idx]
                        idx_nearest=idx_candidate[np.random.randint(len(idx_candidate))]
                        mask=np.load(mask_path)
                        s_neg = self.imgrec.read_idx(idx_nearest)
                        mask_path_neg=self.get_mask_path(idx_nearest)
                        mask_neg=np.load(mask_path_neg)
                        header_neg, img_neg = mx.recordio.unpack(s_neg)
                        sample_neg = mx.image.imdecode(img_neg).asnumpy()
                        sample=self.swap_attr(img_t=sample_neg,img_s=sample, mask_s=mask,mask_t=mask_neg)
                    except Exception as e:
                        pass
                
                sample = self.transform(sample)
                label = torch.tensor(label, dtype=torch.long)
                flag=False
            except Exception as e:
                print(label,identity,e)
                index=torch.randint(low=0,high=len(self),size=(1,)).item()
        return sample, label

    def __len__(self):
        return len(self.imgidx)
