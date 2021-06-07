import os
import msgpack
from PIL import Image
import glob
import numpy as np
from tqdm import tqdm
from joblib import delayed, Parallel
import lmdb
from io import BytesIO
import random
import json
import cv2

import msgpack_numpy as m
m.patch()

from dataloaders.rawvideo_util import RawVideoExtractor

anno_path='/users/vlad/CLIP4Clip/msrvtt_data/msrvtt_all.txt'

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def pil_from_raw(raw):
    return Image.open(BytesIO(raw))

def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data

def read_file(path):
    with open(path, 'r') as f:
        content = f.readlines()
    content = [i.strip() for i in content]
    return content

def read_json(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content

def resize_dim(w, h, target):
    '''resize (w, h), such that the smaller side is target, keep the aspect ratio'''
    if w >= h:
        return (int(target * w / h), int(target))
    else:
        return (int(target), int(target * h / w))

rawVideoExtractor = RawVideoExtractor(framerate=1, size=224)

def get_array(v_path, dim=224):
    v = rawVideoExtractor.get_video_data(v_path)
    return v['video'].cpu().detach().numpy()

def make_dataset_lmdb(dataset_path, filename, mode='val'):
    filtered_txt = read_file(anno_path)

    lmdb_path = filename
    isdir = os.path.isdir(lmdb_path)
    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=int(2e12), readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)

    print('check video availability')
    video_list = sorted(glob.glob(os.path.join(dataset_path, '*.mp4')))

    # shuffle, assign id
    random.seed(0)
    video_name_list = sorted(filtered_txt)
    random.shuffle(video_name_list)
    with open(lmdb_path+'-order', 'w') as f:
        f.write('\n'.join(video_name_list))
    get_video_id = dict(zip(video_name_list, video_name_list))
    filtered_video_set = set(video_name_list)

    # filter
    null_video = []
    global_key_list = []

    video_list = [i for i in video_list if os.path.basename(i)[0:-4] in filtered_video_set]
    # for train set only
    video_list_global = [video_list[0:4000], video_list[4000:8000], video_list[8000:12000], video_list[12000::]]

    for video_list in video_list_global:
        array_list = Parallel(n_jobs=12)(delayed(get_array)(vp) for vp in tqdm(video_list, total=len(video_list)))
        for i, (array, vp) in tqdm(enumerate(zip(array_list, video_list)), total=len(video_list)):
            vname = vp.split('/')[-1][0:-4]
            vid = get_video_id[vname]
            if array is not None:
                success = txn.put(vid.encode('ascii'), msgpack.dumps(array))
                if not success: print('%s failed to put in lmdb' % vname)
                global_key_list.append(vid.encode('ascii'))
            else:
                null_video.append(vname)
            if i % 100 == 0:
                txn.commit()
                txn = db.begin(write=True)

    video_name_list = [i for i in video_name_list if i not in null_video]
    txn.put(b'__keys__', msgpack.dumps(global_key_list))
    txn.put(b'__len__', msgpack.dumps(len(global_key_list)))
    txn.put(b'__order__', msgpack.dumps(video_name_list))

    txn.commit()
    print("Flushing database ...")
    db.sync()
    db.close()

import time

def read_lmdb(db_path):
    tic = time.time()
    env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                    readonly=True, lock=False,
                    readahead=False, meminit=False)
    print('Loading lmdb takes %.2f seconds' % (time.time()-tic))
    with env.begin(write=False) as txn:
        length = msgpack.loads(txn.get(b'__len__'))
        keys = msgpack.loads(txn.get(b'__keys__'))

    with env.begin(write=False) as txn:
        raw = msgpack.loads(txn.get('video1'.encode()))
    return (keys, length, raw.shape)

if __name__ == '__main__':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


    make_dataset_lmdb(dataset_path='/users/vlad/CLIP4Clip/msrvtt_data/MSRVTT_Videos',
                      filename='msrvtt_data/lmdb/msrvtt_all.lmdb',
                      mode='train')

    print(read_lmdb('msrvtt_data/lmdb/msrvtt_all.lmdb'))
