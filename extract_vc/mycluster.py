#!/export/home/zzhang99/.linuxbrew/bin/python
from __future__ import division
import cv2
import numpy as np
import pickle
import time,math
from sklearn.cluster import KMeans
from sys import argv
cluster_num = 256
cat=argv[1]
layer_name = 'pool3'
save_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL3/vgg16_'+cat+'_K'+str(cluster_num)+'.pickle'
file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature3/L3Feature'+cat


# number of files to read in
file_num = 10
maximg_cnt=20000
patch_size=44

fname = file_path+str(0)+'.npz'
ff = np.load(fname)

feat_dim = ff['res'].shape[0]
img_cnt = ff['res'].shape[1]
oldimg_index=0

feat_set = np.zeros((feat_dim, maximg_cnt*file_num))
feat_set[:,0:img_cnt] = ff['res']

loc_dim = ff['loc_set'].shape[1]
loc_set = np.zeros((maximg_cnt*file_num, loc_dim))
loc_set[0:img_cnt,:] = ff['loc_set']

img_dim = ff['img_set'].shape[1:]
print(img_dim)
img_set = np.zeros([maximg_cnt*file_num]+list(img_dim))
print(ff['img_set'].shape)
print(img_cnt)
img_set[0:img_cnt] = ff['img_set']

oldimg_index+=img_cnt

for ii in range(1,file_num):
    print(ii)
    fname = file_path+str(ii)+'.npz'
    ff = np.load(fname)
    img_cnt=ff['res'].shape[1]
    print(img_cnt)
    feat_set[:,oldimg_index:(oldimg_index + img_cnt)] = ff['res']
    loc_set[oldimg_index:(oldimg_index + img_cnt),:] = ff['loc_set']
    img_set[oldimg_index:(oldimg_index + img_cnt)] = ff['img_set']
    oldimg_index+=img_cnt

feat_set=feat_set[:,:oldimg_index]
img_set=img_set[:oldimg_index]
loc_set=loc_set[:oldimg_index,:]

print('all feat_set')
print(feat_set.shape)
print('all img_set')
print(img_set.shape)

# L2 normalization as preprocessing
feat_norm = np.sqrt(np.sum(feat_set**2, 0))
feat_set = feat_set/feat_norm


print('Start K-means...')
_s = time.time()
km = KMeans(n_clusters=cluster_num, init='k-means++', random_state=99, n_jobs=1)
assignment = km.fit_predict(feat_set.T)
centers = km.cluster_centers_
_e = time.time()
print('K-means running time: {0}'.format((_e-_s)/60))

with open(save_path, 'wb') as fh:
    pickle.dump([assignment, centers], fh)
    
# the num of images for each cluster
num = 100
print('save top {0} images for each cluster'.format(num))
example = [None for nn in range(cluster_num)]

for k in range(cluster_num):
    target = centers[k]
    index = np.where(assignment == k)[0]
    num = min(num, len(index))
    
    tempFeat = feat_set[:,index]
    error = np.sum((tempFeat.T - target)**2, 1)
    sort_idx = np.argsort(error)
    patch_set = np.zeros(((patch_size**2)*3, num)).astype('uint8')
    for idx in range(num):
        patch = img_set[index[sort_idx[idx]]]
        patch_set[:,idx] = patch.flatten()
        
    example[k] = np.copy(patch_set)
    if k%20 == 0:
        print(k)
        

with open(save_path, 'wb') as fh:
    pickle.dump([assignment, centers, example], fh)
