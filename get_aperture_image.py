from __future__ import division
import pickle
import numpy as np
import cv2
import math
from sys import argv
from copy import *
from ProjectUtils import *
from GetDataPath import *
from testvgg import TestVgg

sys.path.insert(0, './')
sys.path.append('/home/xuyangf/project/ML_deliverables/Siamese_iclr17_tf-master/src/')

import network as vgg
from feature_extractor import FeatureExtractor
from utils import *
# import importlib
# importlib.reload(sys)
reload(sys)  
sys.setdefaultencoding('utf8')

# now, just get top 10 visual concepts as aperture input 

cat=argv[1]
cluster_num = 256
myimage_path=LoadImage(cat)
image_path=[]
for mypath in myimage_path:
    myimg=cv2.imread(mypath, cv2.IMREAD_UNCHANGED)
    if(max(myimg.shape[0],myimg.shape[1])>100):
        image_path.append(mypath)
img_num=len(image_path)
layer_name = 'pool3'
file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature3/L3Feature'+cat
#cluster_file = '/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL3/vgg16_'+cat+'_K'+str(cluster_num)+'.pickle'
prun_file = '/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL3/dictionary_'+cat+'.pickle'

train_open1_name='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/train/top10vc/'

print('loading data...')

# number of files to read in
file_num = 10
maximg_cnt=20000

fname = file_path+str(0)+'.npz'
ff = np.load(fname)

feat_dim = ff['res'].shape[0]
img_cnt = ff['res'].shape[1]
oldimg_index=0

originimage=[]
feat_set = np.zeros((feat_dim, maximg_cnt*file_num))
feat_set[:,0:img_cnt] = ff['res']

originimage+=list(ff['originpath'])
loc_dim = ff['loc_set'].shape[1]
print(loc_dim)
loc_set = np.zeros((maximg_cnt*file_num, loc_dim))
loc_set[0:img_cnt,:] = ff['loc_set']

#img_dim = ff['img_set'].shape[1:]
#img_set = np.zeros([maximg_cnt*file_num]+list(img_dim))
#img_set[0:img_cnt] = ff['img_set']

oldimg_index+=img_cnt

for ii in range(1,file_num):
    print(ii)
    fname = file_path+str(ii)+'.npz'
    ff = np.load(fname)
    originimage+=list(ff['originpath'])
    img_cnt=ff['res'].shape[1]
    print(img_cnt)
    feat_set[:,oldimg_index:(oldimg_index + img_cnt)] = ff['res']
    loc_set[oldimg_index:(oldimg_index + img_cnt),:] = ff['loc_set']
    #img_set[oldimg_index:(oldimg_index + img_cnt)] = ff['img_set']
    oldimg_index+=img_cnt

feat_set=feat_set[:,:oldimg_index]
#img_set=img_set[:oldimg_index]
loc_set=loc_set[:oldimg_index,:]

print('all feat_set')
print(feat_set.shape)
print('all img_set')
#print(img_set.shape)
#assert(len(originimage)==len(img_set))

with open(prun_file, 'rb') as fh:
    assignment, centers, _,norm = pickle.load(fh)

print('load finish')

print('get top 10 vc')
fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/layer3/cat'+str(cat)+'.npz'
ff=np.load(fname)
img_vc=ff['vc_score']
vc_num=len(img_vc[0])
img_num=len(img_vc)
#print(img_vc[0])
img_vc_avg=[]
for i in range(vc_num):
    img_vc_avg.append(float(np.sum(img_vc[np.where(img_vc[:,i]!=-1),i]))/img_num)
img_vc_avg=np.asarray(img_vc_avg)
rindexsort=np.argsort(-img_vc_avg)

def disttresh(input_index,cluster_center):
    thresh1=0.5
    temp_feat=feat_set[:,input_index]
    error = np.sum((temp_feat.T - cluster_center)**2, 1)
    sort_idx = np.argsort(error)
    return input_index[sort_idx[:int(thresh1*len(sort_idx))]]


for k in rindexsort[:10]:
    target=centers[k]
    index=np.where(assignment==k)[0]
    index=disttresh(index,target)
    for n in range(0,img_num):
        myindex=[]
        for i in range(len(index)):
            if image_path[n]==originimage[index[i]]:
                myindex.append(index[i])
        #myindex=OnlyTheClosest(myindex,target), or other preprocessing method
        if len(myindex)==0:
            continue
        original_img=cv2.imread(image_path[n], cv2.IMREAD_UNCHANGED)
        oimage,_,__=process_image(original_img, '_',0)
        oimage+=np.array([104., 117., 124.])
        aperture_img=np.zeros((224,224,3)).astype('uint8')
        print(oimage.shape)
        for i in range(len(myindex)):
            hi=int(loc_set[myindex[i],3])
            wi=int(loc_set[myindex[i],4])
            Arf=int(loc_set[myindex[i],5])-int(loc_set[myindex[i],3])
            aperture_img[hi:hi+Arf,wi:wi+Arf,:]=oimage[hi:hi+Arf,wi:wi+Arf,:]
        findex=image_path[n].rfind('/')
        fname=train_open1_name+image_path[n][findex:-5]+'_VC_'+str(k)+'.jpeg'
        cv2.imwrite(fname,aperture_img)       
