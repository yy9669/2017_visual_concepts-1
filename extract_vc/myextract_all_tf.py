#!/export/home/zzhang99/.linuxbrew/bin/python
import cv2,os,glob,pickle,sys
import numpy as np
import tensorflow as tf
from sys import argv

sys.path.insert(0, './')
sys.path.append('/home/xuyangf/project/ML_deliverables/Siamese_iclr17_tf-master/src/')
from tensorflow.python.client import timeline
from datetime import datetime
from utils import *
from global_variables import *
import cv2
import json
import sys
from copy import *
from feature_extractor import FeatureExtractor
from GetDataPath import *

# datapath=[]
# datapath.append('/data2/xuyangf/OcclusionProject/NaiveVersion/CroppedImage')
# Alexnet
# Apad_set = [0, 0, 16, 16, 32, 48, 64] # padding sizev
# Astride_set = [4, 8, 8, 16, 16, 16, 16] # stride size
# featDim_set = [96, 96, 256, 256, 384, 384, 256] # feature dimension
# Arf_set = [11, 19, 51, 67, 99, 131, 163]
# offset_set = np.ceil(np.array(Apad_set)/np.array(Astride_set)).astype(int)

Apad_set = [2, 6, 18, 42, 90] # padding size
Astride_set = [2, 4, 8, 16, 32] # stride size
featDim_set = [64, 128, 256, 512, 512] # feature dimension
Arf_set = [6, 16, 44, 100, 212]
offset_set = np.ceil(np.array(Apad_set).astype(float)/np.array(Astride_set)).astype(int)

# get pool1 layer parameters
pool_n = 2
Apad = Apad_set[pool_n]
Astride = Astride_set[pool_n]
featDim = featDim_set[pool_n]
Arf = Arf_set[pool_n]
offset = offset_set[pool_n]
print offset
cat=argv[1]
savepath = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature3/L3Feature'+cat

# number of patches to include for each save file (84*100 images)
#samp_size = 100000
samp_images = 300
#min_img_per_obj = 12
scale_size = 224

# Specify the dataset
# Dataset path
tf.logging.set_verbosity(tf.logging.INFO)
# data_file = os.path.join(g_data_folder, 'data_set')
# data_set = json.load(open(data_file, 'r'))
# data_set = data_set['train']
# paths = []
# for i in data_set.keys():
#     for j in data_set[i].keys():
#         if len(data_set[i][j]) >= min_img_per_obj:
#             paths += data_set[i][j]

myimage_path=LoadImage(cat)
image_path=[]
for mypath in myimage_path:
    myimg=cv2.imread(mypath, cv2.IMREAD_UNCHANGED)
    if(max(myimg.shape[0],myimg.shape[1])>100):
        image_path.append(mypath)

print(len(image_path))
# for i in range(len(datapath)):
#     image_path+=[os.path.join(datapath[i],s) for s in os.listdir(datapath[i])]

extractor = FeatureExtractor(which_layer='pool3', which_snapshot=0, from_scratch=False)


batch_num = 10
batch_size = int(len(image_path)/10)+1
print('batch_num: {0}'.format(batch_num))
check_num = 1  # save 1 batch to one file

step = int(0)
res=[]
irec = []
img_rec = []
originpath=[]
imageindex=[]
ggtmp=[]
for i in range (0,batch_num):
    print('batch :' + str(i))
    numwi=[]
    numhi=[]
    featurenumwi=[]
    featurenumhi=[]

    curr_paths = image_path[i * batch_size:(i + 1) * batch_size]
    features, images, blanks = extractor.extract_from_paths(curr_paths)

    tmp = features

    # for j in range(0,len(images)):
    #     fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/'+str(j) + '.png'
    #     cv2.imwrite(fname, images[j])
    
    images+=np.array([104., 117., 124.])
    img_rec.append(deepcopy(images))
    
    # for j in range(0,len(images)):
    #     fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/pp'+str(j) + '.png'
    #     cv2.imwrite(fname, img_rec[0][j])
    # break

    #assert (tmp.shape[0] == batch_size)
    #assert (tmp.shape[3] == featDim)
    height, width = tmp.shape[1:3]

    # remove offset patches
    for j in range(0,len(curr_paths)):
        woffset=0
        hoffset=0
        if(blanks[j][0]>blanks[j][1]):
            woffset=np.ceil((np.array(blanks[j][0]).astype(float)+np.array(Apad_set)[pool_n].astype(float))/np.array(Astride_set)[pool_n]).astype(int)
            hoffset=offset
        else:
            hoffset=np.ceil((np.array(blanks[j][1]).astype(float)+np.array(Apad_set)[pool_n].astype(float))/np.array(Astride_set)[pool_n]).astype(int)
            woffset=offset

        print('batch'+ str(i)+'image'+str(j) )
        print(blanks[j])
        print(woffset)
        print(hoffset)
        #break
        print('j ='+str(j))
        temp = tmp[j:j+1, hoffset:height - hoffset, woffset:width - woffset, :]

        ntmp = np.transpose(temp, (3, 0, 1, 2))
        gtmp = ntmp.reshape(ntmp.shape[0], -1)
        ggtmp.append(deepcopy(gtmp))
        print(gtmp.shape)

        imageindex+=[j for ixx in range(gtmp.shape[1])]
        originpath+=[curr_paths[j] for ixx in range(gtmp.shape[1])]
        for ixx in range(gtmp.shape[1]):
            hhi,wwi=np.unravel_index(ixx, (height - 2 * hoffset, width - 2 * woffset))
            featurenumhi.append(hhi+hoffset)
            featurenumwi.append(wwi+woffset)
            phi = Astride * (hhi + hoffset) - Apad
            pwi = Astride * (wwi + woffset) - Apad
            numhi.append(phi)
            numwi.append(pwi)
            #numhi.append(hhi+hoffset)
            #numwi.append(wwi+woffset)
            if ixx==0:
                print 'test'
                print(phi)
                print(pwi)
    # print(ggtmp)
    # ggtmp=np.array(ggtmp)
    # print(ggtmp.shape)

    # ggtmp=np.transpose(ggtmp,(1,0,2))
    # ggtmp=ggtmp.reshape(ggtmp.shape[0],-1)
    ggtmp=np.concatenate(ggtmp,axis=1)
    print(ggtmp.shape)
    res.append(deepcopy(ggtmp))
#numpy lie he bin de dao ggtemp
    irec += [i for ixx in range(ggtmp.shape[1])]

    if (i + 1) % check_num == 0 or i == batch_num - 1:
        print('output file {0}'.format(i / check_num))

        res = np.array(res)
        res = np.transpose(res, (1, 0, 2))
        itotal = res.shape[1]


        res = res.reshape(res.shape[0], -1)
        #print res.shape[1]
        # print(len(np.random.permutation(res.shape[1])))
        # rand_idx = np.random.permutation(res.shape[1])
        # res = res[:, rand_idx]
        print 'middle'
        #break
        # should also save the loc_set
        loc_set = []
        img_set = []
        irec = np.array(irec)
        #batch_size_f = batch_size * (height - 2 * offset) * (width - 2 * offset)
        num=0
        for rr in range(res.shape[1]):
            #aa, bb = np.unravel_index(rr, (itotal, batch_size_f))
            #ni, ihi, iwi = np.unravel_index(bb, (batch_size, height - 2 * hoffset, width - 2 * woffset))
            ni=imageindex[rr]
            # ihi,iwi=
            # hi = Astride * (ihi + offset) - Apad
            # wi = Astride * (iwi + offset) - Apad
            hi=numhi[rr]
            wi=numwi[rr]
            fhi=featurenumhi[rr]
            fwi=featurenumwi[rr]
            # assert (aa == irec[rr] % check_num)
            # assert (ni < batch_size)
            # assert (hi >= 0)
            # assert (hi <= 224 - Arf)
            # assert (wi >= 0)
            # assert (wi <= 224 - Arf)

            #img = img_rec[aa][ni].copy()

            img = img_rec[0][ni][hi:hi + Arf, wi:wi + Arf, :]
            #print(sys.getsizeof(img))
            #break
            img_set.append(img)
            # fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/'+ str(rr) + '.png'
            # cv2.imwrite(fname, img)
            #loc_set.append([i // check_num, irec[rr], ni, hi, wi, hi + Arf, wi + Arf])
            loc_set.append([i // check_num, irec[rr], ni, hi, wi, hi + Arf, wi + Arf,fhi,fwi])
            #if rr == rand_idx[50]:
            # print(loc_set)
        print 'last'
        # for ii in range(0,len(img_set)):
        #     fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/'+str(i)+'_'+ str(ii) + '.png'
        #     cv2.imwrite(fname, img_set[ii])
        np.savez(savepath + str(i // check_num), res=np.asarray(res), loc_set=np.asarray(loc_set),
                 img_set=np.asarray(img_set),originpath=np.asarray(originpath))


        res = []
        irec = []
        img_rec = []
        imgindex=[]
        originpath=[]
        ggtmp=[]
        imageindex=[]
        print 'file finish'
