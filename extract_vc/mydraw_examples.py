import pickle
import numpy as np
import cv2
import math
from sys import argv

cat=argv[1]
fname='/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL3/dictionary_'+cat+'.pickle'
#fname='/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL3/vgg16_0_K256.pickle'
# fname='dictionary_shapenet_VGG16_pool4_K512.pickle'
with open(fname,'rb') as fh:
    assignment, centers, example = pickle.load(fh)

ss=int(math.sqrt(example[0].shape[0]/3))
print ss
for ii in range(len(example)):
    big_img = np.zeros((10+(ss+10)*4, 10+(ss+10)*5, 3))
    for iis in range(20):
        if iis >= example[ii].shape[1]:
            continue
        aa = iis//5
        bb = iis%5
        rnum = 10+aa*(ss+10)
        cnum = 10+bb*(ss+10)
        big_img[rnum:rnum+ss, cnum:cnum+ss, :] = example[ii][:,iis].reshape(ss,ss,3).astype(int)

    fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/example'+cat+'/'+ str(ii) + '.png'
    cv2.imwrite(fname, big_img)
    print 'draw finish'