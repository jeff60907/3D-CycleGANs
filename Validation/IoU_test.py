import os
import numpy as np
import sys

if len(sys.argv) < 2:
    print 'Input your data name and groud_truth dir eg. chair ../Data/chair/train_3d/'
    sys.exit()

voxel_name = sys.argv[1]
data_dir = sys.argv[2]

# load fake data name
dir = []
def load_name():
    for filename in os.listdir('./'+voxel_name):
        if filename[0:-5] == voxel_name:
	    dir.append(filename)
    dir.sort()


def IoU_test(ground_truth, fake_data, test_name):
	data = np.zeros((64, 64, 64,1))
	for i in ground_truth:
	    data[int(i[0]), int(i[1]), int(i[2]),0] = 1
	ground_truth = data

	data = np.zeros((64, 64, 64,1))
	for i in fake_data:
	    data[int(i[0]), int(i[1]), int(i[2]),0] = 1
	fake_data = data


	ground_truth = np.reshape(ground_truth,(64,64,64))   # real
	fake_data= np.reshape(fake_data,(64,64,64))   # fake


	####  overlap  #### 
	tmp = np.logical_and(fake_data , ground_truth)  # fake_data & real
	x,y,z = np.where(tmp == 1)
	overlap = len(x)

	#### diff  ####
	diff_real = fake_data - tmp
	x,y,z = np.where(diff_real == 1)
	diff = len(x)

	#### real  ####
	x,y,z = np.where(ground_truth == 1)
	real = len(x)

	IoU_detail.write(test_name +'\n')
	print 'overlap: %d, real: %d, diff: %d' % (overlap,real,diff)
	IoU_detail.write('overlap: %d, real: %d, diff: %d' % (overlap,real,diff) +'\n')
	iou = (float(overlap)/(real+diff))    #  intersection-over-union 
	print 'IoU :', iou
	IoU.write(str(iou)+'\n')
	IoU_detail.write('IoU :%f' % iou +'\n')


if __name__ == '__main__':
    IoU_detail = open('IoU_detail_' +  voxel_name + '.txt','w')
    IoU = open('IoU_' + voxel_name +'.txt','w')
    load_name()
    real_data = os.listdir(data_dir)
    real_data.sort()
    for i in range(100):
        try:
            ground_truth = np.loadtxt(data_dir + real_data[i])    # real
            test_name = voxel_name + '/' + dir[i] + '/fake_' + dir[i] + '.asc'
            fake_data = np.loadtxt(test_name)    # fake
        except:
            print voxel_name + '/' + dir[i] +'/fake_' + dir[i] + '.asc_ERROR'
            continue
        IoU_test(ground_truth, fake_data, test_name)
    IoU_detail.close()
    IoU.close()



