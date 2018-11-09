import h5py

f = h5py.File('/hdd/robik/CLEVR/faster-rcnn/train.hdf5', 'r')

img_features = f['image_features'][0:100]
print("img_features.shape {}".format(img_features.shape))

print("sum1: {}".format(img_features[:, :,].sum()))
print("sum14: {}".format(img_features[:, :, 14].sum()))