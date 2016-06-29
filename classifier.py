import os
import numpy as np
from matplotlib import pyplot as plt
import skimage
import theano
import lasagne
import pretrained_vgg_models

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer, Pool2DLayer, Conv2DLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

path = 'test.jpg'

img = plt.imread(path)

# load the pre-trained model
vgg19_params = pretrained_vgg_models.VGG19Model.load_params()

# make the required network
def build_vgg19_network():

    net = {}
    	# Images are 224X224, 3 channels. Batch size will be set later.
    net['input'] = InputLayer((None, 3, 224, 224))
			 
# First two convolutional layers: 64 filters, 3x3 convolution, 1 pixel padding
    # flip_filters is on because we are using a network imported from Caffe
	# this is a bit of a pain as it's not supported in later versions of theano
	# (probably should switch to tf)
    net['conv1_1'] = Conv2DLayer(net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = Conv2DLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    # 2x2 max-pooling
    net['pool1'] = Pool2DLayer(net['conv1_2'], 2)
    
    # Two convolutional layers, 128 filters
    net['conv2_1'] = Conv2DLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = Conv2DLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    # 2x2 max-pooling
    net['pool2'] = Pool2DLayer(net['conv2_2'], 2)
    
    # Four convolutional layers, 256 filters
    net['conv3_1'] = Conv2DLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = Conv2DLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = Conv2DLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_4'] = Conv2DLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
    # 2x2 max-pooling
    net['pool3'] = Pool2DLayer(net['conv3_4'], 2)
  # Four convolutional layers, 512 filters
    net['conv4_1'] = Conv2DLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = Conv2DLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = Conv2DLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['conv4_4'] = Conv2DLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
    
    net['pool4'] = Pool2DLayer(net['conv4_4'], 2)
    
    # Four convolutional layers, 512 filters
    net['conv5_1'] = Conv2DLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = Conv2DLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = Conv2DLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['conv5_4'] = Conv2DLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
    #
    net['pool5'] = Pool2DLayer(net['conv5_4'], 2)
    
    # Fully connectred layer
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    # 50% dropout used in training
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    
    # Dense layer
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    
    # 1000 units: 1 for each class
    net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=1000, nonlinearity=None)
    # Softmax for p
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

# Build net
vgg19_net = build_vgg19_network()

lasagne.layers.set_all_param_values(vgg19_net['prob'], vgg19_params['param values'])
in_var = theano.tensor.tensor4('x')
prob_expr = lasagne.layers.get_output(vgg19_net['prob'], in_var, deterministic=True)
prob_fn = theano.function([in_var], prob_expr)

def vgg_prepare_image(im, image_mean, image_size=224):
        
    # Scale the image
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (image_size, w * image_size / h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h * image_size / w, image_size), preserve_range=True)

    # Crop the central 224x224
    h, w, _ = im.shape
    im = im[h//2 - image_size // 2:h // 2 + image_size // 2, w // 2 - image_size // 2:w // 2 + image_size // 2]

    # Convert to uint8 type
    rawim = np.copy(im).astype('uint8')

    # (height, width, channel) to (channel, height, width)
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Images come in RGB channel order, while VGG net expects BGR:
    im = im[::-1, :, :]
    
    # If necessary, add 2 axes to the mean so that it will broadcast when we subtract
    # it from the image
    if len(image_mean.shape) == 1:
        image_mean = image_mean[:,None,None]

    # Subtract the mean
    im = im - image_mean
    
    # Add the sample axis 
    im = im[np.newaxis]
    
    return rawim, floatX(im)

raw_img, img_for_vgg = vgg_prepare_image(img, image_mean=vgg19_params['mean value'])

pred_prob = prob_fn(img_for_vgg)
pred_cls = np.argmax(pred_prob, axis=1)
print('Predicted class index {} with probability {:.2f}%, named "{}"'.format(
    pred_cls[0], pred_prob[0, pred_cls[0]]*100.0, vgg19_params['synset words'][pred_cls[0]]))    
