import mxnet as mx
from mxnet import image

from gluoncv.data.transforms.presets.segmentation import test_transform

from gluoncv.model_zoo.segbase import get_segmentation_model

import mxnet as mx

def get_segment_mask(img_path):
    # using cpu
    ctx = mx.cpu(0)
    file_path = img_path
    img = image.imread(file_path)

    img = test_transform(img, ctx)

    model = get_segmentation_model(model='fcn', dataset='pascal_aug',
                                            backbone='resnet50', norm_layer=mx.gluon.nn.BatchNorm,
                                            norm_kwargs={}, aux=False,
                                            crop_size=480)
    model.load_parameters('./segmentation/checkpoint.params')
    output = model.predict(img)
    data = mx.nd.argmax(output[0], 1)
    predict = mx.nd.squeeze(*data).asnumpy()


    return predict