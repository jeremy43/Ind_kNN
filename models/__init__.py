from __future__ import absolute_import
from .Naive import *
from .Naive_mnist import *
from .ResNet import *
from .SEResNet import *
from .DenseNet import *
from .MuDeep import *
from .HACNN import *
from .SqueezeNet import *
from .MobileNet import *
from .ShuffleNet import *
from .Xception import *
from .InceptionV4 import *
from .NASNet import *
from .DPN import *
from .InceptionResNetV2 import *

__factory = {
    'naive_mnist': Naive_mnist,
    'naive': Naive,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'seresnet50': SEResNet50,
    'seresnet101': SEResNet101,
    'seresnext50': SEResNeXt50,
    'seresnext101': SEResNeXt101,
    'resnet50m': ResNet50M,
    'resnet50ma2': ResNet50MA2,
    'densenet121': DenseNet121,
    'squeezenet': SqueezeNet,
    'mobilenet': MobileNetV2,
    'shufflenet': ShuffleNet,
    'xception': Xception,
    'inceptionv4': InceptionV4ReID,
    'nasnet': NASNetAMobile,
    'dpn92': DPN,
    'inceptionresnetv2': InceptionResNetV2,
    'mudeep': MuDeep,
    'hacnn': HACNN,
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
