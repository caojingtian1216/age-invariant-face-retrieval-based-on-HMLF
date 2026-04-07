import os
from .generic import ImageListRelevants2
from .generic import ImageListRelevants3

DB_ROOT = os.environ['DB_ROOT']

class fgnet_train(ImageListRelevants3):
    def __init__(self):
        ImageListRelevants3.__init__(self, os.path.join(DB_ROOT, 'FGNET/fgnet_train.pkl'),
                                 root=os.path.join(DB_ROOT, 'FGNET'))
class fgnet(ImageListRelevants2):
    def __init__(self):
        ImageListRelevants2.__init__(self, os.path.join(DB_ROOT, 'FGNET/gnd_fgnet.pkl'),
                                 root=os.path.join(DB_ROOT, 'FGNET'))
class fgnet_trainvalid(ImageListRelevants2):
    def __init__(self):
        ImageListRelevants2.__init__(self, os.path.join(DB_ROOT, 'FGNET/fgnet_trainvalid.pkl'),
                                 root=os.path.join(DB_ROOT, 'FGNET'))

