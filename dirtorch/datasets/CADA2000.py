import os
from .generic import ImageListRelevants2
from .generic import ImageListRelevants3

DB_ROOT = os.environ['DB_ROOT']

class CADA2000_train(ImageListRelevants3):
    def __init__(self):
        ImageListRelevants3.__init__(self, os.path.join(DB_ROOT, 'CACD/data/cada_train.pkl'),
                                 root=os.path.join(DB_ROOT, 'CACD/data'))
class CADA2000_1(ImageListRelevants2):
    def __init__(self):
        ImageListRelevants2.__init__(self, os.path.join(DB_ROOT, 'CACD/data/gnd_cada1.pkl'),
                                 root=os.path.join(DB_ROOT, 'CACD/data'))
class CADA2000_2(ImageListRelevants2):
    def __init__(self):
        ImageListRelevants2.__init__(self, os.path.join(DB_ROOT, 'CACD/data/gnd_cada2.pkl'),
                                 root=os.path.join(DB_ROOT, 'CACD/data'))
class CADA2000_3(ImageListRelevants2):
    def __init__(self):
        ImageListRelevants2.__init__(self, os.path.join(DB_ROOT, 'CACD/data/gnd_cada3.pkl'),
                                 root=os.path.join(DB_ROOT, 'CACD/data'))
class CADA2000_trainvalid(ImageListRelevants2):
    def __init__(self):
        ImageListRelevants2.__init__(self, os.path.join(DB_ROOT, 'CACD/data/cada_trainvalid.pkl'),
                                 root=os.path.join(DB_ROOT, 'CACD/data'))

