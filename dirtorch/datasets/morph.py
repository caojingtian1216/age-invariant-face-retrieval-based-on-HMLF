import os
from .generic import ImageListRelevants2
from .generic import ImageListRelevants3

DB_ROOT = os.environ['DB_ROOT']

class morph_train(ImageListRelevants3):
    def __init__(self):
        ImageListRelevants3.__init__(self, os.path.join(DB_ROOT, 'morph/data/morph_train.pkl'),
                                 root=os.path.join(DB_ROOT, 'morph/data'))
class morph(ImageListRelevants2):
    def __init__(self):
        ImageListRelevants2.__init__(self, os.path.join(DB_ROOT, 'morph/data/gnd_morph.pkl'),
                                 root=os.path.join(DB_ROOT, 'morph/data'))
class morph_trainvalid(ImageListRelevants2):
    def __init__(self):
        ImageListRelevants2.__init__(self, os.path.join(DB_ROOT, 'morph/data/morph_trainvalid.pkl'),
                                 root=os.path.join(DB_ROOT, 'morph/data'))

