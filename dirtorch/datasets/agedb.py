import os
from .generic import ImageListRelevants2
from .generic import ImageListRelevants3

DB_ROOT = os.environ['DB_ROOT']

class agedb_train(ImageListRelevants3):
    def __init__(self):
        ImageListRelevants3.__init__(self, os.path.join(DB_ROOT, 'AgeDB/agedb_train.pkl'),
                                 root=os.path.join(DB_ROOT, 'AgeDB/archive'))
class agedb(ImageListRelevants2):
    def __init__(self):
        ImageListRelevants2.__init__(self, os.path.join(DB_ROOT, 'AgeDB/gnd_agedb.pkl'),
                                 root=os.path.join(DB_ROOT, 'AgeDB/archive'))
class agedb_trainvalid(ImageListRelevants2):
    def __init__(self):
        ImageListRelevants2.__init__(self, os.path.join(DB_ROOT, 'AgeDB/agedb_trainvalid.pkl'),
                                 root=os.path.join(DB_ROOT, 'AgeDB/archive'))
