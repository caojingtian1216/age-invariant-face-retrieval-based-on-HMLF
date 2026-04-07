import os
from .generic import ImageListRelevants2
from .generic import ImageListRelevants3

DB_ROOT = os.environ['DB_ROOT']

class imdb_train(ImageListRelevants3):
    def __init__(self):
        ImageListRelevants3.__init__(self, os.path.join(DB_ROOT, 'imdb-clean-1024/imdb_train.pkl'),
                                 root=os.path.join(DB_ROOT, 'imdb-clean-1024'))
class imdb(ImageListRelevants2):
    def __init__(self):
        ImageListRelevants2.__init__(self, os.path.join(DB_ROOT, 'imdb-clean-1024/gnd_imdb.pkl'),
                                 root=os.path.join(DB_ROOT, 'imdb-clean-1024'))
class imdb_trainvalid(ImageListRelevants2):
    def __init__(self):
        ImageListRelevants2.__init__(self, os.path.join(DB_ROOT, 'imdb-clean-1024/imdb_trainvalid.pkl'),
                                 root=os.path.join(DB_ROOT, 'imdb-clean-1024'))