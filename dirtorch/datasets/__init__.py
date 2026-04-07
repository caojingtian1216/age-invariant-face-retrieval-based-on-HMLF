try: from .oxford import *
except ImportError: pass
try: from .paris import *
except ImportError: pass
try: from .CADA2000 import *
except ImportError: pass
try: from .morph import *
except ImportError: pass
try: from .fgnet import *
except ImportError: pass
try: from .agedb import *
except ImportError: pass
try: from .imdb import *
except ImportError: pass
try: from .distractors import *
except ImportError: pass
try: from .landmarks import Landmarks_clean, Landmarks_clean_val, Landmarks_lite
except ImportError: pass
try: from .landmarks18 import *
except ImportError: pass

# create a dataset from a string
from .create import *
create = DatasetCreator(globals())#globals表示传入所import的模块的所有变量、类等等

from .dataset import split, deploy, deploy_and_split
from .generic import *
