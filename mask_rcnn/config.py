"""
Inherit original Mask R-CNN config.py.
Add an initializer with input kwargs for convenience.
"""
from .mrcnn import config

class Config(config.Config):
    def __init__(self, **kwargs):
        """Set values of computed attributes."""
        for k in kwargs:
            if k in dir(self):
                setattr(self, k, kwargs[k])
        super(Config, self).__init__()
