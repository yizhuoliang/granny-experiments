from invoke import Collection

from . import dummySched,profile, slow

ns = Collection(dummySched, profile, slow)