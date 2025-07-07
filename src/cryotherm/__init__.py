from importlib import resources as _res

DATA_PATH = _res.files(__name__) / "data"
