import os
import pickle as pkl

class Embedding:
    def __init__(self, model_dir):
        pass
    def embed_one_by_path(self, path):
        dir = os.path.dirname(path)
        file, _ = os.path.splitext(os.path.basename(path))
        # print(path, os.path.join(dir, "embeddings.pkl"))
        embeddings = pkl.load(open(os.path.join(dir, "embeddings.pkl"), "rb"))
        return embeddings[file]