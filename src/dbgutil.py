import scipy
from importlib import reload

def extract_sess(embedding_sess):
    return {
        "align_sess": embedding_sess.align_sess,
        "embed_sess": embedding_sess.embed_sess,
        "pnet": embedding_sess.pnet,
        "rnet": embedding_sess.rnet,
        "onet": embedding_sess.onet
    }

def reload_sess(embedding_sess, stored_sess):
    embedding_sess.align_sess = stored_sess["align_sess"]
    embedding_sess.embed_sess = stored_sess["embed_sess"]
    embedding_sess.pnet = stored_sess["pnet"]
    embedding_sess.rnet = stored_sess["rnet"]
    embedding_sess.onet = stored_sess["onet"]
    
def load_test_img(filename="../../datasets/test/Abdullah_Gul/Abdullah_Gul_0001.jpg"):
    return [scipy.misc.imread(filename)]
    