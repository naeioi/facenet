from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import facenet
import argparse
import tensorflow as tf
import numpy as np
import align.detect_face
import random
from time import sleep

class Embedding:
    """
        self.align_sess: Tensorflow session for MTCNN graph of pre-train model
        self.embed_sess: Tensorflow session for facenet of pre-train model
    """
    def __init__(self, model_dir="../../model/20170512-110547", dont_load=False):
        if(dont_load == False):
            assert isinstance(model_dir, str), "A directory to facenet pre-trained model should be given to initialize <class Embedding>!"        
            with tf.Graph().as_default():
                self.align_sess = tf.Session()
                with self.align_sess.as_default():
                    self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.align_sess, None)
        
            with tf.Graph().as_default():
                self.embed_sess = tf.Session()
                with self.embed_sess.as_default():
                    facenet.load_model(model_dir)

    def __align__(self, images, size=160, margin=32):
        align_sess = self.align_sess
        pnet = self.pnet
        rnet = self.rnet
        onet = self.onet
        
        aligned_images = np.empty((len(images), size, size, 3))
        aligned_nums = 0
        
        minsize = 20 # minimum size of face
        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        factor = 0.709 # scale factor
        
        with align_sess.graph.as_default():
            with align_sess.as_default():
                for image in images:
                    img = image[:,:,0:3] # make a copy. 

                    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces>0:
                        det = bounding_boxes[:,0:4]
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces>1:
                            bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                            img_center = img_size / 2
                            offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                            offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                            index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                            det = det[index,:]
                        det = np.squeeze(det)
                        bb = np.zeros(4, dtype=np.int32)
                        # seems like margin gives some extra context beyond boundingbox given by MTCNN
                        bb[0] = np.maximum(det[0]-margin/2, 0)
                        bb[1] = np.maximum(det[1]-margin/2, 0)
                        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                        scaled = misc.imresize(cropped, (size, size), interp='bilinear')
                        ### The cropped image is stored here in scaled 
                        ### For our purpose, this image is not neccesarily to be saved on disk 
                        ###     but passed directly to facenet for an embedding.
                        aligned_images[aligned_nums,:,:,:] = scaled[:,:,:]
                        aligned_nums += 1
                    
        return aligned_images

    def __embed__(self, aligned_images, size=160):
        imgs = aligned_images
        embed_sess = self.embed_sess

        images = facenet.preprocess_data(aligned_images, False, False, size)
        
        with embed_sess.graph.as_default():
            with embed_sess.as_default():                
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array = embed_sess.run(embeddings, feed_dict=feed_dict)

                return emb_array
    
    def embed_one(self, image):
        return self.__embed__(self.__align__([image]))[0, :]
        
    def embed_batch(self, images): 
        """ Embed a batch or unaligend image
        
        Param:
            images: narray. in shape of [index, w, h, channels]
                An array of unaligend images.
            
        Returns:
            An array of corresponding embeddings.
        """
        
        return self.__embed__(self.__align__(images))
