import os
import cv2
import numpy as np
import tensorflow as tf

class imagefolder:
    def __init__(self, basedir, imgshape, nchannels):
        self.basedir = basedir
        self.nchannels = nchannels
        self.imgshape = imgshape
        self.classdict = {}
        
        self.pathdict = {}
        self.labeldict = {}
        
        self.tfdict = {}
        self.npdict = {}
        
        self._tfnorm = False
        self._nparray = np.zeros((1,1) )
        self._nplist = list()
    
    def loadpaths(self, subdir = '.', classes = 'all'):
        imagepaths, labels = list(), list()
        datadir = self.basedir + '/' + subdir
        _, classlist, imagelist = os.walk(datadir).__next__()
        
        if len(classlist) > 0 and classes is not 'all':
            classlist = classes
            
        label = 0
        for c in classlist:
            c_dir = os.path.join(datadir, c)
            filelist = os.walk(c_dir).__next__()[2]
            
            for sample in filelist:
                if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(label)
            if c not in self.classdict:
                self.classdict[label] = c
            label += 1
        
        setname = subdir if subdir is not '.' else 'all'
        self.pathdict[setname] = imagepaths
        
        if label > 0:
            self.labeldict[setname] = labels
            
    def onehot(self, classes):
        nclasses = len(self.classdict)
        return np.eye(nclasses, dtype=float)[classes]
        
    def _tfparse(self, filename, label, norm = False):
        imgstr = tf.read_file(filename)
        imgdec = tf.image.decode_jpeg(imgstr, channels = self.nchannels)
        imgresize = tf.image.resize_images(imgdec, [self.imgshape[0], self.imgshape[1]])
        
        if self._tfnorm:
            imgresize = tf.image.per_image_standardization(imgresize)
        return imgresize, label

    def tfload(self, batchsize = 1, norm = False, onehot = True, shuffle = True):
        assert len(self.pathdict) > 0
        
        for xtype in self.pathdict:
            paths = self.pathdict[xtype]
            
            if onehot:
                labels = self.onehot(np.array(self.labeldict[xtype]))
            else:
                labels = np.array(self.labeldict[xtype])
            
            if norm:
                self._tfnorm = True
                
            data = tf.data.Dataset.from_tensor_slices((paths, labels))
            data = data.map(self._tfparse)
                
            data = data.shuffle(buffer_size = len(paths)) if shuffle is True else data
            self.tfdict[xtype] = data.batch(batchsize)
    
    def _npresize(self, filename):
        img = cv2.imread(filename )
        rimg = cv2.resize(img, (self.imgshape[0], self.imgshape[1]) )

        idx = self._nplist.index(filename)
        self._nparray[idx] = rimg
    
    def npload(self, norm = False, onehot = True, ncores = 4):
        assert len(self.pathdict) > 0
        
        from multiprocessing import pool
        from multiprocessing.dummy import Pool as DumbPool
        
        for xtype in self.pathdict:
            paths = self.pathdict[xtype]
            
            self._nplist = paths
            self._nparray = np.zeros((len(self._nplist), self.imgshape[0], \
                                      self.imgshape[1], self.nchannels))
            pool = DumbPool(ncores)
            pool.map(self._npresize, self._nplist )
            
            if norm:
                for i in range(len(self._nparray)):
                    imgmean = np.mean(self._nparray[i])
                    numelem = np.prod(self.imgshape)*self.nchannels
                    adjstd = np.amax([np.std(self._nparray[i]), 1.0/np.sqrt(numelem)] )
                    self._nparray[i] = (self._nparray[i] - imgmean)/adjstd
            
            if onehot:
                labels = self.onehot(np.array(self.labeldict[xtype]))
            else:
                labels = np.array(self.labeldict[xtype])
            
            self.npdict[xtype] = (self._nparray, np.array(labels) )
            
    def npbatchid(self, xtype, batchsize = 1):
        assert xtype in self.npdict
        trainsize = len(self.npdict[xtype][0])
        
        randid = np.random.permutation(trainsize)
        batchid = [randid[i:i+batchsize] for i in np.arange(0, trainsize, batchsize) \
                   if i+batchsize < trainsize]

        return batchid