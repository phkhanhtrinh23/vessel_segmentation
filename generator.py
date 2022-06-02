import h5py
import numpy as np
import os

class DatasetGenerator:
    def __init__(self, dataset_path, batch_size):
        self.batch_size = batch_size
        
        self.db = h5py.File(dataset_path)
        self.numImages = self.db["images"].shape[0]
        print("The number of total images:",self.numImages)
        self.num_batches_per_epoch = int((self.numImages-1)/batch_size) + 1
    
    def getNumberOfTotalImages(self):
        return self.numImages
        
    def generator(self, passes=np.inf):
        epochs = 0
        
        while epochs < passes:
            shuffle_indices = np.arange(self.numImages) 
            shuffle_indices = np.random.permutation(shuffle_indices)
            for batch_num in range(self.num_batches_per_epoch):
                
                start_index = batch_num * self.batch_size
                end_index = min((batch_num + 1) * self.batch_size, self.numImages)

                batch_indices = sorted(list(shuffle_indices[start_index:end_index]))
                
                images = self.db["images"][batch_indices,:,:,:]
                labels = self.db["masks"][batch_indices,:,:,:]
                
                yield (images, labels)
            
            epochs += 1
            
    def close(self):
        self.db.close()
    
class DatasetReader:
    def __init__(self, dataset_path, batch_size):
        self.batch_size = batch_size
        
        self.db = h5py.File(dataset_path)
        self.numImages = self.db["images"].shape[0]
        print("The number of total images:",self.numImages)
        self.num_batches_per_epoch = int((self.numImages-1)/batch_size) + 1
        
    def getNumberOfTotalImages(self):
        return self.numImages
        
    def generator(self, passes=np.inf):
        epochs = 0
        while epochs < passes:
            images = self.db["images"][epochs,:,:,:]
            labels = self.db["masks"][epochs,:,:,:]
            
            yield (images, labels)
            
            epochs += 1
            
    def close(self):
        self.db.close()
    
class DatasetWriter:
    def __init__(self, image_dims, mask_dims, output_path, buffer_size=200):
        if os.path.exists(output_path):
            raise ValueError("The output_path already exists", output_path)
        
        self.db = h5py.File(output_path, "w")
        self.data = self.db.create_dataset("images", image_dims, dtype="float")
        self.masks = self.db.create_dataset("masks", mask_dims, dtype="int")
        
        self.buffer_size = buffer_size
        self.buffer = {"data": [], "masks": []}
        self.idx = 0

    def add(self, rows, masks):
        self.buffer["data"].extend(rows)
        self.buffer["masks"].extend(masks)
        
        if len(self.buffer["data"]) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i,:,:,:] = self.buffer["data"]
        self.masks[self.idx:i,:,:,:] = self.buffer["masks"]
        print("DatasetWriter have writen %d data"%i)
        self.idx = i
        self.buffer = {"data": [], "masks": []}
        
    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()
        
        self.db.close()
        return self.idx