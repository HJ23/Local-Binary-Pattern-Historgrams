import cv2
import numpy as np

class lbph:
    
    def __init__(self,window_size=3,base_size=256,block_size=8):
        self.WINDOW_SIZE=window_size
        self.CENTER_POINT=window_size//2
        self.BASE_SIZE=base_size
        self.BLOCK_SIZE=block_size
    
    def addImage(self,path):
        self.image=cv2.imread(path)
        self.image=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        self.image=cv2.resize(self.image,(self.BASE_SIZE,self.BASE_SIZE))
        
    
    def binarize(self,arr):
        ret=0
        for index,i in enumerate(arr):
            ret+= i*(2**index)
        return ret

   # get gradient of image

    def convolve(self):
        ret=np.zeros(self.image.shape,dtype=np.uint8)
        h,w=self.image.shape
        for x in range(0,h-self.WINDOW_SIZE):
            for y in range(0,w-self.WINDOW_SIZE):
                thresh=self.image[x+self.CENTER_POINT,y+self.CENTER_POINT]
                res=self.image[x:x+self.WINDOW_SIZE,y:y+self.WINDOW_SIZE]>thresh
                ret[x,y]=self.binarize(res.reshape(1,-1)[0])
        return ret

   # histogram feature of image
    def createHistogram(self,image_block):
        ret=[0 for x in range(256)]
        h,w=image_block.shape
        for y in range(h):
            for x in range(w):
                ret[image_block[y,x]]+=1
        return ret


    def feature_extractor(self,gradientImage):
        features=[]
        final_feature=[]
        block=self.BASE_SIZE//self.BLOCK_SIZE

        for blocky in range(self.BLOCK_SIZE+1):
            for blockx in range(self.BLOCK_SIZE+1):
                features.append(self.createHistogram(gradientImage[block*blocky:block*blocky+block, \
		                                                  block*blockx:block*blockx+block]))
        
        for feature in features:
            for data in  feature:
                final_feature.append(data)
        return final_feature


# euclidiean distance
def distance(x1,x2):
    dist=0
    for (a,b) in zip(x1,x2):
        dist+=(a-b)**2
    return np.sqrt(dist)*100/(256*256)
	


obj=lbph()

image1=obj.addImage("./img/obama1.png")   
gradient1=obj.convolve()
feature1=obj.feature_extractor(gradient1)

image2=obj.addImage("./img/obama2.png")   
gradient2=obj.convolve()
feature2=obj.feature_extractor(gradient2)

image3=obj.addImage("./img/mark.png")   
gradient3=obj.convolve()
feature3=obj.feature_extractor(gradient3)

# lower is better : if distance between 2 image is low then they are similar

print(distance(feature1,feature2))
print(distance(feature1,feature3))




