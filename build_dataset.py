import os
import random
import shutil
import Augmentor
from imutils import paths
from PIL import Image
from keras.utils import np_utils

dest = "lobular_carcinoma"
ORIG_INPUT_DATASET = "New Folder/"
BASE_PATH = ORIG_INPUT_DATASET
TRAIN_PATH = BASE_PATH

#TRAIN_PATH = os.path.sep.join([BASE_PATH, "train/"+dest])
#VAL_PATH = os.path.sep.join([BASE_PATH, "valid/"+dest])
#TEST_PATH = os.path.sep.join([BASE_PATH, "test/"+dest])

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

#filename = [TRAIN_PATH, VAL_PATH, TEST_PATH]

'''for i in filename:
	if not os.path.exists(i):
		print("\n[INFO] creating '{}' directory".format(i))
		os.makedirs(i)'''

imagePaths = list(paths.list_images(ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)

	
    
'''train_images = imagePaths[:int(len(imagePaths)*TRAIN_SPLIT)]
val_images = imagePaths[int(len(imagePaths)*TRAIN_SPLIT):int(len(imagePaths)*(TRAIN_SPLIT+VAL_SPLIT))]
test_images = imagePaths[int(len(imagePaths)*(TRAIN_SPLIT+VAL_SPLIT)):int(len(imagePaths))]'''

'''for i in range(len(train_images)):
    shutil.copy(train_images[i],TRAIN_PATH)'''

p = Augmentor.Pipeline(TRAIN_PATH)
p.invert(0.66)
p.random_brightness(1,0.75,1.25)
p.random_color(1,0.5,2)
p.random_contrast(1,0.5,2)
p.flip_left_right(0.6)
#p.rotate(0.5,20,20)
#additional preprocessing methods for cancer sub-types only!
p.rotate_without_crop(0.5, 20, 20, expand=False, fillcolor=None)
p.crop_random(0.5, 0.75, randomise_percentage_area=False)
p.sample(760)
    
'''for i in range(len(val_images)):
    shutil.copy(val_images[i],VAL_PATH)

for i in range(len(test_images)):
    shutil.copy(test_images[i],TEST_PATH)'''