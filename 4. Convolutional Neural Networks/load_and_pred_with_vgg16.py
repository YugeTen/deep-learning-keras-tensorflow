from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import os

vgg16 = VGG16(include_top=True, weights='imagenet')
vgg16.summary()

IMAGENET_FOLDER = 'imgs/imagenet'
os.system('ls imgs/imagenet')

from keras.preprocessing import image
import numpy as np

# load & preprocess one image and predict with vgg
img_path = os.path.join(IMAGENET_FOLDER, 'strawberry_1.jpeg')
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
plt.imshow(img)
plt.show()
preds = vgg16.predict(x)
print ('Predicted:', decode_predictions(preds))