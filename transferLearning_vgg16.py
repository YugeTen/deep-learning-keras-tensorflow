from helper import *

batch_size = 128
n_classes = 10
n_epoch = 5

deep_img_shape = [224,224]
n_filters = 32
pool_size = 2
kernel_size = 3



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = reshape_for_retrain(x_train, deep_img_shape)
x_test = reshape_for_retrain(x_test, deep_img_shape)


vgg16 = VGG16(include_top=True, weights='imagenet')
vgg16.summary()
model = Sequential(vgg16.layers)
model.add(Dense(768,activation='sigmoid'))
model.add(Dropout(0.0))
model.add(Dense(n_classes,activation='softmax'))



train_model(model, (x_train, y_train), (x_test, y_test), n_classes, batch_size,n_epoch)