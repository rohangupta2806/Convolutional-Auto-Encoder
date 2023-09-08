import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.applications.vgg16 import VGG16
mod = VGG16(include_top=False, weights='imagenet')
mod.trainable = False
from tensorflow.keras import backend as K

data_path = r'/home/astr211/gal_hsc_128x128'
clean_data = utils.image_dataset_from_directory(data_path, shuffle=False, image_size=(128, 128), batch_size=1, smart_resize=True)

data_path = r'/home/astr211/gal_sdss_128x128'
noisy_data = utils.image_dataset_from_directory(data_path, shuffle=False, image_size=(128, 128), batch_size=1, smart_resize=True)

def generator(array):
    for el in array:
        yield el

def get_img_arrs(gen1, gen2, bound1 = 0.8, bound2 = 0.9): 
    g1 = generator(gen1)
    g2 = generator(gen2)
    
    test_arr1, test_arr2 = [], []
    train_arr1, train_arr2 = [], []
    val_arr1, val_arr2 = [], []
    while True:
        try: # try to read an image
            image1 = next(g1)
            image1 = ((image1[0])[0].numpy()).astype('int32')
            r = np.random.uniform(0., 1.)
            if r <= bound1:
                train_arr1.append(image1/255)
            elif r <= bound2:
                val_arr1.append(image1/255)
            else:
                test_arr1.append(image1/255)
        except: # if file ended, above will generate error
            break # exit the loop
    
        image2 = next(g2)
        image2 = ((image2[0])[0].numpy()).astype('int32')
        if r <= bound1:
            train_arr2.append(image2/255)
        elif r <= bound2:
            val_arr2.append(image2/255)
        else:
            test_arr2.append(image2/255)
    return np.array(train_arr1), np.array(test_arr1), np.array(val_arr1), np.array(train_arr2), np.array(test_arr2), np.array(val_arr2)
    
clean_train, clean_test, clean_val, noisy_train, noisy_test, noisy_val = get_img_arrs(clean_data, noisy_data)

def display3(array1, array2, array3):
    """
    Displays 10 random images from 3 arrays
    """
    n=10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    images3 = array3[indices, :]
    plt.figure(figsize=(20,6))
    for i, (image1, image2, image3) in enumerate(zip(images1, images2, images3)):
        ax = plt.subplot(3, n, i+1)
        plt.imshow(image1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(image2)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(image3)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("Best_Model_200.png")
    
def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()
    
# Display the original and noisy images
display(clean_train, noisy_train)

input = layers.Input(shape=(128, 128, 3))

# Encoder
x1 = layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l1(10e-10))(input)
x2 = layers.Conv2D(256, (3, 3), activation ="relu", padding="same", kernel_regularizer=regularizers.l1(10e-10))(x1)
x3 = layers.MaxPooling2D((2, 2), padding="same")(x2)

x4 = layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l1(10e-10))(x3)
x5 = layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l1(10e-10))(x4)
x6 = layers.MaxPooling2D((2, 2), padding="same")(x5)

encoded = layers.Conv2D(256, (3, 3), activation = "relu", padding = "same", kernel_regularizer=regularizers.l1(10e-10))(x6)

# Decoder
x7 = layers.Conv2DTranspose(256, (3, 3), activation="relu", padding="same", strides = 2)(encoded)
x8 = layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l1(10e-10))(x7)
x9 = layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l1(10e-10))(x8)
x10 = layers.Add()([x5, x9])

x11 = layers.Conv2DTranspose(256, (3, 3), padding = "same", strides = 2, activation = "relu")(x10)
x12 = layers.Conv2D(256, (3, 3), activation = "relu", padding = "same", kernel_regularizer=regularizers.l1(10e-10))(x11)
x13 = layers.Conv2D(256, (3, 3), activation = "relu", padding = "same", kernel_regularizer=regularizers.l1(10e-10))(x12)
x14 = layers.Add()([x2, x13])

decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same", kernel_regularizer = regularizers.l1(10e-10))(x14)

def VGGloss(y_true, y_pred):
    pred = K.concatenate([y_pred])
    true = K.concatenate([y_true])
    vggmodel = mod
    f_p = vggmodel(pred)
    f_t = vggmodel(true)
    return K.mean(K.square(f_p - f_t))

def MSE_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def total_loss(y_true, y_pred):
    return VGGloss(y_true, y_pred) + MSE_loss(y_true, y_pred)

# Autoencoder
autoencoder = Model(input, decoded)
autoencoder.compile("Adam", loss=total_loss)
autoencoder.summary()

autoencoder.fit(
    x=noisy_train,
    y=clean_train,
    epochs=200,
    batch_size=1,
    shuffle=True,
    validation_data=(noisy_val, clean_val),
)

autoencoder.save('Best_Model_200')

predictions = autoencoder.predict(noisy_test)
display3(noisy_test, predictions, clean_test)




