import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.layers import Conv2D, Input, Conv2DTranspose, MaxPooling2D, concatenate
from skimage.transform import resize
from sklearn.model_selection import train_test_split



""" ============================================================================== """
""" ------------------------------ Loading Data  --------------------------------- """
""" ============================================================================== """

data_dir = "C:/Users/Tominho/Desktop/Projects/DL/TGS/competition_data/competition_data"

train_df = pd.read_csv(data_dir+"/train.csv", index_col="id", usecols=[0], nrows=100)
depths_df = pd.read_csv(data_dir+"/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

train_df["images"] = [np.array(load_img(data_dir+"/train/images/{}.png".format(idx),
                                        color_mode = "grayscale"))/ 255 for idx in (train_df.index)]

train_df["masks"] = [np.array(load_img(data_dir+"/train/masks/{}.png".format(idx),
                                       color_mode = "grayscale"))/ 255 for idx in (train_df.index)]

print("done loading images...")
""" ============================================================================ """
""" ------------------------------  Data Stats --------------------------------- """
""" ============================================================================ """

count = 0


# -------------> data example
# printing
print("-------------------------------------------------------------")
id = 4
image = train_df["images"][id]
print(image)
print(image.shape)
print("-------------------------------------------------------------")
mask = train_df["masks"][id]
print(mask)
print(mask.shape)


# Plotting
def plot_image_example():
    fig_imgshow, (axs_mask, axs_img) = plt.subplots(1, 2)
    fig_imgshow.suptitle("Data example")
    axs_img.imshow(np.dstack((image, image, image)))  # interpolation='nearest'
    axs_img.set(title="image")
    tmp = np.squeeze(mask).astype(np.float32)
    axs_mask.imshow(np.dstack((tmp, tmp, tmp)))
    axs_mask.set(title="mask")


plot_image_example()

""" =============================================================================== """
""" ------------------------------  Prepossessing --------------------------------- """
""" =============================================================================== """
# -------------> resize to a pow of 2
# either pad with zeros or resize with interpolation
resize_to = 128
original_size = 101


def upsample(original_img):
    if resize_to == original_size:
        return original_img
    return resize(original_img, (resize_to, resize_to), mode='constant', preserve_range=True)


def pad_zeros(array):
    padded_image = np.zeros(shape=(resize_to, resize_to))
    padded_image[13:114, 13:114] = array
    return padded_image


resizing_function_to_use = upsample

images_resized = np.array(train_df.images.map(resizing_function_to_use).tolist()).reshape((-1, resize_to, resize_to, 1))
masks_resized = np.array(train_df.masks.map(resizing_function_to_use).tolist()).reshape((-1, resize_to, resize_to, 1))


# Plotting
def plot_reshape_example():
    fig_reshape, (axs_reshape_mask, axs_reshape_img) = plt.subplots(1, 2)
    fig_reshape.suptitle("Reshaped data example")
    axs_reshape_img.set(title="Reshaped image")
    axs_reshape_mask.set(title="Reshaped mask")
    axs_reshape_img.imshow(images_resized[id])
    axs_reshape_mask.imshow(masks_resized[id])


plot_reshape_example()

# -------------> split train/dev
(ids_train, ids_valid,
x_train, x_valid,
y_train, y_valid
# cov_train, cov_test,
#depth_train, depth_test
 ) = train_test_split(
    train_df.index.values,
    images_resized,
    masks_resized,
    # train_df.coverage.values,
    # train_df.z.values,
    test_size=0.2,
    # stratify=train_df.coverage_class,
    random_state=1337)


""" ======================================================================= """
""" ------------------------------  U-Net --------------------------------- """
""" ======================================================================= """
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

print("START BUILDING MODEL")
s = Input((128, 128, 1))
print("BUILD INPUT")


c1 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal' )(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
p1 = MaxPooling2D((2, 2))(c1)
print("BUILD C1")

c2 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(p2)
c3 = tf.keras.layers.Dropout(0.1)(c3)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(p3)
c4 = tf.keras.layers.Dropout(0.1)(c4)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(p4)
c5 = tf.keras.layers.Dropout(0.1)(c5)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(u6)
c6 = tf.keras.layers.Dropout(0.1)(c6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(u7)
c7 = tf.keras.layers.Dropout(0.1)(c7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(c9)

output = Conv2D(1, (1, 1), activation='sigmoid')(c9)
print("BUILD OUTPUT")

model = Model(inputs=[s], outputs=[output])
print("BUILD MODEL")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
print("COMPILE")
model.summary()

epochs = 25
history = model.fit(x_train, y_train, epochs=epochs)

