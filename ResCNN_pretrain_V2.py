import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Conv2D, Input, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Activation, Add, Dropout
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


import tensorflow.keras.backend as K
import gc
from keras import optimizers


print("tensorflow version: ", tf.__version__)


# #### Plotting model results and cleaning up

def plot_history(hs, epochs, metric):
    print()
    plt.style.use('dark_background')
    plt.rcParams['figure.figsize'] = [15, 8]
    plt.rcParams['font.size'] = 16
    plt.clf()
    for label in hs:
        print(len(hs[label].history['loss']))
        plt.plot(hs[label].history[metric], label='{0:s} train {1:s}'.format(label, metric), linewidth=2)
        # plt.plot(hs[label].history['val_{0:s}'.format(metric)], label='{0:s} validation {1:s}'.format(label, metric), linewidth=2)
    x_ticks = np.arange(0, epochs + 1, epochs / 10)
    x_ticks[0] += 1
    plt.xticks(x_ticks)
    plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss' if metric == 'loss' else 'Accuracy')
    plt.legend()
    plt.show()


def clean_up(model):
    K.clear_session()
    del model
    gc.collect()


# # LOAD DATA
# ------------------------------------------------------------------------------- #
#                              Important Variables                                #
# ------------------------------------------------------------------------------- #
data_dir = "C:/Users/G/T/X/S"
validation_data_dir = "C:/Users/G/T/X/S"
nrows = "all"                 # Set to 'all' to load the whole set
load_validation = True      # Only load the validation images and masks??
split_train_test = False    # Split data to train and test sets??
data_augmentation = False   # Augment the data??
# ------------------------------------------------------------------------------- #


#load the ids  and depths of 'nrows' from the training data set
if isinstance(nrows, int) and nrows>0:
    train_df = pd.read_csv(data_dir+"/train.csv", index_col="id", usecols=[0], nrows=nrows)
    depths_df = pd.read_csv(data_dir+"/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
elif isinstance(nrows, str) and nrows.upper() == "ALL":
    train_df = pd.read_csv(data_dir+"/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv(data_dir+"/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
else:
    raise ValueError("Invalid nrows value")
        

        
# Function that loads the ids of 'nrows' from the validation data set
def load_validation_data(data_dir, nrows):
    if isinstance(nrows, int) and nrows>0:
        valid_ids = pd.read_csv(data_dir+"/Validation_ids.csv", usecols=[1], nrows=nrows)
    elif isinstance(nrows, str) and nrows.upper() == "ALL":
        valid_ids = pd.read_csv(data_dir+"/Validation_ids.csv")
    else:
        raise ValueError("Invalid nrows value")
    return valid_ids.ids


if load_validation:
    ids_to_load = load_validation_data(data_dir, nrows)
    index_list = list(train_df.index)
    ids_list = [index_list.index(i) for i in ids_to_load]
    train_df = train_df.iloc[ids_list]
else:
    ids_to_load = train_df.index
    
print("Loading images...")
train_df["images"] = [np.array(load_img(data_dir+"/train/images/{}.png".format(idx),
                                        color_mode = "grayscale"))/255 for idx in (list(ids_to_load))]
print("Loading masks...")
train_df["masks"] = [np.array(load_img(data_dir+"/train/masks/{}.png".format(idx),
                                       color_mode = "grayscale"))/65535 for idx in (list(ids_to_load))]

print("done loading images")


# -------------> data example
# printing
print("-------------------------------------------------------------")
id = '2a070f3dc6'
id_index = np.where(train_df.index == id)
id_index = id_index[0][0]
print("id_index = ", id_index)
image = train_df["images"][id]
print(image)
print(image.shape)
print("-------------------------------------------------------------")
mask = train_df["masks"][id]
print(mask)
print(mask.shape)


def plot_image_example():
    fig_imgshow, (axs_mask, axs_img) = plt.subplots(1, 2)
    fig_imgshow.suptitle("Data example")
    axs_img.imshow(np.dstack((image, image, image)))  # interpolation='nearest'
    axs_img.set(title="image")
    tmp = np.squeeze(mask).astype(np.float32)
    axs_mask.imshow(np.dstack((tmp, tmp, tmp)))
    axs_mask.set(title="mask")


plot_image_example()


# # **Prepossessing Data**

# #### ------> Resize to a pow of 2

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


resizing_function_to_use = pad_zeros

images_resized = np.array(train_df.images.map(resizing_function_to_use).tolist()).reshape((-1, resize_to, resize_to, 1))
masks_resized = np.array(train_df.masks.map(resizing_function_to_use).tolist()).reshape((-1, resize_to, resize_to, 1))



# Plotting
def plot_reshape_example():
    fig_reshape, (axs_reshape_mask, axs_reshape_img) = plt.subplots(1, 2)
    fig_reshape.suptitle("Reshaped data example")
    axs_reshape_img.set(title="Reshaped image")
    axs_reshape_mask.set(title="Reshaped mask")
    axs_reshape_img.imshow(images_resized[id_index], cmap='gray')
    axs_reshape_mask.imshow(masks_resized[id_index], cmap='gray')


plot_reshape_example()


# #### -----> Calculating the salt coverage

train_df["coverage"] = train_df.masks.map(np.sum) / (train_df["masks"][0].shape[0]*train_df["masks"][0].shape[1])

# Generate salt coverage classes
def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
        
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)


# Plotting the salt coverage classes

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
n_bins = 20
axs[0].hist(train_df.coverage, bins=n_bins)
axs[1].hist(train_df.coverage, bins=10)

plt.suptitle("Salt coverage")
axs[0].set_xlabel("Coverage")
axs[1].set_xlabel("Coverage class")


# #### ------> Split train/dev

if split_train_test:
    (ids_train, ids_valid, x_train, x_valid, y_train, y_valid,
     cov_train, cov_test, depth_train, depth_test) = train_test_split(train_df.index.values, 
                                                                      images_resized, masks_resized, 
                                                                      train_df.coverage.values, 
                                                                      train_df.z.values, 
                                                                      test_size=0.2, 
                                                                      stratify=train_df.coverage_class,
                                                                      random_state=1337)
else:
    x_train = images_resized
    y_train = masks_resized
    x_valid = np.array([])  # Just to print the x_valid.shape([0]) in the end
    y_valid = np.array([]) 

print("Train/ Valid shape = %d/ %d"%(x_train.shape[0], x_valid.shape[0]))


#print(ids_valid.shape)
#ids = {"ids": list(ids_valid)}
#df = pd.DataFrame(ids) # .to_csv("../DL_data/competition_data/Validation_ids.csv")
#df.to_csv("../DL_data/competition_data/Validation_ids.csv")


# #### ------> Data augmentation
if data_augmentation:
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)


x_train = np.repeat(x_train,3,axis=3)

print("x_rain shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)


# # U-Net

# #### ------> Additional blocks

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x


# #### -----> IoU metric

def get_iou_vector(A, B):
    # Numpy version    
    batch_size = A.shape[0]
    #print("A shape: ", A.shape)
    #print("B shape: ", B.shape)
    metric = 0.0
    for batch in range(batch_size):
        
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)
        
        #print("True: ", t)
        #print("Predicted: ", p)
        #print("----------------------")
        #print("True sum: ", true)
        #print("Predicted sum: ", pred)
        
        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue
        
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        
        # Scale the iou function: iou -> 2*iou - 0.9
        # This will map the threshold values [0.5, 0.55, 0.6, ...., 0.9] to the
        # values [0.1, 0.2, 0.3, ....., 0.9]. Then multiply 2*iou - 0.9 by 10 and floor.
        # This will map the iou values to the number of thresholds that the iou sutisfies
        # witch is what we are intreasted in in the first playse.
        # For example if iou=0.552 then floor(2*iou - 0.9)*10 = 2. Since
        # eveery value <0 means an original value<0.5 we take the max with 0. Finaly we 
        # devide by 10 = |threshods|.
        iou = np.floor(max(0, (iou - 0.45)*20)) / 10
        
        metric += iou
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.compat.v1.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


# ##### -----> Model definition
# --- pretrained model
input_shape = (128, 128, 3)

backbone = MobileNetV2(input_shape=input_shape,weights='imagenet',include_top=False)
inputs = backbone.input

print(backbone.summary())

# --- 2nd part of unet with res blocks
start_neurons = 16
DropoutRatio = 0.5

print("START BUILDING MODEL")

# Middle
middle_1 = backbone.get_layer('block_13_expand_BN')
middle_1 = middle_1.output
print(middle_1.shape)
    
filters_m = 384
convm = Conv2D(filters_m, (3, 3), activation=None, padding="same")(middle_1)
convm = residual_block(convm,filters_m)
convm = residual_block(convm,filters_m, True)
print(convm.shape)
    
# 8 -> 8
deconv4 = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding="same")(convm)
print("deconv4 = ",deconv4.shape)
to_add = backbone.get_layer("block_10_expand_BN")
to_add = to_add.output
uconv4 = concatenate([deconv4, to_add])
uconv4 = Dropout(DropoutRatio)(uconv4)
    
uconv4 = Conv2D(64, (3, 3), activation=None, padding="same")(uconv4)
uconv4 = residual_block(uconv4,64)
uconv4 = residual_block(uconv4,64, True)

print("uconv4.shape = ", uconv4.shape)

# 8 -> 16
#deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
filters_3 = 32
deconv3 = Conv2DTranspose(filters_3, (3, 3), strides=(2, 2), padding="same")(uconv4)
to_add_3 = backbone.get_layer("block_6_expand_BN")
to_add_3 = to_add_3.output
uconv3 = concatenate([deconv3, to_add_3])  
uconv3 = Dropout(DropoutRatio)(uconv3)
    
uconv3 = Conv2D(filters_3, (3, 3), activation=None, padding="same")(uconv3)
uconv3 = residual_block(uconv3,filters_3)
uconv3 = residual_block(uconv3,filters_3, True)


# 16 -> 32
filters_3 = 24
deconv2 = Conv2DTranspose(filters_3, (3, 3), strides=(2, 2), padding="same")(uconv3)
print("deconv2 = ", deconv2.shape)
to_add_4 = backbone.get_layer("block_3_expand_BN")
to_add_4 = to_add_4.output
uconv2 = concatenate([deconv2, to_add_4])
print("uconv2 after concatenate = ", uconv2.shape)

uconv2 = Dropout(DropoutRatio)(uconv2)
uconv2 = Conv2D(filters_3, (3, 3), activation=None, padding="same")(uconv2)
uconv2 = residual_block(uconv2,filters_3)
uconv2 = residual_block(uconv2,filters_3, True)


# 32 -> 64
filters_4 = 16
deconv1 = Conv2DTranspose(filters_4, (3, 3), strides=(2, 2), padding="same")(uconv2)
to_add_5 = backbone.get_layer("expanded_conv_project_BN")
to_add_5 = to_add_5.output
uconv1 = concatenate([deconv1, to_add_5])
    
uconv1 = Dropout(DropoutRatio)(uconv1)
uconv1 = Conv2D(filters_4, (3, 3), activation=None, padding="same")(uconv1)

uconv1 = residual_block(uconv1,filters_4)
uconv1 = residual_block(uconv1,filters_4, True)


# 64 -> 128
filters_5 = 8
deconv0 = Conv2DTranspose(filters_5, (3, 3), strides=(2, 2), padding="same")(uconv1)
    
uconv0 = Dropout(DropoutRatio)(deconv0)
uconv0 = Conv2D(filters_5, (3, 3), activation=None, padding="same")(uconv0)
uconv0 = residual_block(uconv0,filters_5)
uconv0 = residual_block(uconv0,filters_5, True)


output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv0)
output_layer =  Activation('sigmoid')(output_layer_noActi)
print("output_layer = ", output_layer.shape)



model = Model(inputs=[inputs], outputs=[output_layer])
print("BUILD MODEL")
adam_optimizer = optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=[my_iou_metric])
print("COMPILE")
model.summary()


# #### -----> Training
def ModelTraining(model, x_train, y_train):
    epochs = 5
    batch_len = 4
    history = model.fit(x_train, y_train, epochs=epochs, shuffle=True, batch_size=batch_len)

    plot_history(hs={'U-Net': history}, epochs=epochs, metric='loss')
    plot_history(hs={'U-Net': history}, epochs=epochs, metric='my_iou_metric')

    return history


clean_up(model)
