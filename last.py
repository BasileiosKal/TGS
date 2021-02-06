#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Conv2D, Input, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Activation, Add, Dropout, DepthwiseConv2D, Flatten, Dense
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow import keras
import tensorflow.keras.backend as K
import gc
from tensorflow.keras import optimizers
from tqdm.notebook import tqdm_notebook

from tensorflow.keras.utils import to_categorical
from tqdm.notebook import tqdm_notebook
import seaborn as sns


# #### Callbacks

# In[2]:


from keras.callbacks import ModelCheckpoint,  CSVLogger, EarlyStopping, ReduceLROnPlateau

savePath="C:/Users/gauss/Documents/TGS/Notebooks/Test_1_checkpoints/saved-{epoch:02d}-{val_my_iou_metric:.2f}.hdf5"

checkPoint = ModelCheckpoint(savePath, monitor='val_my_iou_metric', verbose=1, save_best_only= True, mode='max')
early_stop = EarlyStopping(monitor='val_my_iou_metric', patience=15, verbose=1, mode='max')
reduce_lr_OnPlateau = ReduceLROnPlateau(factor=0.75,monitor='val_my_iou_metric', mode='max', patience=5, min_lr=0.0001, verbose=1)

#CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger('my_logs.csv', separator=',', append=False)
callbacks_list = [early_stop, checkPoint, reduce_lr_OnPlateau]


# #### Loading the data 
# For faster development, we have fixed a validation dataset. For sanity checks and debuging we have the option to load just the first "rows" from the dataset.

# In[3]:


# ------------------------------------------------------------------------------- #
#                              Important Variables                                #
# ------------------------------------------------------------------------------- #
np.random.seed(42)
data_dir = "../../DL_data/competition_data"
validation_data_dir = "../../DL_data/competition_data"
nrows = "all"  # Set to 'all' to load the whole set
load_validation = False  # Only load the validation images and masks??
split_train_test = True  # Split data to train and test sets??
data_augmentation = True  # Augment the data??
# ------------------------------------------------------------------------------- #


# load the ids  and depths of 'nrows' from the training data set
if isinstance(nrows, int) and nrows > 0:
    train_df = pd.read_csv(data_dir + "/train.csv", index_col="id", usecols=[0], nrows=nrows)
    depths_df = pd.read_csv(data_dir + "/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
elif isinstance(nrows, str) and nrows.upper() == "ALL":
    train_df = pd.read_csv(data_dir + "/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv(data_dir + "/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
else:
    raise ValueError("Invalid nrows value")


# Function that loads the ids of 'nrows' from the validation data set
def load_validation_data(data_dir, nrows):
    if isinstance(nrows, int) and nrows > 0:
        valid_ids = pd.read_csv(data_dir + "/Validation_ids.csv", usecols=[1], nrows=nrows)
    elif isinstance(nrows, str) and nrows.upper() == "ALL":
        valid_ids = pd.read_csv(data_dir + "/Validation_ids.csv")
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
    

def global_contrast_normalization(img, s=0.1, lmda=0, epsilon=0.000000001):
    X = img
    X_average = np.mean(X)
    # print('Mean: ', X_average)
    X = X - X_average

    # `su` is here the mean, instead of the sum
    contrast = np.sqrt(lmda + np.mean(X**2))

    X = s * X / max(contrast, epsilon)
    
    return X


print("Loading images...")
train_df["images"] = [np.array(load_img(data_dir + "/train/images/{}.png".format(idx),
                                        color_mode="grayscale")) / 255 for idx in tqdm_notebook(list(ids_to_load))]
print("Loading masks...")
train_df["masks"] = [np.array(load_img(data_dir + "/train/masks/{}.png".format(idx),
                                       color_mode="grayscale")) / 65535 for idx in tqdm_notebook(list(ids_to_load))]

print("done loading images")


# In[ ]:


#del train_df
#gc.collect()


# ### Example of the data

# In[4]:


# printing
print("-------------------------------------------------------------")
id = '7d5c34a95a'
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


# ### Preprossessing the Data

# In[ ]:



# #### Resizeing the images

# In[5]:


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


# #### Calculate the salt coverage for each immage

# In[6]:


train_df["coverage"] = train_df.masks.map(np.sum) / (train_df["masks"][0].shape[0] * train_df["masks"][0].shape[1])


# Generate salt coverage classes
def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

# Plotting the salt coverage classes

# In[13]:


fig, axs = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(train_df["coverage"], bins=10,kde=False, ax=axs[0])
sns.distplot(train_df["coverage_class"], bins=10, kde=False, ax=axs[1])
plt.suptitle("Salt coverage")
axs[0].set_xlabel("Coverage")
axs[1].set_xlabel("Coverage class")


# #### Take a test set stratified based on the salt coverage.
# Sample uniformly for each salt coverage class

# In[7]:


if split_train_test:
    (ids_train, ids_test, x_train, x_test, y_train, y_test,
     cov_train, cov_test, depth_train, depth_test) = train_test_split(train_df.index.values,
                                                                      images_resized, masks_resized,
                                                                      train_df.coverage.values,
                                                                      train_df.z.values,
                                                                      test_size=500,
                                                                      stratify=train_df.coverage_class,
                                                                      random_state=1337)
else:
    x_train = images_resized
    y_train = masks_resized
    ids_train = train_df.index.values
    x_test = np.array([])  # Just to print the x_valid.shape([0]) in the end
    y_test = np.array([])

train_temp_df=pd.DataFrame(data={"images": list(x_train), "masks":list(y_train)}, index=list(ids_train))

print("Train/ Test shape = %d/ %d" % (y_train.shape[0], y_test.shape[0]))


# ##### Recompute the coverage class of the remaining training set to take an additional validation set from the training set, again stratified based on the salt coverage class.

# In[8]:


train_temp_df["coverage"] = train_temp_df.masks.map(np.sum) / (101*101)


# Generate salt coverage classes
def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


train_temp_df["coverage_class"] = train_temp_df.coverage.map(cov_to_class)

fig, axs = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(train_temp_df["coverage"], bins=10,kde=False, ax=axs[0])
sns.distplot(train_temp_df["coverage_class"], bins=10, kde=False, ax=axs[1])
plt.suptitle("Salt coverage")
axs[0].set_xlabel("Coverage")
axs[1].set_xlabel("Coverage class")


# In[9]:


del x_train
del y_train
gc.collect()

temp_images = np.array(train_temp_df.images.tolist()).reshape((-1, resize_to, resize_to, 1))
temp_masks =  np.array(train_temp_df.masks.tolist()).reshape((-1, resize_to, resize_to, 1))


# In[10]:


if split_train_test:
    (x_train, x_valid, y_train, y_valid) = train_test_split(temp_images, temp_masks, 
                                                            test_size=300,
                                                            stratify=train_temp_df["coverage_class"],
                                                            random_state=1337)
else:
    x_train = temp_images
    y_train = temp_masks
    x_valid = np.array([])  # Just to print the x_valid.shape([0]) in the end
    y_valid = np.array([])


del temp_images
del temp_masks
gc.collect()

print("Train/ Valid/ Test split = %d/ %d/ %d" % (x_train.shape[0], x_valid.shape[0], x_test.shape[0]))


# #### Data augmentation
# Fliping the images from left to right and from up to down.

# In[11]:


x_train_temp = np.copy(x_train)
y_train_temp = np.copy(y_train)
if data_augmentation:
    print("fliping left/right")
    x_train = np.append(x_train, [np.fliplr(x) for x in tqdm_notebook(x_train_temp)], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in tqdm_notebook(y_train_temp)], axis=0)
    
    print("fliping up/down")
    x_train = np.append(x_train, [np.flipud(x) for x in tqdm_notebook(x_train_temp)], axis=0)
    y_train = np.append(y_train, [np.flipud(x) for x in tqdm_notebook(y_train_temp)], axis=0)
    
del x_train_temp
del y_train_temp
gc.collect()

print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)


# #### Creating 3 chanels in the input images

# In[12]:


x_train = np.repeat(x_train, 3, axis=3)
x_test = np.repeat(x_test, 3, axis=3)
x_valid = np.repeat(x_valid, 3, axis=3)


# In[13]:


print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_valid shape: ", x_valid.shape)
print("y_valid shape: ", y_valid.shape)
print("x_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)


# In[14]:


del train_temp_df
gc.collect()


# # Model

# In[64]:


# #### -----> IoU metric

def get_iou_vector(A, B):
    # Numpy version
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):

        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

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
        iou = np.floor(max(0, (iou - 0.45) * 20)) / 10

        metric += iou

    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.numpy_function(get_iou_vector, [label, pred > 0.5], tf.float64)


# In[16]:


# #### ------> Additional blocks

# In[20]:


def BatchActivate(x,lambda2):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x =  tf.keras.layers.Dropout(DropoutRatio)(x)
    if lambda2 == True:
        x =  tf.keras.layers.Dropout(DropoutRatio)(x)
    return x


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x, True)
    return x



def block(block_in, expand_filters, project_filters, depthwise_stride=(1, 1), depthwise_padding="same",
          zero_padding=False):
    x = convolution_block(block_in, expand_filters, (1, 1))

    # if zero_padding:
    #    x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
    # x = DepthwiseConv2D((3,3), strides=(1, 1), padding=depthwise_padding)(x)
    # x = BatchActivate(x)

    x = convolution_block(x, project_filters, (3, 3))

    x = convolution_block(x, expand_filters, (1, 1), activation=False)
    # x = BatchNormalization()(x)
    return x


def transpose_block(block_in, to_add, expand_filters, project_filters):
    # x = convolution_block(block_in, expand_filters, (1, 1))
    # x = BatchActivate(x)

    x = Conv2DTranspose(expand_filters, (3, 3), strides=(2, 2), padding="same")(block_in)
    x = concatenate([x, to_add])
    print("after concatenate = ", x.shape)
    # x = BatchActivate(x)

    x = convolution_block(x, expand_filters, (3, 3), activation=False)
    # x = BatchNormalization()(x)
    return x


# In[17]:


cov_model_v2 = keras.models.load_model('cov_model_v2')

input_shape = (128, 128, 3)

#backbone = MobileNetV2(input_shape=input_shape, weights='imagenet', include_top=False)

backbone = cov_model_v2
inputs = backbone.input

# print(backbone.summary())


# In[18]:


start_neurons = 16
DropoutRatio = 0.1

print("START BUILDING MODEL")

# input_layer = Input(input_shape)
# print("BUILD INPUT")

middle_1 = backbone.get_layer('block_13_expand_BN')
middle_1 = middle_1.output
# conv4 = LeakyReLU(alpha=0.1)(conv4)
# pool4 = MaxPooling2D((2, 2))(conv4)
# pool4 = Dropout(0.1)(pool4)
print("middle_1.shape = ", middle_1.shape)

# Middle
filters_m_expand = 128
filters_m_project = 128
convm = Conv2D(filters_m_project, (3, 3), activation=None, padding="same")(middle_1)
convm_BA = BatchActivate(convm, True )

convm_1 = block(convm_BA, filters_m_expand, filters_m_project)
convm_add_1 = Add()([convm_1, convm])

convm_add_1_BA = BatchActivate(convm_add_1, True)
convm_2 = block(convm_add_1_BA, filters_m_expand, filters_m_project)
convm_add_2 = Add()([convm_2, convm_add_1])

convm_add_2 = BatchActivate(convm_add_2, False)
# convm = residual_block(convm,filters_m)
# convm = residual_block(convm,filters_m, True)
print("convm.shape = ", convm_add_2.shape)


# 8 -> 16
print("--------------------------------------------------------")
expand_filters_1 = 2 * 64
project_filters_1 = 64
to_add_1 = backbone.get_layer('block_6_expand_BN')
to_add_1 = to_add_1.output

uconv1_1 = transpose_block(convm_add_2, to_add_1, expand_filters_1, project_filters_1)

#res block 1
uconv1_1_BA = BatchActivate(uconv1_1, True)
uconv1_2 = block(uconv1_1_BA, expand_filters_1, project_filters_1)
uconv1_add_1 = Add()([uconv1_1, uconv1_2])

#res block 2
uconv1_add_1_BA = BatchActivate(uconv1_add_1, True)
uconv1_3 = block(uconv1_add_1_BA, expand_filters_1, project_filters_1)
uconv1_add_2 = Add()([uconv1_add_1, uconv1_3])

#res block 3
uconv1_add_2_BA = BatchActivate(uconv1_add_2, True)
uconv1_4 = block(uconv1_add_2_BA, expand_filters_1, project_filters_1)
uconv1_add_3 = Add()([uconv1_add_2, uconv1_4])

uconv1_add_3_BA = BatchActivate(uconv1_add_3, False)

print("uconv1_1.shape = ", uconv1_1.shape)
print("uconv1_2.shape = ", uconv1_2.shape)
print("uconv1_add_2.shape = ", uconv1_add_2.shape)



# 16 -> 32
print("--------------------------------------------------------")
expand_filters_2 = 2 * 32
project_filters_2 = 32
to_add_2 = backbone.get_layer('block_3_expand_BN')
to_add_2 = to_add_2.output

uconv2_1 = transpose_block(uconv1_add_3_BA, to_add_2, expand_filters_2, project_filters_2)

#res block 1
uconv2_1_BA = BatchActivate(uconv2_1, True)
uconv2_2 = block(uconv2_1_BA, expand_filters_2, project_filters_2)
uconv2_add_1 = Add()([uconv2_1, uconv2_2])

#res block 2
uconv2_add_1_BA = BatchActivate(uconv2_add_1, True)
uconv2_3 = block(uconv2_add_1_BA, expand_filters_2, project_filters_2)
uconv2_add_2 = Add()([uconv2_add_1, uconv2_3])

#res block 3
uconv2_add_2_BA = BatchActivate(uconv2_add_2, True)
uconv2_4 = block(uconv2_add_2_BA, expand_filters_2, project_filters_2)
uconv2_add_3 = Add()([uconv2_add_2, uconv2_4])

uconv2_add_3_BA = BatchActivate(uconv2_add_3, False)

print("uconv2_1.shape = ", uconv2_1.shape)
print("uconv2_2.shape = ", uconv2_2.shape)
print("uconv2_add_2.shape = ", uconv2_add_2.shape)


# 32 -> 64
print("--------------------------------------------------------")
expand_filters_3 = 2 * 16
project_filters_3 = 16
to_add_3 = backbone.get_layer('expanded_conv_project_BN')
to_add_3 = to_add_3.output

uconv3_1 = transpose_block(uconv2_add_3_BA, to_add_3, expand_filters_3, project_filters_3)

#res block 1
uconv3_1_BA = BatchActivate(uconv3_1, True)
uconv3_2 = block(uconv3_1_BA, expand_filters_3, project_filters_3)
uconv3_add_1 = Add()([uconv3_1, uconv3_2])

#res block 2
uconv3_add_1_BA = BatchActivate(uconv3_add_1, True)
uconv3_3 = block(uconv3_add_1_BA, expand_filters_3, project_filters_3)
uconv3_add_2 = Add()([uconv3_add_1, uconv3_3])

#res block 3
uconv3_add_2_BA = BatchActivate(uconv3_add_2, True)
uconv3_4 = block(uconv3_add_2_BA, expand_filters_3, project_filters_3)
uconv3_add_3 = Add()([uconv3_add_2, uconv3_4])

uconv3_add_3_BA = BatchActivate(uconv3_add_3, False)

print("uconv3_1.shape = ", uconv3_1.shape)
print("uconv3_2.shape = ", uconv3_2.shape)
print("uconv3_add_2.shape = ", uconv3_add_2.shape)


# 64 -> 128
print("--------------------------------------------------------")
filters_5 = 2*8
deconv0 = Conv2DTranspose(filters_5, (3, 3), strides=(2, 2), padding="same")(uconv3_add_3_BA)
print("deconv0 = ", deconv0.shape)
uconv0 = Conv2D(filters_5, (3, 3), activation=None, padding="same")(deconv0)

#res block 1
uconv0_BA = BatchActivate(uconv0, True)
uconv0_1 = block(uconv0_BA, filters_5, 8)
uconv0_add_1 = Add()([uconv0, uconv0_1])

#res block 2
uconv0_add_1_BA = BatchActivate(uconv0_add_1, True)
uconv0_2 = block(uconv0_add_1_BA, filters_5, 8)
uconv0_add_2 = Add()([uconv0_2, uconv0_add_1])


uconv0_add_2_BA = BatchActivate(uconv0_add_2, True)
uconv0_3 = block(uconv0_add_1_BA, filters_5, 8)
uconv0_add_3 = Add()([uconv0_3, uconv0_add_2])

uconv0_add_3_BA = BatchActivate(uconv0_add_3, False)

output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv0_add_3_BA)
output_layer = Activation('sigmoid')(output_layer_noActi)
print("output_layer = ", output_layer.shape)


# In[19]:


from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


# In[20]:


model = Model(inputs=[inputs], outputs=[output_layer])
print("BUILD MODEL")
adam_optimizer = optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=adam_optimizer, loss=bce_dice_loss, metrics=[my_iou_metric]) # 'binary_crossentropy'
print("COMPILE")
model.summary()


# In[21]:


epochs = 50
batch_len = 10
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, shuffle=True, batch_size=batch_len, callbacks = callbacks_list)


# In[22]:


model_save_dir = "C:/Users/gauss/Documents/TGS/Notebooks/Saved_models/test_1_latest"

def plot_history(hs, epochs, metric):
    print()
    plt.style.use('dark_background')
    plt.rcParams['figure.figsize'] = [15, 8]
    plt.rcParams['font.size'] = 16
    plt.clf()
    for label in hs:
        print(len(hs[label].history['loss']))
        plt.plot(hs[label].history[metric], label='{0:s} train {1:s}'.format(label, metric), linewidth=2)
        plt.plot(hs[label].history['val_{0:s}'.format(metric)], label='{0:s} validation {1:s}'.format(label, metric), linewidth=2)
    
    
    
    x_ticks = np.arange(0, epochs + 1, epochs / 10)
    x_ticks [0] += 1
    plt.xticks(x_ticks)
    plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss' if metric=='loss' else 'Accuracy')
    plt.legend()
    plt.savefig('{0:s}/fig_{1:s}.png'.format(model_save_dir, metric))
    plt.show()


# In[23]:


plot_history(hs={'': history}, epochs=epochs, metric='loss')
plot_history(hs={'': history}, epochs=epochs, metric='my_iou_metric')


# ### Saving the results

# In[70]:


#loss_history = np.transpose(np.array(history.history["loss"]))
#IoU_history = np.transpose(np.array(history.history["my_iou_metric"]))
loss_history = np.array(history.history["loss"]).reshape((50, 1))
IoU_history = (np.array(history.history["my_iou_metric"])).reshape((50 , 1))
val_loss_history = np.array(history.history["val_loss"]).reshape((50 , 1))
val_IoU_history = (np.array(history.history["val_my_iou_metric"])).reshape((50 , 1))
#numpy_history = np.array([(i,j) for (i, j) in zip(loss_history, IoU_history)], dtype=np.float32).reshape(40, 2)

print(IoU_history)
np.savetxt(model_save_dir+"/logs/IoU.txt", 
           IoU_history, delimiter=",")
np.savetxt(model_save_dir+"/logs/Loss.txt", 
           loss_history, delimiter=",")
np.savetxt(model_save_dir+"/logs/val_IoU.txt", 
           val_IoU_history, delimiter=",")
np.savetxt(model_save_dir+"/logs/val_Loss.txt", 
           val_loss_history, delimiter=",")


# In[71]:


best_model = keras.models.load_model("C:/Users/gauss/Documents/TGS/Notebooks/Test_1_checkpoints\saved-45-0.75.hdf5", 
                                    custom_objects={"my_iou_metric": my_iou_metric, "bce_dice_loss": bce_dice_loss}, compile=True)


# In[ ]:


#del model
#gc.collect()


# In[ ]:


#model = Model(inputs=[inputs], outputs=[output_layer])
print("BUILD MODEL")
#adam_optimizer = optimizers.Adam(learning_rate=0.006)
best_model.compile(optimizer=adam_optimizer, loss=bce_dice_loss, metrics=[my_iou_metric])
print("COMPILE")
#best_model.summary()


# In[ ]:


#epochs = 40
#batch_len = 8
#history2 = best_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, shuffle=True, batch_size=batch_len, callbacks = callbacks_list)


# In[ ]:


#del best_model
#gc.collect()


# In[ ]:





# In[73]:


best_model.evaluate(x=x_valid, y=y_valid, batch_size=1)


# ### Applying CRF

# In[ ]:





# In[72]:


x_testing_set = x_valid
y_testing_set = y_valid
model_to_use = best_model

predictions = model_to_use.predict(x_testing_set)

true_count = 0
sum_iou = 0
# mask = y_testing_set[0]
for (img, pred_mask, mask) in zip(x_testing_set, predictions, y_testing_set):
    # print(mask.shape)
    
    pred_mask = pred_mask.reshape(1, pred_mask.shape[0], pred_mask.shape[1], pred_mask.shape[2])
    mask = mask.reshape(1, mask.shape[0], mask.shape[1], mask.shape[2])
    
    metric = get_iou_vector(mask, pred_mask>0.5)
    # print("metric = ", metric)
    
    sum_iou += metric
    if metric>0.6:
        true_count+=1

print("percentaile true_count = ", true_count/len(x_testing_set))
print("percentaile sum_iou = ", sum_iou/len(x_testing_set))


# In[74]:


#del predictions
#gc.collect()


# In[ ]:


#K.clear_session()
#gc.collect()
#del backbone
#del model
#del best_model
#del train_df
#del x_train
#del x_test
#del x_valid
#del y_train
#del y_test
#del y_valid
#gc.collect()


# In[ ]:




