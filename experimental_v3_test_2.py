import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, \
    Activation, Add, Dropout, DepthwiseConv2D
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import elasticdeform
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow.keras.backend as K
import gc
from keras import optimizers

# ------------------------------------------------------------------------------- #
#                                   Callbacks                                     #
# ------------------------------------------------------------------------------- #
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau

savePath = "C:/Users/Tominho/Desktop/PyCharm_project/saved-{epoch:02d}-{my_iou_metric:.2f}.hdf5"
checkPoint = ModelCheckpoint(savePath, monitor='my_iou_metric', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='my_iou_metric', patience=10, verbose=1)
reduce_lr_OnPlateau = ReduceLROnPlateau(factor=0.1, monitor='my_iou_metric', patience=2, min_lr=0.0001, verbose=1)
# CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger('my_logs.csv', separator=',', append=False)
callbacks_list = [checkPoint, early_stop, log_csv, reduce_lr_OnPlateau]
# ------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------- #
#                              Important Variables                                #
# ------------------------------------------------------------------------------- #
np.random.seed(42)
data_dir = "C:/Users/Tominho/Desktop/DL/TGS/competition_data/competition_data"
validation_data_dir = "C:/Users/Tominho/Desktop/DL/TGS"
nrows = "all"  # Set to 'all' to load the whole set
load_validation = True  # Only load the validation images and masks??
split_train_test = False  # Split data to train and test sets??
data_augmentation = True  # Augment the data??


# ------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------- #
#                              Function to distort image                          #
# ------------------------------------------------------------------------------- #
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    #     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


# ------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------- #
#          load the ids  and depths of 'nrows' from the training data set         #
# ------------------------------------------------------------------------------- #

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


# ------------------------------------------------------------------------------- #
#                load the ids of 'nrows' from the validation data set             #
# ------------------------------------------------------------------------------- #

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
# ------------------------------------------------------------------------------- #


print("Loading images...")
train_df["images"] = [np.array(load_img(data_dir + "/train/images/{}.png".format(idx),
                                        color_mode="grayscale")) / 255 for idx in list(ids_to_load)]
print("Loading masks...")
train_df["masks"] = [np.array(load_img(data_dir + "/train/masks/{}.png".format(idx),
                                       color_mode="grayscale")) / 255 for idx in list(ids_to_load)]

print("done loading images")


def plot_image_example():
    fig_imgshow, (axs_mask, axs_img) = plt.subplots(1, 2)
    fig_imgshow.suptitle("Data example")
    axs_img.imshow(np.dstack((image, image, image)))  # interpolation='nearest'
    axs_img.set(title="image")
    tmp = np.squeeze(mask).astype(np.float32)
    axs_mask.imshow(np.dstack((tmp, tmp, tmp)))
    axs_mask.set(title="mask")


# plot_image_example()

# ------------------------------------------------------------------------------- #
#                                  pad with zeros                                 #
# ------------------------------------------------------------------------------- #

resize_to = 128
original_size = 101


def pad_zeros(array):
    padded_image = np.zeros(shape=(resize_to, resize_to))
    padded_image[13:114, 13:114] = array
    return padded_image


resizing_function_to_use = pad_zeros

images_resized = np.array(train_df.images.map(resizing_function_to_use).tolist()).reshape((-1, resize_to, resize_to, 1))
masks_resized = np.array(train_df.masks.map(resizing_function_to_use).tolist()).reshape((-1, resize_to, resize_to, 1))


# ------------------------------------------------------------------------------- #


# Plotting
def plot_reshape_example():
    fig_reshape, (axs_reshape_mask, axs_reshape_img) = plt.subplots(1, 2)
    fig_reshape.suptitle("Reshaped data example")
    axs_reshape_img.set(title="Reshaped image")
    axs_reshape_mask.set(title="Reshaped mask")
    axs_reshape_img.imshow(images_resized[id_index], cmap='gray')
    axs_reshape_mask.imshow(masks_resized[id_index], cmap='gray')


# plot_reshape_example()


# ------------------------------------------------------------------------------- #
#                        Generate salt coverage classes                           #
# ------------------------------------------------------------------------------- #

train_df["coverage"] = train_df.masks.map(np.sum) / (train_df["masks"][0].shape[0] * train_df["masks"][0].shape[1])


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

# Plotting the salt coverage classes

# In[13]:


# TO_DO: Change that to use matplotlib
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
n_bins = 20
axs[0].hist(train_df.coverage, bins=n_bins)
axs[1].hist(train_df.coverage, bins=10)

plt.suptitle("Salt coverage")
axs[0].set_xlabel("Coverage")
axs[1].set_xlabel("Coverage class")
# ------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------- #
#                        Split train/dev                                          #
# ------------------------------------------------------------------------------- #

if split_train_test == False:
    (ids_train, ids_valid, x_train, x_valid, y_train, y_valid,
     cov_train, cov_test, depth_train, depth_test) = train_test_split(train_df.index.values,
                                                                      images_resized, masks_resized,
                                                                      train_df.coverage.values,
                                                                      train_df.z.values,
                                                                      test_size=0.1,
                                                                      stratify=train_df.coverage_class,
                                                                      random_state=1337)
else:
    x_train = images_resized
    y_train = masks_resized
    x_valid = np.array([])  # Just to print the x_valid.shape([0]) in the end
    y_valid = np.array([])

print("Train/ Valid shape = %d/ %d" % (x_train.shape[0], x_valid.shape[0]))
# ------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------- #
#                              elastic deformation                                #
# ------------------------------------------------------------------------------- #

if data_augmentation:
    x_train = np.append(x_train, [elasticdeform.deform_random_grid(x, 128 * 2, 10) for x in x_train], axis=0)
    y_train = np.append(y_train, [elasticdeform.deform_random_grid(y, 128 * 2, 10) for y in y_train], axis=0)

x_train = np.repeat(x_train, 3, axis=3)

print("x_rain shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)


# ------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------- #
#                                     IoU metric                                  #
# ------------------------------------------------------------------------------- #

def get_iou_vector(A, B):
    # Numpy version
    batch_size = A.shape[0]
    # print("A shape: ", A.shape)
    # print("B shape: ", B.shape)
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

    # take the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.numpy_function(get_iou_vector, [label, pred > 0.5], tf.float64)


# ------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------- #
#                              Additional blocks                                  #
# ------------------------------------------------------------------------------- #


def BatchActivate(x, lambda2):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if lambda2 == True:
        x = tf.keras.layers.Dropout(DropoutRatio)(x)
    return x


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x, False)
    return x


def residual_block(blockInput, num_filters=16, batch_activate=False):
    x = BatchActivate(blockInput, False)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x, False)
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


# ------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------- #
#                                Model definition                                 #
# ------------------------------------------------------------------------------- #
input_shape = (128, 128, 3)

backbone = MobileNetV2(input_shape=input_shape, weights='imagenet', include_top=False)
inputs = backbone.input

# print(backbone.summary())

start_neurons = 16
DropoutRatio = 0.1

print("START BUILDING MODEL")

# input_layer = Input(input_shape)
# print("BUILD INPUT")

middle_1 = backbone.get_layer('block_13_expand_BN')
middle_1 = middle_1.output
print("middle_1.shape = ", middle_1.shape)

# Middle
filters_m_expand = 256
filters_m_project = 256
convm = Conv2D(filters_m_project, (3, 3), activation=None, padding="same")(middle_1)
convm_BA = BatchActivate(convm, True)

convm_1 = block(convm_BA, filters_m_expand, filters_m_project)
convm_add_1 = Add()([convm_1, convm])

convm_add_1_BA = BatchActivate(convm_add_1, True)
convm_2 = block(convm_add_1_BA, filters_m_expand, filters_m_project)
convm_add_2 = Add()([convm_2, convm_add_1])

convm_add_2 = BatchActivate(convm_add_2, False)
print("convm.shape = ", convm_add_2.shape)

# 8 -> 16
print("--------------------------------------------------------")
expand_filters_1 = 2 * 128
project_filters_1 = 128
to_add_1 = backbone.get_layer('block_6_expand_BN')
to_add_1 = to_add_1.output

uconv1_1 = transpose_block(convm_add_2, to_add_1, expand_filters_1, project_filters_1)

uconv1_1_BA = BatchActivate(uconv1_1, True)
uconv1_2 = block(uconv1_1_BA, expand_filters_1, project_filters_1)
uconv1_add_1 = Add()([uconv1_1, uconv1_2])

uconv1_add_1_BA = BatchActivate(uconv1_add_1, True)
uconv1_3 = block(uconv1_add_1_BA, expand_filters_1, project_filters_1)
uconv1_add_2 = Add()([uconv1_add_1, uconv1_3])

uconv1_add_2 = BatchActivate(uconv1_add_2, False)

print("uconv1_1.shape = ", uconv1_1.shape)
print("uconv1_2.shape = ", uconv1_2.shape)
print("uconv1_add_2.shape = ", uconv1_add_2.shape)

# 16 -> 32
print("--------------------------------------------------------")
expand_filters_2 = 2 * 64
project_filters_2 = 64
to_add_2 = backbone.get_layer('block_3_expand_BN')
to_add_2 = to_add_2.output

uconv2_1 = transpose_block(uconv1_add_2, to_add_2, expand_filters_2, project_filters_2)

uconv2_1_BA = BatchActivate(uconv2_1, True)
uconv2_2 = block(uconv2_1_BA, expand_filters_2, project_filters_2)
uconv2_add_1 = Add()([uconv2_1, uconv2_2])

uconv2_add_1_BA = BatchActivate(uconv2_add_1, True)
uconv2_3 = block(uconv2_add_1_BA, expand_filters_2, project_filters_2)
uconv2_add_2 = Add()([uconv2_add_1, uconv2_3])

uconv2_add_2 = BatchActivate(uconv2_add_2, False)

print("uconv2_1.shape = ", uconv2_1.shape)
print("uconv2_2.shape = ", uconv2_2.shape)
print("uconv2_add_2.shape = ", uconv2_add_2.shape)

# 32 -> 64
print("--------------------------------------------------------")
expand_filters_3 = 2 * 32
project_filters_3 = 32
to_add_3 = backbone.get_layer('expanded_conv_project_BN')
to_add_3 = to_add_3.output

uconv3_1 = transpose_block(uconv2_add_2, to_add_3, expand_filters_3, project_filters_3)

uconv3_1_BA = BatchActivate(uconv3_1, True)
uconv3_2 = block(uconv3_1_BA, expand_filters_3, project_filters_3)
uconv3_add_1 = Add()([uconv3_1, uconv3_2])

uconv3_add_1_BA = BatchActivate(uconv3_add_1, True)
uconv3_3 = block(uconv3_add_1_BA, expand_filters_3, project_filters_3)
uconv3_add_2 = Add()([uconv3_add_1, uconv3_3])

uconv3_add_2 = BatchActivate(uconv3_add_2, False)

print("uconv3_1.shape = ", uconv3_1.shape)
print("uconv3_2.shape = ", uconv3_2.shape)
print("uconv3_add_2.shape = ", uconv3_add_2.shape)

# 64 -> 128
print("--------------------------------------------------------")
filters_5 = 16
deconv0 = Conv2DTranspose(filters_5, (3, 3), strides=(2, 2), padding="same")(uconv3_add_2)
print("deconv0 = ", deconv0.shape)

uconv0 = Conv2D(filters_5, (3, 3), activation=None, padding="same")(deconv0)
uconv0_BA = BatchActivate(uconv0, True)
uconv0_1 = block(uconv0_BA, 16, 8)
uconv0_add_1 = Add()([uconv0, uconv0_1])

uconv0_add_1_BA = BatchActivate(uconv0_add_1, True)
uconv0_2 = block(uconv0_add_1_BA, 16, 8)
uconv0_add_2 = Add()([uconv0_2, uconv0_add_1])

uconv0_add_2 = BatchActivate(uconv0_add_2, False)

output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv0_add_2)
output_layer = Activation('sigmoid')(output_layer_noActi)
print("output_layer = ", output_layer.shape)

model = Model(inputs=[inputs], outputs=[output_layer])
print("BUILD MODEL")
adam_optimizer = optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=[my_iou_metric])
print("COMPILE")
model.summary()
# ------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------- #
#                                       Training                                  #
# ------------------------------------------------------------------------------- #
epochs = 40
batch_len = 4
history = model.fit(x_train, y_train, epochs=epochs, shuffle=True, batch_size=batch_len, callbacks=callbacks_list)

model.save('experimental_v3.1_small')
loss_history = np.array(history.history["loss"]).reshape((40, 1))
IoU_history = (np.array(history.history["my_iou_metric"])).reshape((40, 1))

numpy_history = np.array([(i, j) for (i, j) in zip(loss_history, IoU_history)], dtype=np.float32).reshape(40, 2)

print(numpy_history)
np.savetxt("loss_history_v3_large.txt", numpy_history, delimiter=",")


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
    gc.collect()
    del model


plot_history(hs={'U-Net': history}, epochs=epochs, metric='loss')
plot_history(hs={'U-Net': history}, epochs=epochs, metric='my_iou_metric')

clean_up(model)

K.clear_session()
gc.collect()
del backbone
gc.collect()

K.clear_session()
gc.collect()
del model
gc.collect()

print(model)

K.clear_session()
