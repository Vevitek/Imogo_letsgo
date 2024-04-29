import cv2
import os
import numpy as np
from natsort import natsorted
from skimage.io import imshow
import imageio.v2 as imageio
import tensorflow as tf
import random
import matplotlib as plt

seed = 42
np.random.seed = seed
def path_creation(output_folder,Folders_path,crop_width = 2000,crop_height = 2000):
    # If there is no output folder, it creates it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read source folders before conversion
    new_subfolders = []
    for folder_path in Folders_path:
        # Create subfolder path for B&W and good dimensions
        output_subfolder = os.path.join(output_folder, os.path.basename(folder_path))
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        new_subfolders.append(output_subfolder)

        # Browse files in source folders
        for filename in os.listdir(folder_path):
            # Build path
            input_path = os.path.join(folder_path, filename)

            # Load iamge
            image = cv2.imread(input_path)

            # Crop images based on their actual dimensions
            h, w, _ = image.shape
            x = (w - crop_width) // 2
            y = (h - crop_height) // 2
            cropped_image = image[y:y + crop_height, x:x + crop_width]

            # Save cropped image in the corresponding folder
            output_path = os.path.join(output_subfolder, filename)
            cv2.imwrite(output_path, cropped_image)

    ##-----------------tainvaltest---------------##
    train_X, train_Y = [],[]
    val_X,val_Y = [],[]
    test_X,test_Y = [],[]

    # trainvaltest_list = [train_X,train_Y,val_X,val_Y,test_X,test_Y]

    for i, path in enumerate(new_subfolders):
        im_list = natsorted(os.listdir(path))
        imgs = [imageio.imread(os.path.join(path, f)) for f in im_list]
        # Convertir les images en niveaux de gris
        imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs]
        imgs = [cv2.resize(img, (crop_width, crop_height), interpolation=cv2.INTER_AREA) for img in imgs]
        imgs = np.array(imgs)[..., np.newaxis]

        if i == 0:
            train_X = imgs
        elif i == 1:
            train_Y = imgs
        elif i == 2:
            val_X = imgs
        elif i == 3:
            val_Y = imgs
        elif i == 4:
            test_X = imgs
        else:
            test_Y = imgs

    trainvaltest_list = train_X,train_Y,val_X,val_Y,test_X,test_Y

    print(train_X.shape[0], "images for training,", val_X.shape[0], "images for validation, and",
          test_X.shape[0], "images for testing")

    inputs = tf.keras.layers.Input((crop_width, crop_height, 1))
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    tf.print('c1 =', c1.shape)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    tf.print('p1 =', p1.shape)

    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    tf.print('c2 =', c2.shape)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    tf.print('p2 =', p2.shape)

    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    tf.print('c3 =', c3.shape)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    tf.print('p3 =', p3.shape)

    c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    tf.print('c4 =', c4.shape)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    tf.print('p4 =', p4.shape)

    c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    tf.print('c5 =', c5.shape)

    # Partie d'expansion
    u6 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Model checkpoint

    checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_imo.h5', verbose=1, save_best_only=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

    results = model.fit(train_X, train_Y, validation_split=0.1, batch_size=1, epochs=3, callbacks=callbacks)

    ####################################

    idx = random.randint(0, len(train_X))

    preds_train = model.predict(train_X[:int(train_X.shape[0] * 0.9)], verbose=1)
    preds_val = model.predict(train_X[int(train_X.shape[0] * 0.9):], verbose=1)
    preds_test = model.predict(test_X, verbose=1)

    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    # Perform a sanity check on some random training samples
    ix = random.randint(0, len(preds_train_t))
    imshow(train_X[ix])
    plt.show()
    imshow(np.squeeze(train_Y[ix]))
    plt.show()
    imshow(np.squeeze(preds_train_t[ix]))
    plt.show()

    # Perform a sanity check on some random validation samples
    ix = random.randint(0, len(preds_val_t))
    imshow(train_X[int(train_X.shape[0] * 0.9):][ix])
    plt.show()
    imshow(np.squeeze(train_Y[int(train_Y.shape[0] * 0.9):][ix]))
    plt.show()
    imshow(np.squeeze(preds_val_t[ix]))
    plt.show()
