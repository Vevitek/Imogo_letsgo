from Folders_treatment import *

# img_width = 3838
# img_height = 3710
#
# img_width_c = int(3712/2)
# img_height_c = int(3840/2)
#
# inputs = tf.keras.layers.Input((img_width,img_height,1))
# inputs = tf.keras.layers.Resizing(img_height_c,img_width_c)(inputs)
#
# s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

TRAIN_I_PATH = r'C:/Users/za274317/Documents/Imogo/TF_trainimages1BW'
TRAIN_M_PATH = r'C:/Users/za274317/Documents/Imogo/TF_trainmasks1BW'

VALID_I_PATH = r"C:\Users\za274317\Documents\Imogo\TF_validationimages1BW" #'C:/Users/za274317/Documents/Imogo/TF_validationimages1BW'
VALID_M_PATH = r"C:\Users\za274317\Documents\Imogo\TF_validationmasks1BW" #'C:/Users/za274317/Documents/Imogo/TF_validationmasks1BW'

TEST_I_PATH = r"C:\Users\za274317\Documents\Imogo\TF_testimages1" #r"C:\Users\za274317\Documents\Imogo\TF_testimages1BW"
TEST_M_PATH = r"C:\Users\za274317\Documents\Imogo\TF_testmasks1BW" #r'C:/Users/za274317/Documents/Imogo/TF_testmasks1BW'

Folders_path = [TRAIN_I_PATH,TRAIN_M_PATH,VALID_I_PATH,VALID_M_PATH,TEST_I_PATH,TEST_M_PATH]

output_folder = 'C:/Users/za274317/Documents/Imogo/Cropped_images'

# Dimensions de recadrage
crop_width = 1500
crop_height = 1500

trainvaltest = path_creation(output_folder,Folders_path,crop_width,crop_height)

