from keras.preprocessing.image import ImageDataGenerator

TRAIN_SOURCE = 'Images/training'  # Source for training images
VALID_SOURCE = 'Images/validation'  # source for validation images


# Create ImageDataGenerator for training and validation
def train_val_gen(TRAIN_SOURCE, VALID_SOURCE):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=20,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(TRAIN_SOURCE,
                                                        target_size=(300, 300),
                                                        batch_size=32,
                                                        class_mode='binary')
    valid_generator = validation_datagen.flow_from_directory(VALID_SOURCE,
                                                             target_size=(300, 300),
                                                             batch_size=32,
                                                             class_mode='binary')

    return train_generator, valid_generator


# Outer function
def get_generators():
    return train_val_gen(TRAIN_SOURCE, VALID_SOURCE)
