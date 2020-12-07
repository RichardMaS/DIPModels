import os

from keras import applications
from keras.preprocessing import image

## A standard data_generator with flow_from_directoy for keras.
## Define own data_generator for complicated preprocessing and structure.
def data_generator(train_processor, valid_processor, train_dir, valid_dir,
                   batch_size, class_mode,
                   target_size=None, save_to_dir=None, **kwargs):
    # Training generator with augmentation
    train_datagen = image.ImageDataGenerator(
        preprocessing_function=lambda x: train_processor(x, kwargs))
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=target_size,
        save_to_dir=save_to_dir,
        batch_size=batch_size,
        class_mode=class_mode
    )
    
    # Validation generator without augmentation
    valid_datagen = image.ImageDataGenerator(
        preprocessing_function=lambda x: valid_processor(x, kwargs))
    valid_generator = valid_datagen.flow_from_directory(
        directory=valid_dir,
        target_size=target_size,
        save_to_dir=save_to_dir,
        batch_size=batch_size,
        class_mode=class_mode
    )
    
    return train_generator, valid_generator

