# Here we are prepearing our future dataset
import os
import zipfile
import shutil

FILENAME = 'archive.zip'  # downloaded from kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog
BASE_DIR = 'Images'  # Base dir for ImageDataset
TARGET_DIR = 'tmp/train'  # Base dir for copying
DESTINATION_DIR = BASE_DIR  # Where to copy images
SPLIT = 0.8  # Splitting data for training and validation


# Extracting files from zipfile
def extract_files():
    os.makedirs('tmp')
    print('Created temporary folder for zip extraction')

    with zipfile.ZipFile(FILENAME, 'r') as zpfile:
        print('Extracting files from zip')
        zpfile.extractall('tmp')
        print('Extraction finished')
        print('-' * 30)


# Creating folders for dataset
def create_folders():
    dirs = ('training',
            'validation',
            'training/hotdog',
            'training/nothotdog',
            'validation/hotdog',
            'validation/nothotdog')

    os.makedirs(BASE_DIR)
    for dir1 in dirs:
        path = os.path.join(BASE_DIR, dir1)
        os.makedirs(path)

    print('Created folders for dataset')


# Moving files to new created dataset folders
def move_files():
    print('starting copying files to dataset')

    dirs = ('hot_dog', 'not_hot_dog')
    for dir1 in dirs:
        target = os.path.join(TARGET_DIR, dir1)
        split = int(len(os.listdir(target)) * SPLIT)
        count = 0
        for filename in os.listdir(target):
            dest = dir1.replace('_', '')
            trainvalid = 'training' if count < split else 'validation'
            dest = os.path.join(DESTINATION_DIR, trainvalid, dest)
            shutil.copyfile(os.path.join(target, filename), os.path.join(dest, filename))
            count += 1
    print('all files copied')


# Deleting temporary files from zip extraction, we don`t need them now.
def clear_dirs():
    print('clearing from uneccesary files')

    try:
        shutil.rmtree('temp')
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    extract_files()
    create_folders()
    move_files()
    clear_dirs()
