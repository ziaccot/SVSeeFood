from datagen_prep import get_generators
from dataset_prep import create_dataset
from model_prep import create_model
from matplotlib import pyplot as plt

EPOCHS = 20 # num of epochs

# Training model and saving it to local drive
def train_model():
    create_dataset()
    training_generator, validation_generator = get_generators()
    model = create_model()

    history = model.fit(training_generator,
                        epochs=EPOCHS,
                        validation_data=validation_generator)

    # Plotting the training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Saving the model
    model.save('hotdog_not_hotdog_model.h5')

if __name__ == '__main__':
    train_model()