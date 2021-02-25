#!/usr/bin/env python3

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Input, Sequential
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam


def create_model():
    # Create a sequential model to stack different layers. See:
    # https://keras.io/guides/sequential_model/
    model = Sequential()

    # model.add(InceptionV3(include_top=False, input_shape=(256, 256, 3)))

    # Add a Conv2D layer with 32 filters of size 5*5. See:
    # https://keras.io/api/layers/convolution_layers/convolution2d/
    model.add(
        Conv2D(32, kernel_size=(5, 5), activation="relu", input_shape=(256, 256, 3))
    )
    model.add(Conv2D(32, kernel_size=(5, 5), activation="relu"))

    # Add a MaxPooling2D layer. See:
    # https://keras.io/api/layers/pooling_layers/max_pooling2d/
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add a Dropout layer. See:
    # https://keras.io/api/layers/regularization_layers/dropout/
    model.add(Dropout(0.5))

    # Add a Flatten layer. See:
    # https://keras.io/api/layers/reshaping_layers/flatten/
    model.add(Flatten())

    # Add a Dense layer of size 256 w/ the rectified linear unit activation
    # function. See: https://keras.io/api/layers/core_layers/dense/, and:
    # https://keras.io/api/layers/activations/
    model.add(Dense(256, activation="relu"))

    # Add a Dropout layer with a rate of 0.5
    model.add(Dropout(0.5))

    # Add a Dense layer of size 256 w/ a pre-defined activation
    model.add(Dense(1, activation="sigmoid"))

    # Compile the model. See: https://keras.io/api/models/model_training_apis/
    model.compile(
        loss="binary_crossentropy",
        metrics=["accuracy"],
        optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
    )

    return model


def main():
    # We'll use an ImageDataGenerator object to augment the dataset.
    # ImageDataGenerator uses a Dynamic data augmentation method. See:
    # https://studymachinelearning.com/keras-imagedatagenerator/
    image_data_generator = ImageDataGenerator(rescale=1.0 / 255)

    # 1. Build the model from the training data

    # Apply the ImageDataGenerator to the train dataset. See:
    # https://keras.io/api/preprocessing/image/#flowfromdirectory-method
    train_data_generator = image_data_generator.flow_from_directory(
        "./data/TRAINDATA/train", class_mode="binary"
    )

    # Apply the ImageDataGenerator to the validation dataset
    validation_data_generator = image_data_generator.flow_from_directory(
        "./data/TRAINDATA/val", class_mode="binary"
    )

    # Create a hand-crafted model. See:
    model = create_model()

    # Fit the model
    # https://keras.io/api/models/model_training_apis/#fit-method
    history = model.fit(
        train_data_generator,
        epochs=80,
        steps_per_epoch=train_data_generator.n // train_data_generator.batch_size,
        validation_data=validation_data_generator,
        validation_steps=validation_data_generator.n
        // validation_data_generator.batch_size,
    )

    # Evaluate the model
    score = model.evaluate(validation_data_generator)
    print(f"Score - loss: {score[0]}, accuracy: {score[1]}")

    pyplot.figure()
    pyplot.plot(history.history["accuracy"], "orange", label="Training accuracy")
    pyplot.plot(history.history["val_accuracy"], "blue", label="Validation accuracy")
    pyplot.plot(history.history["loss"], "red", label="Training loss")
    pyplot.legend()

    pyplot.savefig("history.png")


if __name__ == "__main__":
    main()
