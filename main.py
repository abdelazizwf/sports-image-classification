from pathlib import Path
from utils import load_images, write_predictions
from models import *


DATA_PATH = Path('Data')
TRAIN_DATA_PATH = DATA_PATH / 'Train'
TEST_DATA_PATH = DATA_PATH / 'Test'


if __name__ == '__main__':
    _, dataset = load_images(TRAIN_DATA_PATH)
    dataset = dataset.shuffle(buffer_size=1700)

    validation_dataset = dataset.take(500)
    train_dataset = dataset.skip(500)

    train_dataset = train_dataset.batch(64)
    validation_dataset = validation_dataset.batch(1)

    model = SEResNet()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=5
        ),
    ]

    model.fit(train_dataset, epochs=60, validation_data=validation_dataset, callbacks=callbacks)

    test_file_names, test_dataset = load_images(TEST_DATA_PATH, include_labels=False)
    predictions = model.predict(test_dataset.batch(1))

    write_predictions('/kaggle/working/predictions.csv', test_file_names, predictions)        
