from pathlib import Path
from utils import load_images, write_predictions
from models import get_alexnet


DATA_PATH = Path('Data')
TRAIN_DATA_PATH = DATA_PATH / 'Train'
TEST_DATA_PATH = DATA_PATH / 'Test'


if __name__ == '__main__':
    _, train_dataset = load_images(TRAIN_DATA_PATH)
    train_dataset = train_dataset.shuffle(buffer_size=840).batch(64)

    model = get_alexnet()

    model.fit(train_dataset, epochs=1)

    test_file_names, test_dataset = load_images(TEST_DATA_PATH, include_labels=False)
    predictions = model.predict(test_dataset.batch(1))

    write_predictions('predictionss.csv', test_file_names, predictions)
        
