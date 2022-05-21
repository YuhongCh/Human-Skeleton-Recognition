import os
import pandas as pd

"""
This class load the data from the dataset file and split them into training and validation set
"""
class Dataloader:
    def __init__(self, image_path: str, annotation_path: str, batch_size: int = 128):
        # check for parameter errors
        if not os.path.isdir(image_path):
            raise ValueError("Error: image path either does not exists or it is not a folder")
        if not os.path.isfile(annotation_path):
            raise ValueError("Error: annotation path either does not exists or it is not a file")

        # get data
        self.annotation = pd.read_json(annotation_path)
        self.images = os.listdir(image_path)
        if len(self.annotation) == len(self.images):
            raise ValueError("Error: the length of the image and annotation is different, please check your input")

        # initialize other parameters
        self.batch_size = batch_size
        self.data_size = len(self.images)
        

    def split_data(self, ratio: float):
        """Set up the data into training (ratio) and validation (1 - ratio)"""
        if len(ratio) != 2 or sum(ratio) != 1:
            raise ValueError("Error: ratio should only have 2 float sum up to 1")

        self.training_data = self.annotation.sample(ratio)
        self.validation_data = self.annotation.drop(self.training_data.index)

    
    def __len__(self) -> int:
        """Return the number of batches for training"""
        return self.data_size // self.batch_size


"""
This class provides functionalities for a dataset
"""
class Dataset:
    def __init__(self, df: pd.DataFrame):
        pass




