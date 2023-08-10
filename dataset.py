import numpy as np
import cv2


class Dataset:
    '''
    Dataset representation.
    '''

    def __init__(self, files: list[str]):
        '''
        :param files: a list of file paths
        '''

        self.files = files

    def __len__(self) -> int:
        '''
        :return: number of files
        '''
        return len(self.files)

    def __getitem__(self, idx) -> (np.array, int):
        '''

        :param idx: index
        :return: tuple (image at index idx, class label)
        '''

        path = self.files[idx]

        # read image and resize it
        img = cv2.imread(path, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))

        # using backslashes in paths
        label = int(path.split('\\')[-2])

        # one hot encoding
        one_hot = np.zeros((1, 10))
        one_hot[0][label] = 1.0

        return img, one_hot
