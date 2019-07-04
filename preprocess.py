import pandas as pd
import numpy as np

path = 'data/fer2013.csv'

def preprocess(path, set_type):
    """
	Args:
		path: string, optional - Path to dataset
		set_type: string, optional - Training / PublicTest / PrivateTest
	Returns data from set_type as a dataframe
    """
    data = pd.read_csv(path)
    data = data[data.Usage == set_type]
    data['pixels'] = data['pixels'].apply(lambda x: np.fromstring(x, dtype=np.int, sep=' '))
    return data

if __name__=='__main__':
    preprocess(path)
