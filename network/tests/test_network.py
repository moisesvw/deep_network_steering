from  train import Train
import os

import os.path
class TestNetwork(object):
    def test_train(self):
        assert (os.path.isfile('./tests/test.h5') == False)

        opts = {
            'train_logs': './tests/data/driving_log.csv',
            'train_data': './tests/data/',
            'test_logs': './tests/data/driving_log.csv',
            'test_data': './tests/data/',
            'model_path': './tests/test.h5'
        }
        train = Train(opts, bath_size=10, epochs=1)
        train.perform_training()
        assert (train.model_evaluation > 0.001)
        assert (os.path.isfile('./tests/test.h5') )
        os.remove('./tests/test.h5')

