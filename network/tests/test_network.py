from  train import Train
import os

import os.path
class TestNetwork(object):
    def test_train(self):
        model_path = './tests/test.h5' 
        assert (os.path.isfile(model_path) == False)
        train = Train('./tests/data/driving_log.csv', './tests/data/', model_path,
                        bath_size=10, epochs=1)
        train.perform_training()
        assert (train.model_evaluation > 0.001)
        assert (os.path.isfile(model_path) )
        os.remove(model_path)

