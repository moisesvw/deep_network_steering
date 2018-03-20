from  train import Train
class TestNetwork(object):

    def test_train(self):
        train = Train('./data/driving_log.csv', './data')
        train.perform()
        assert (1 == 1)

    def test_two(self):

        assert (1 == 1)