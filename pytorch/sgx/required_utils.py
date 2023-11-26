from abc import ABC


class AbstractFactory(ABC):
    def __init__(self, product_class, *args, **kwargs):
        self.product_class = product_class
        self.args = args
        self.kwargs = kwargs

    def create(self, *args):
        return self.product_class(*args, *self.args, **self.kwargs)


class DataLoaderFactory(AbstractFactory):
    def create(self, data_path):
        return super().create(data_path)


class NetworkFactory(AbstractFactory):
    def create(self):
        return super().create()


class OptimizerFactory(AbstractFactory):
    def create(self, network_parameters):
        return super().create(network_parameters)
