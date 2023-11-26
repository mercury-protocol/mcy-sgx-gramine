from abc import ABC
# TODO: update readme about required data loader factory
# TODO: dataset factory creation could be a little bit simpler?


class AbstractFactory(ABC):
    def __init__(self, product_class, *args, **kwargs):
        self.product_class = product_class
        self.args = args
        self.kwargs = kwargs

    def create(self, *args):
        return self.product_class(*args, *self.args, **self.kwargs)


class DataSetFactory(AbstractFactory):
    def create(self, data_path):
        return super().create(data_path)


class DataLoaderFactory(AbstractFactory):
    def __init__(self, data_set_factory: DataSetFactory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_set_factory = data_set_factory

    def create(self, data_path):
        data_set = self.data_set_factory.create(data_path)
        return super().create(data_set)


class NetworkFactory(AbstractFactory):
    def create(self):
        return super().create()


class OptimizerFactory(AbstractFactory):
    def create(self, network_parameters):
        return super().create(network_parameters)
