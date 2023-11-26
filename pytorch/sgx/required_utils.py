class NetworkFactory:
    def __init__(self, network_class, *args, **kwargs):
        self.network_class = network_class
        self.args = args or list()
        self.kwargs = kwargs or dict()

    def create_network(self):
        return self.network_class(*self.args, **self.kwargs)


class OptimizerFactory:
    def __init__(self, optimizer_class, *args, **kwargs):
        self.optimizer_class = optimizer_class
        self.args = args or list()
        self.kwargs = kwargs or dict()

    def create_optimizer(self, network_parameters):
        return self.optimizer_class(network_parameters, *self.args, **self.kwargs)
