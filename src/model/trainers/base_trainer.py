from abc import abstractmethod, ABC


class Base_Trainer(ABC):
    @abstractmethod
    def train(self):
        pass

    def lr_test(self):
        pass
