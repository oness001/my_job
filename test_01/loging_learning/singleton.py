class Singleton:
    """
    单例装饰器。
    """
    __cls = dict()

    def __init__(self, cls):
        self.__key = cls

    def __call__(self, *args, **kwargs):
        if self.__key not in self.cls:
            self[self.__key] = self.__key(*args, **kwargs)
        return self[self.__key]

    def __setitem__(self, key, value):
        self.cls[key] = value

    def __getitem__(self, item):
        return self.cls[item]

    @property
    def cls(self):
        return self.__cls

    @cls.setter
    def cls(self, cls):
        self.__cls = cls