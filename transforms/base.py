class BaseTransform:
    def __call__(self, eeg, **kwargs):
        raise NotImplementedError

    @property
    def repr_body(self):
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}({self.repr_body})"