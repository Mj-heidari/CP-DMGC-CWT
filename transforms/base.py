class BaseTransform:
    def __call__(self, eeg, **kwargs):
        raise NotImplementedError

    @property
    def repr_body(self):
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}({self.repr_body})"
    
if __name__ == "__main__":
    from base import BaseTransform
    import numpy as np
    
    class Dummy(BaseTransform):
        def __call__(self, eeg, **kwargs):
            return eeg

    eeg = np.random.randn(2, 100)
    print("BaseTransform subclass works:", Dummy()(eeg).shape)