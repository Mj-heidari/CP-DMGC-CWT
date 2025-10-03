from models.EEGNet import EEGNet
from models.CE_stSENet.CE_stSENet import CE_stSENet
from models.cspnet import CSPNet
from models.stnet import STNet
from models.TSception import TSception
from models.FBMSNet import FBMSNet

def model_builder(model_class, **kwargs):
    """
    Returns a function that builds a fresh model instance.
    This avoids weight leakage across folds.

    Args:
        model_class: torch.nn.Module class (e.g., EEGNet)
        kwargs: parameters to initialize the model

    Returns:
        A callable that builds a new model each time it's called
    """

    def build():
        return model_class(**kwargs)

    return build


def get_builder(model: str = "CE-stSENet"):
    match model:
        case "EEGNET":
            builder = model_builder(
                EEGNet,
                chunk_size=640,
                num_electrodes=18,
                dropout=0.5,
                kernel_1=64,
                kernel_2=16,
                F1=8,
                F2=16,
                D=2,
                num_classes=2,
            )
            return builder
        case "CE-stSENet":
            builder = model_builder(
                CE_stSENet,
                inc=18,
                class_num=2,
                si=128,
            )
            return builder
        case "cspnet":
            builder = model_builder(
                CSPNet,
                chunk_size=128*5,
                num_electrodes=18,
                num_classes=2,
                dropout=0.5,
                num_filters_t=20,
                filter_size_t=25,
                num_filters_s=2,
                filter_size_s=-1,
                pool_size_1=100,
                pool_stride_1=25,
            )
            return builder
        case "stnet":
            builder = model_builder(
                STNet,
                chunk_size=128*5,
                grid_size=(9, 9),
                num_classes=2,
                dropout=0.2
            )
            return builder
        case "TSception":
            builder = model_builder(
                TSception,
                num_classes = 2,
                input_size = (18, 640),
                sampling_rate = 256,
                num_T = 9,
                num_S = 6,
                hidden = 128,
                dropout_rate = 0.2
            )
            return builder        
        case "FBMSNet":
            builder = model_builder(
                FBMSNet,
                nChan = 18,
                nTime = 640,
                nClass = 2
            )
            return builder       

        case _:
            raise NotImplementedError
