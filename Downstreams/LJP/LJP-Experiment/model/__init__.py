from model.ljp.Bert import LJPBert
from model.ljp.TopJudge import TopJudge
from model.ljp.CNN import LJPCNN
from model.ljp.Gating import Gating
from model.ljp.LSTM import LSTM
from model.ljp.DPCNN import DPCNN

model_list = {
    "LJPBert": LJPBert,
    "LJPTopjudge": TopJudge,
    "LJPCNN": LJPCNN,
    "LJPGating": Gating,
    "LJPLSTM": LSTM,
    "LJPDPCNN": DPCNN
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
