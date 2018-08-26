"""Defines the model types.
"""

from model.debug_model import DebugModel
from model.conductor_net import ConductorNet
from model.fusion_net import FusionNet
from model.match_lstm import MatchLstm
from model.mnemonic_reader import MnemonicReader
from model.mnemonic_reader_scrl import MnemonicReaderScrl
from model.qa_model import QaModel
from model.rnet import Rnet

MODEL_TYPES = {
    "debug": DebugModel,
    "conductor_net": ConductorNet,
    "fusion_net": FusionNet,
    "match_lstm": MatchLstm,
    "mnemonic_reader": MnemonicReader,
    "mnemonic_reader_scrl": MnemonicReaderScrl,
    "qa_model": QaModel,
    "rnet": Rnet,
}
