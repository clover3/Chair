from tlm.data_gen.base import MaskedPairGen
from tlm.data_gen.lm_worker import LMJobRunner

if __name__ == "__main__":
    runner = LMJobRunner(1000, "masked_pair", MaskedPairGen)
    runner.start()

