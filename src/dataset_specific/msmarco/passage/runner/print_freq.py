import sys
from omegaconf import OmegaConf

from cache import load_pickle_from

if __name__ == '__main__':
    dfs = load_pickle_from(sys.argv[1])
    
    for term, cnt in dfs.most_common(100000):
        print(term)
