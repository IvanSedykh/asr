import json
import numpy as np

def main():
    # index_fpath = 'data/datasets/librispeech/test-other_index.json'
    # index_fpath = 'data/datasets/librispeech/test-clean_index.json'
    # index_fpath = 'data/datasets/librispeech/train-clean-100_index.json'
    # index_fpath = 'data/datasets/librispeech/train-clean-360_index.json'
    index_fpath = 'data/datasets/librispeech/train-other-500_index.json'
    print(f"{index_fpath=}")
    with open(index_fpath) as f:
        index = json.load(f)

    MAX_DUR = 20
    lens = []
    for item in index:
        text = item['text']
        lens.append(len(text))

    print(f"mean len : {np.mean(lens)}")
    q = [50, 75, 90, 95, 99]
    print(f"q={q}")
    print(f"{np.percentile(lens, q)}")

if __name__ == '__main__':
    main()