import json
import os
from torch.utils.data import Dataset
from tools.dataset_tool import dfs_search


class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.query_path = config.get("data", "query_path")
        self.cand_path = config.get("data", "cand_path")
        self.labels = json.load(open(config.get("data", "label_path"), 'r'))
        self.data = []

        test_file = config.get("data", "test_file")
        querys = []
        for i in range(5):
            if mode == 'train':
                if test_file == str(i):
                    continue
                else:
                    querys += json.load(open(os.path.join(self.query_path, 'query_%d.json' % i), 'r'))
            else:
                if test_file == str(i):
                    querys = json.load(open(os.path.join(self.query_path, 'query_%d.json' % i), 'r'))
        for query in querys:
            que = query["q"]
            path = os.path.join(self.cand_path, str(query["ridx"]))
            for fn in os.listdir(path):
                cand = json.load(open(os.path.join(path, fn), "r"))
                self.data.append({
                    "query": que,
                    "cand": cand["ajjbqk"],
                    "label": int(fn.split('.')[0]) in self.labels[str(query["ridx"])],
                    "index": (query["ridx"], fn.split('.')[0])
                })

    def __getitem__(self, item):
        return self.data[item % len(self.data)]

    def __len__(self):
        if self.mode == "train":
            return len(self.data)
        else:
            return len(self.data)
