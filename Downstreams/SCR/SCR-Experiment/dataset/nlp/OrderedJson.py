import json
import os
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class OrderedJsonDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding

        filename_list = config.get("data", "%s_file_list" % mode).replace(" ", "").split(",")
        recursive = config.getboolean("data", "recursive")

        for name in filename_list:
            self.file_list = self.file_list + dfs_search(os.path.join(self.data_path, name), recursive)
        self.file_list.sort()

        self.json_format = config.get("data", "json_format")

        self.total = 0

        for filename in self.file_list:
            if self.json_format == "single":
                data = json.load(open(filename, "r", encoding=encoding))
                for a in range(0, len(data)):
                    if self.check(data[a]):
                        self.total += 1
            else:
                f = open(filename, "r", encoding=encoding)
                for line in f:
                    data = json.loads(line)
                    if self.check(data):
                        self.total += 1
                f.close()

        self.init_zero()

    def init_zero(self):
        self.item_id = 0
        self.file_id = 0
        if self.mode == "single":
            self.temp_data = json.load(open(self.file_list[self.file_id], "r", encoding=self.encoding))
            self.cur_id = 0
        else:
            self.temp_file = open(self.file_list[self.file_id], "r", encoding=self.encoding)

    def check(self, data):
        return True

    def __getitem__(self, item):
        self.item_id += 1

    def __len__(self):
        return self.total
