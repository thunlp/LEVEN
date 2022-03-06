import json
import os
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class JsonFromFilesDataset(Dataset):
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

        self.load_mem = config.getboolean("data", "load_into_mem")
        self.json_format = config.get("data", "json_format")

        if self.load_mem:
            self.data = []
            for filename in self.file_list:
                if self.json_format == "single":
                    self.data = self.data + json.load(open(filename, "r", encoding=encoding))
                else:
                    f = open(filename, "r", encoding=encoding)
                    for line in f:
                        self.data.append(json.loads(line))

        else:
            self.total = 0
            self.prefix_file_cnt = []

            if self.json_format == "single":
                self.temp_data = {
                    "data": json.load(open(self.file_list[0], "r", encoding=encoding)),
                    "file_id": 0
                }
            else:
                self.temp_file_list = []

            for filename in self.file_list:
                if self.json_format == "single":
                    data = json.load(open(filename, "r", encoding=encoding))
                    self.prefix_file_cnt.append(len(data))
                else:
                    f = open(filename, "r", encoding=encoding)
                    cnt = 0
                    for line in f:
                        cnt += 1
                    f.close()
                    self.temp_file_list.append({
                        "file": open(filename, "r", encoding=encoding),
                        "cnt": 0
                    })
                    self.prefix_file_cnt.append(cnt)

            for a in range(1, len(self.prefix_file_cnt)):
                self.prefix_file_cnt[a] += self.prefix_file_cnt[a - 1]
            self.total = self.prefix_file_cnt[-1]

    def get_file_id(self, item):
        l = 0
        r = len(self.prefix_file_cnt)
        while l + 1 != r:
            m = (l + r) // 2
            if self.prefix_file_cnt[m-1] <= item:
                l = m
            else:
                r = m

        return l

    def __getitem__(self, item):
        if self.load_mem:
            return self.data[item]
        else:
            which = self.get_file_id(item)
            if which == 0:
                idx = item
            else:
                idx = item - self.prefix_file_cnt[which - 1]

            if self.json_format == "single":
                if self.temp_data["file_id"] != which:
                    self.temp_data = {
                        "data": json.load(open(self.file_list[which], "r", encoding=self.encoding)),
                        "file_id": 0
                    }

                return self.temp_data["data"][idx]

            else:
                if self.temp_file_list[which]["cnt"] > idx:
                    self.temp_file_list[which] = {
                        "file": open(self.file_list[which], "r", encoding=self.encoding),
                        "cnt": 0
                    }

                delta = idx - self.temp_file_list[which]["cnt"]
                self.temp_file_list[which]["file"].readlines(delta)

                data = json.loads(self.temp_file_list[which]["file"].readline())
                self.temp_file_list[which]["cnt"] = idx + 1

                return data

    def __len__(self):
        if self.load_mem:
            return len(self.data)
        else:
            return self.total
