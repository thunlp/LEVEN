import os
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class FilenameOnlyDataset(Dataset):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = config.get("data", "%s_data_path" % mode)

        filename_list = config.get("data", "%s_file_list" % mode).replace(" ", "").split(",")
        recursive = config.getboolean("data", "recursive")

        for name in filename_list:
            self.file_list = self.file_list + dfs_search(os.path.join(self.data_path, name), recursive)
        self.file_list.sort()

    def __getitem__(self, item):
        return self.file_list[item]

    def __len__(self):
        return len(self.file_list)
