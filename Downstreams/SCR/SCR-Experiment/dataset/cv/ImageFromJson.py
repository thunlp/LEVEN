import json
from torch.utils.data import Dataset
import cv2
import os


class ImageFromJsonDataset(Dataset):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.prefix = config.get("data", "%s_data_prefix" % mode)

        self.data_list = json.load(open(self.data_path, "r"))
        self.load_mem = config.getboolean("data", "load_into_mem")

        if self.load_mem:
            for a in range(0, len(self.data_list)):
                self.data_list[a]["data"] = cv2.imread(os.path.join(self.prefix, self.data_list[a]["path"]))

    def __getitem__(self, item):
        if self.load_mem:
            return {
                "data": self.data_list[item]["data"],
                "label": self.data_list[item]["label"]
            }
        else:
            return {
                "data": cv2.imread(os.path.join(self.prefix, self.data_list[item]["path"])),
                "label": self.data_list[item]["label"]
            }

    def __len__(self):
        return len(self.data_list)
