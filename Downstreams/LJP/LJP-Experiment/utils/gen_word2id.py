import json
import os

min_freq = 100

keep = set()

input_path = "/data/disk3/private/zhx/theme/data/ljp/frequency.txt"

if __name__ == "__main__":
    data = json.load(open(input_path, "r"))
    word_list = ["[UNK]", "[PAD]"]
    for word in data.keys():
        if data[word] > min_freq or word in keep:
            word_list.append(word.strip())
    for word in keep:
        word_list.append(word)
    word_list = list(set(word_list))

    word2id = {}
    for a in range(0, len(word_list)):
        word2id[word_list[a]] = a
    print(len(word2id))

    json.dump(word2id, open("/data/disk3/private/zhx/theme/data/ljp/word2id.txt", "w", encoding="utf8"),
              ensure_ascii=False,
              indent=2)
