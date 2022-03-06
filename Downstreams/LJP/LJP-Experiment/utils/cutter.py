import os
import json
import thulac

cutter = thulac.thulac(seg_only=True)
frequency = {}

path_list = [
    ["/data/disk3/private/zhx/theme/data/ljp/final_all_data/exercise_contest",
     "/data/disk3/private/zhx/theme/data/ljp/final_all_data/exercise_contest_cutted"],
    ["/data/disk3/private/zhx/theme/data/ljp/final_all_data/first_stage",
     "/data/disk3/private/zhx/theme/data/ljp/final_all_data/first_stage_cutted"],
    ["/data/disk3/private/zhx/theme/data/ljp/final_all_data/restData",
     "/data/disk3/private/zhx/theme/data/ljp/final_all_data/restData_cutted"],
]


def cut(s):
    arr = list(cutter.fast_cut(s))
    for a in range(0, len(arr)):
        arr[a] = arr[a][0]
    for word in arr:
        if not (word in frequency):
            frequency[word] = 0
        frequency[word] += 1
    return arr


if __name__ == "__main__":
    for input_path, output_path in path_list:
        os.makedirs(output_path, exist_ok=True)
        for filename in os.listdir(input_path):
            print(os.path.join(input_path, filename))
            data = []

            f = open(os.path.join(input_path, filename), "r", encoding="utf8")

            for line in f:
                x = json.loads(line)
                x["fact"] = cut(x["fact"])

                data.append(x)

            f = open(os.path.join(output_path, filename), "w", encoding="utf8")
            for x in data:
                print(json.dumps(x, ensure_ascii=False, sort_keys=True), file=f)
            f.close()

    json.dump(frequency, open("/data/disk3/private/zhx/theme/data/ljp/frequency.txt", "w", encoding="utf8"),
              indent=2,
              ensure_ascii=False)
