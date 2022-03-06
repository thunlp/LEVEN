import json

from .accuracy_tool import gen_micro_macro_result


def null_output_function(data, config, *args, **params):
    return ""


def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)


def ljp_output_function(data, config, *args, **params):
    temp = {}
    temp["zm"] = gen_micro_macro_result(data["zm"])
    temp["ft"] = gen_micro_macro_result(data["ft"])
    temp["xq"] = data["xq"]
    result = {}
    for name in ["zm", "ft"]:
        result[name] = []
        for name_ in ["mip", "mir", "mif", "map", "mar", "maf"]:
            result[name].append(temp[name][name_])

    result["xq"] = data["xq"][1] / data["xq"][0]

    return json.dumps(result, sort_keys=True), result
