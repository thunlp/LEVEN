import os
import json
import math


def ndcg(ranks, K):
    dcg_value = 0.
    idcg_value = 0.
    log_ki = []

    sranks = sorted(ranks, reverse=True)

    for i in range(0, K):
        logi = math.log(i+2, 2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi

    return dcg_value/idcg_value


class Metric:
    def __init__(self, data_path):
        self.avglist = json.load(open(os.path.join(data_path, "label", "label_top30_dict.json"), "r"))
        self.combdic = json.load(open(os.path.join(data_path, "prediction", "combined_top100.json"), "r"))
    
    def NDCG(self, pred, K):
        sndcg = 0.0
        for key in pred.keys():
            rawranks = [self.avglist[key][str(i)] for i in pred[key] if i in list(self.combdic[key][:30])]
            ranks = rawranks + [0]*(30-len(rawranks))
            if sum(ranks) != 0:
                sndcg += ndcg(ranks, K)
        return round(sndcg/len(pred), 4)
    
    def P(self, pred, K):
        sp = 0.0
        for key in pred.keys():
            ranks = [i for i in pred[key] if i in list(self.combdic[key][:30])]
            sp += float(len([j for j in ranks[:K] if self.avglist[key][str(j)] == 3])/K)
        return round(sp/len(pred), 4)
    
    def MAP(self, pred):
        smap = 0.0
        for key in pred.keys():
            ranks = [i for i in pred[key] if i in list(self.combdic[key][:30])]
            rels = [ranks.index(i) for i in ranks if self.avglist[key][str(i)] == 3]
            tem_map = 0.0
            for rel_rank in rels:
                tem_map += float(len([j for j in ranks[:rel_rank+1] if self.avglist[key][str(j)] == 3])/(rel_rank+1))
            if len(rels) > 0:
                smap += tem_map / len(rels)
        return round(smap/len(pred), 4)

    def pred_path(self, path):
        fnames = os.listdir(path)
        res = {}
        for fn in fnames:
            fsp = fn.split("-")
            epoch = int(fsp[-1][0])
            tfile = int(fsp[-2][0])
            metric = {}
            pred = json.load(open(os.path.join(path, fn)))
            for K in [5, 10, 20, 30]:
                metric["NDCG@%d" % K] = self.NDCG(pred, K)
                metric["P%d" % K] = self.P(pred, K)
            metric["MAP"] = self.MAP(pred)
            modelname = fsp[0]
            if modelname not in res:
                res[modelname] = {}
            if tfile not in res[modelname]:
                res[modelname][tfile] = {"MAP": -1, "best": -1}
            if metric["MAP"] > res[modelname][tfile]["MAP"]:
                res[modelname][tfile] = metric
                res[modelname][tfile]["best"] = epoch
        for model in res:
            overall = {}
            for tf in res[model]:
                for key in res[model][tf]:
                    if key not in overall:
                        overall[key] = 0
                    overall[key] += res[model][tf][key]
            for key in overall:
                overall[key] /= len(res[model])
            print("==" * 20)
            print(json.dumps(overall, ensure_ascii=False, sort_keys=True))

    def pred_single_path(self, path):
        pred = json.load(open(path))

        metric = {}
        for K in [5, 10, 20, 30]:
            metric["NDCG@%d" % K] = self.NDCG(pred, K)
            metric["P%d" % K] = self.P(pred, K)
        metric["MAP"] = self.MAP(pred)
        print(json.dumps(metric, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    met = Metric("./input_data")

    # vanilla BERT
    print('vanilla BERT')
    met.pred_path('./result/BERT/test0')   # path to the predicted files

    # BERT with event
    print('BERT with event')
    met.pred_path('./result/EDBERT/test0')