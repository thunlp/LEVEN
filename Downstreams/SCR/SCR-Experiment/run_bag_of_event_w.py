import json
import jsonlines
import torch
import os
from tqdm import tqdm
from utils.gen_metric import Metric


def get_event_vector(input_ids, q, mode='count'):
    vector = torch.zeros(109)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids[torch.nonzero(input_ids)].view(-1)
    for idx in input_ids:
        if mode != 'binary':
            vector[idx] += 1
        else:
            vector[idx] = 1

    # normalized TF
    # vector /= input_ids.shape[0]
    # IDF
    vector *= torch.log(100 / (candidate_idf_vectors[q]+1))   # comment this line to get vanilla bag-of-event

    return vector.view(1, -1)


def compute_cos_sim(v1, v2):
    return torch.cosine_similarity(v1.cuda(), v2.cuda())


if __name__ == "__main__":
    query_path = './input_data_unsupervised/query/query.json'
    candidate_path = './input_data_unsupervised/candidates'

    query = [{'idx': q['ridx'], 'event_ids': q['inputs']['event_ids']}
             for q in jsonlines.open(query_path)]
    candidates = {}

    for folder in os.listdir(candidate_path):
        candi_list = []
        for file in os.listdir(os.path.join(candidate_path, folder)):
            candi = json.load(open(os.path.join(candidate_path, folder, file), encoding='utf-8'))
            candi_list.append({'idx': int(file[:file.index('.')]),
                               'event_ids': candi['inputs']['event_ids']})
        candidates[folder] = candi_list

    candidate_idf_vectors = {}
    for key in candidates:
        idf_mat = torch.zeros(109)
        for candi in candidates[key]:
            event_ids = torch.tensor(candi['event_ids'])
            event_ids = event_ids[torch.nonzero(event_ids)].view(-1)

            idf_single = torch.zeros(109)
            for eid in event_ids:
                idf_single[eid] = 1

            idf_mat += idf_single

        candidate_idf_vectors[int(key)] = idf_mat

    result = {}
    for q in tqdm(query, desc='retrieving query'):
        q_idx = q['idx']
        q_vector = get_event_vector(q['event_ids'], q=q_idx)

        sim_results = []
        for candi in candidates[str(q_idx)]:
            c_idx = candi['idx']
            c_vector = get_event_vector(candi['event_ids'], q=q_idx)

            sim_results.append({'idx': c_idx, 'score': compute_cos_sim(q_vector, c_vector)})

        sim_results = sorted(sim_results, key=lambda x: x['score'], reverse=True)
        result[q_idx] = [q['idx'] for q in sim_results]

    os.makedirs('./result/event/', exist_ok=True)
    saved_file = './result/event/event_vector_result.json'
    with open(saved_file, 'w') as f:
        json.dump(result, f, ensure_ascii=False)

    met = Metric("./input_data")
    met.pred_single_path(saved_file)
