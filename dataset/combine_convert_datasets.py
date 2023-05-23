import pickle
import torch
from torch_geometric.data import Data, Batch



def main(list_of_path):
    out = []

    for path in list_of_path:
        with open(path, "rb") as f:
            ret = pickle.load(f)
        if "question" in ret[0]:
            # {"graph": graph, "question": question, "answer": str(answer)}
            for rec in ret:
                g = rec["graph"]
                graph = Data(x=torch.asarray(g['node_feat']), edge_index=torch.asarray(g['edge_index']), edge_attr=torch.asarray(g['edge_feat']))
                rec["graph"] = graph
                out.append(rec)

        else:
            # {"graph": graph, "abstract": abstract}
            assert False, "should not go here"
            for rec in ret:
                rec["question"] = "Please describe the mechanism of this drug."
                ans = rec.pop("abstract")
                rec["answer"] = ans

    print("total data size:", len(out))
    with open("./dataset/chembl_pubchem_train.pkl", "wb") as f:
        pickle.dump(out, f)


if __name__ == '__main__':
    list_of_path = ["dataset/chembl_QA_train.pkl", "dataset/PubChem_QA_train.pkl"]
    main(list_of_path)