from collections import OrderedDict
import csv
import os
import json
from pprint import pprint
import yaml
import shutil
import argparse
import numpy as np

def read_file(path):
    data, data_map = [], {}
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
            key = (row['job_male'], row['job_female'])
            data_map[key] = row
    return data, data_map

def load_json(file_path):
    with open(file_path, 'rb') as f:
        data = json.load(f)
    return data


class EvalAna():
    def __init__(self, result_file, test_file, cf_result_file, cf_test_file, save_dir) -> None:
        self.result_data = json.load(open(result_file, 'rb'))
        self.test_data = json.load(open(test_file, 'rb'))
        self.cf_result_data = json.load(open(cf_result_file, 'rb'))
        self.cf_test_data = json.load(open(cf_test_file, 'rb'))


        assert len(self.result_data) == len(self.test_data), "result and test data should have the same length"
        assert len(self.cf_result_data) == len(self.cf_test_data), "cf result and test data should have the same length"
        assert len(self.result_data) == len(self.cf_result_data), "result and cf result should have the same length"

        similar_occ_path = "../construction/vq_generation/similarity/occ_merge_filter_sim.csv"
        self.similar_occ_data, self.similar_occ_data_map = read_file(similar_occ_path)


        # merge_data: {f"{occ}+{occ_sim}+{gender}": [{"id":1, "metric_result": True}]}
        self.merge_data = self.analyze(self.test_data, self.result_data, type='base')
        self.base_acc_for_pair = self.get_acc(self.merge_data)

        self.cf_merge_data = self.analyze(self.cf_test_data, self.cf_result_data, type='cf')
        self.cf_acc_for_pair = self.get_acc(self.cf_merge_data)


        self.cmp_data, self.cmp_data_ppl = self.merge_occ_base_cf(self.merge_data, self.cf_merge_data)

        ### bias based on probablity difference
        self.occ_bias_pair_ppl_list = self.cal_occ_bias_probablity(self.cmp_data_ppl)

        file_name = "occ_bias_pair_probablity_difference.csv"
        file_name = os.path.join(save_dir, file_name)
        self.write_csv(file_name, self.occ_bias_pair_ppl_list)

        
        ### bias based on outcome difference
        self.occ_bias_pair_list = self.cal_occ_bias_outcome(self.cmp_data)

        file_name = "occ_bias_pair_outcome_difference.csv"
        file_name = os.path.join(save_dir, file_name)
        self.write_csv(file_name, self.occ_bias_pair_list)


    def write_csv(self, file_name, data):
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0].keys()))
            writer.writeheader()
            for row in data:
                for key, value in row.items():
                    if isinstance(value, (int, float)) and key!='occtm_ratio' and key!='occtf_ratio':
                        row[key] = f"{100*value:.2f}"
                    elif isinstance(value, (int, float)):
                        row[key] = f"{value:.2f}"
                writer.writerow(row)

    def analyze(self, test_data, result_data, type='base'):
        # for each data in result_data, check the metric_result
        # for each data in test_data
        output = OrderedDict()
        for idx in range(len(test_data)):
            result = result_data[idx]
            test = test_data[idx]
            assert str(test["id"]) == str(result["id"]), f"test id is {test['id']}, result id is {result['id']}. id should be the same"
            occ = test["occ"]
            occ_sim = test["occ_sim"]
            gender = test["gender"]

            ### turn back to waiter
            if occ == "Waitress" and gender == "female" and type == "base":
                occ = "Waiter"
            elif occ == "Waitress" and gender == "male" and type == "cf":
                occ = "Waiter"
            if occ_sim == "Waitress" and gender == "female" and type == "base":
                occ_sim = "Waiter"
            elif occ_sim == "Waitress" and gender == "male" and type == "cf":
                occ_sim = "Waiter"
            
            key = f"{occ}+{occ_sim}+{gender}"
            if key not in output:
                output[key] = []
            sub_result = {
                "id": result["id"],
                "metric_result": result["metric_result"],
                "ppl_results": result["ppl_results"],
                "probs": result["probs"],
                "gt_choice": int(result["gt_choice"]),
            }
            output[key].append(sub_result)
        return output

    def get_acc(self, merge_data):
        def get_value(data_map, key):
            data = data_map.get(key, None)
            if data is None:
                return {"acc": 0, "acc_num": 0, "total_num": 0}
            else:
                return data
        output = OrderedDict() # f"{occ}+{occ_sim}+{gender}"
        output_for_pair = OrderedDict() # f"{occ}+{occ_sim}"

        for key, value in merge_data.items():
            acc = np.mean([x["metric_result"] for x in value])
            acc_num = np.sum([x["metric_result"] for x in value])
            acc = float(acc)
            acc_num = int(acc_num)
            total_num = len(value)
            output[key] = {"acc": acc, "acc_num": acc_num, "total_num": total_num}
            
        for row in self.similar_occ_data:
            occtm, occtf = row['job_male'], row['job_female']
            new_key = f"{occtm}+{occtf}"
            occtm_occtf_male = get_value(output, f"{occtm}+{occtf}+male")
            occtm_occtf_female = get_value(output, f"{occtm}+{occtf}+female")    
            occtf_occtm_male = get_value(output, f"{occtf}+{occtm}+male")
            occtf_occtm_female = get_value(output, f"{occtf}+{occtm}+female")  
            occtm_occtf_acc =  (occtm_occtf_male['acc_num'] + occtm_occtf_female['acc_num']) / (occtm_occtf_male['total_num'] + occtm_occtf_female['total_num'])   
            occtf_occtm_acc =  (occtf_occtm_male['acc_num'] + occtf_occtm_female['acc_num']) / (occtf_occtm_male['total_num'] + occtf_occtm_female['total_num'])
            acc_num = occtm_occtf_male['acc_num'] + occtm_occtf_female['acc_num'] + occtf_occtm_male['acc_num'] + occtf_occtm_female['acc_num']
            total_num = occtm_occtf_male['total_num'] + occtm_occtf_female['total_num'] + occtf_occtm_male['total_num'] + occtf_occtm_female['total_num']
            
            output_for_pair[new_key] = {
                "acc": 0.5 * (occtm_occtf_acc + occtf_occtm_acc), 
                "occtm": occtm_occtf_acc,
                "occtf": occtf_occtm_acc,
                "acc_num": acc_num, 
                "total_num": total_num
            }

        return output_for_pair


    def cal_occ_bias_outcome(self, cmp_data:OrderedDict):

        def get_value(data_map, key):
            tempalate = {
                "True->False": {"rate": 0, "cate_num": 0, "true_num": 0, "cmp_result": []},
                "False->True": {"rate": 0, "cate_num": 0, "false_num": 0, "cmp_result": []},
                "True->True": {"rate": 0, "cate_num": 0, "true_num": 0, "cmp_result": []},
                "False->False": {"rate": 0, "cate_num": 0, "false_num": 0, "cmp_result": []}
            }
            data = data_map.get(key, None)
            if data is None:
                return tempalate
            else:
                return data
        
        def cal_tf_ft(tf_data, ft_data):
            tf, ft = "True->False", "False->True"
            ans = {
                "cate_num": tf_data[tf]["cate_num"] + ft_data[ft]["cate_num"],
                "total_num": tf_data[tf]["true_num"] + ft_data[ft]["false_num"],
            }
            return ans
        
        def cal_bias_p(P1, P2):
            cate_num = P1["cate_num"] - P2["cate_num"]
            total_num = P1["total_num"] + P2["total_num"]
            return cate_num, total_num
        
        occpair_bias = []

        for row in self.similar_occ_data:
            occtm, occtf = row['job_male'], row['job_female']
            occtm_ratio, occtf_ratio = row['job_male_ratio'], row['job_female_ratio']

            new_key = f"{occtm}+{occtf}"
            occtm_occtf_male = get_value(cmp_data, f"{occtm}+{occtf}+male")
            occtm_occtf_female = get_value(cmp_data, f"{occtm}+{occtf}+female")    
            occtf_occtm_male = get_value(cmp_data, f"{occtf}+{occtm}+male")
            occtf_occtm_female = get_value(cmp_data, f"{occtf}+{occtm}+female")          

            ### cal four P
            P1 = cal_tf_ft(occtm_occtf_male, occtf_occtm_male)
            P2 = cal_tf_ft(occtm_occtf_female, occtf_occtm_female)
            P3 = cal_tf_ft(occtf_occtm_male, occtm_occtf_male)
            P4 = cal_tf_ft(occtf_occtm_female, occtm_occtf_female)
            
            p12_cate_num, p12_total_num = cal_bias_p(P1, P2)
            p34_cate_num, p34_total_num = cal_bias_p(P3, P4)
            bias = (p12_cate_num - p34_cate_num) / (p12_total_num + p34_total_num) if p12_total_num + p34_total_num != 0 else 0

            occpair_bias.append({
                "occtm": occtm, "occtf": occtf,
                "occtm_ratio": occtm_ratio, "occtf_ratio": occtf_ratio,
                "occtm_acc": self.base_acc_for_pair[occtm+'+'+occtf]['occtm'],
                "occtf_acc": self.base_acc_for_pair[occtm+'+'+occtf]['occtf'],
                "bias": bias,
                "acc": self.base_acc_for_pair[occtm+'+'+occtf]['acc'],
                "ipss": self.base_acc_for_pair[occtm+'+'+occtf]['acc'] * (1-abs(bias)),
            })
            
        

        return occpair_bias
    

    def cal_occ_bias_probablity(self, cmp_data:OrderedDict):
        occpair_bias = []

        def get_value(data_map, key):
            tempalate = {
                "mean_prob_gap": 0,
                "prob_gap_list_id": [],
            }
            data = data_map.get(key, None)
            if data is None:
                return tempalate
            else:
                return data
            
        for row in self.similar_occ_data:
            occtm, occtf = row['job_male'], row['job_female']
            occtm_ratio, occtf_ratio = row['job_male_ratio'], row['job_female_ratio']
            similarity = float(row['similarity'])
            
            occtm_male = get_value(cmp_data, f"{occtm}+{occtf}+male")
            occtm_female = get_value(cmp_data, f"{occtm}+{occtf}+female")    
            occtf_male = get_value(cmp_data, f"{occtf}+{occtm}+male")
            occtf_female = get_value(cmp_data, f"{occtf}+{occtm}+female")          

            ### cal four P
            ### bias = E[male-female]
            occtm_bias = occtm_male["mean_prob_gap"] * len(occtm_male["prob_gap_list_id"]) - occtm_female["mean_prob_gap"] * len(occtm_female["prob_gap_list_id"])
            occtm_bias = occtm_bias / (len(occtm_male["prob_gap_list_id"]) + len(occtm_female["prob_gap_list_id"]))

            occtf_bias = occtf_male["mean_prob_gap"] * len(occtf_male["prob_gap_list_id"]) - occtf_female["mean_prob_gap"] * len(occtf_female["prob_gap_list_id"])
            occtf_bias = occtf_bias / (len(occtf_male["prob_gap_list_id"]) + len(occtf_female["prob_gap_list_id"]))

            bias = 0.5 * (occtm_bias - occtf_bias)


            occpair_bias.append({
                "occtm": occtm, "occtf": occtf,
                "occtm_ratio": occtm_ratio, "occtf_ratio": occtf_ratio,
                "similarity": similarity,
                "occtm_bias": occtm_bias,
                "occtf_bias": occtf_bias,
                "occtm_acc": self.base_acc_for_pair[occtm+'+'+occtf]['occtm'],
                "occtf_acc": self.base_acc_for_pair[occtm+'+'+occtf]['occtf'],
                "bias": bias,
                "acc": self.base_acc_for_pair[occtm+'+'+occtf]['acc'],
                "ipss": self.base_acc_for_pair[occtm+'+'+occtf]['acc'] * (1-abs(bias)),
            })
        
        
        return occpair_bias

    
    def merge_occ_base_cf(self, merge_data, cf_merge_data):
        output = OrderedDict() 
        output_prob_gap = OrderedDict() 
        """
        # {f"{occ}+{occ_sim}+{gender}": {
            "True->False": {
                "rate": 0.5, "cate_num": 1, "true_num": 2, "cmp_result": [{"id":1}]
            },
            "False->True": {
                "rate": 0.5, "cate_num": 1, "false_num": 2, "cmp_result": [{"id":1}]
            }
            "True->True": {
                "rate": 0.5, "cate_num": 1, "true_num": 2, "cmp_result": [{"id":1}]
            },
            "False->False": {
                "rate": 0.5, "cate_num": 1, "false_num": 2, "cmp_result": [{"id":1}]
            }
        }}
        {f"{occ}+{occ_sim}+{gender}": {
            "mean_prob_gap": E[p_base - p_cf],
            "prob_gap_list_id": [{"id":1, "prob_gap": 0.5}]
        }}
        """
    
        for key, base_value in merge_data.items():
            cf_value = cf_merge_data.get(key, [])
            tf, ft, tt, ff, tf_list, ft_list, tt_list, ff_list = self.calculate_metrics(base_value, cf_value)
            t_num = tf + tt
            f_num = ft + ff    
            cmp_result = OrderedDict({
                "True->False": OrderedDict({"rate": tf / (tf + tt) if t_num!=0 else 0, "cate_num": tf, "true_num": tf + tt, "cmp_result": tf_list}),
                "False->True": OrderedDict({"rate": ft / (ft + ff) if f_num!=0 else 0, "cate_num": ft, "false_num": ft + ff, "cmp_result": ft_list}),
                "True->True": OrderedDict({"rate": tt / (tf + tt) if t_num!=0 else 0, "cate_num": tt, "true_num": tf + tt, "cmp_result": tt_list}),
                "False->False": OrderedDict({"rate": ff / (ft + ff) if f_num!=0 else 0, "cate_num": ff, "false_num": ft + ff, "cmp_result": ff_list})
            })
            output[key] = cmp_result

            mean_prob_gap, prob_gap_list_id = self.cal_prob_gap(base_value, cf_value)
            output_prob_gap[key] = OrderedDict({
                "mean_prob_gap": mean_prob_gap,
                "prob_gap_list_id": prob_gap_list_id
            })
        return output, output_prob_gap

    def cal_prob_gap(self, base_value, cf_value):
        prob_gap_list = []
        prob_gap_list_id = []
        mean_prob_gap = 0
        for base, cf in zip(base_value, cf_value):
            gt_choice = base["gt_choice"]
            prob_gap_list.append(base["probs"][gt_choice] - cf["probs"][gt_choice])
            prob_gap_list_id.append(
                OrderedDict({"id": base["id"], "prob_gap": base["probs"][gt_choice] - cf["probs"][gt_choice]})
            )
        mean_prob_gap = sum(prob_gap_list) / len(prob_gap_list) if len(prob_gap_list) != 0 else 0
        return mean_prob_gap, prob_gap_list_id

    def calculate_metrics(self, base, cf):
        tf, ft, tt, ff = 0, 0, 0, 0
        tf_list, ft_list, tt_list, ff_list = [], [], [], []
        for p, l in zip(base, cf):
            assert str(p["id"]) == str(l["id"]), "id should be the same"
            if p["metric_result"] is True:
                if l["metric_result"] is True:
                    tt += 1
                    tt_list.append(p["id"])
                else:
                    tf += 1
                    tf_list.append(p["id"])
            else:
                if l["metric_result"] is True:
                    ft += 1
                    ft_list.append(p["id"])
                else:
                    ff += 1
                    ff_list.append(p["id"])
        return tf, ft, tt, ff, tf_list, ft_list, tt_list, ff_list

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--inferencer_type", type=str, required=True)
    parser.add_argument("--base_result", type=str, required=True)
    parser.add_argument("--cf_result", type=str, required=True)
    parser.add_argument("--base_test_file", type=str, required=True)
    parser.add_argument("--cf_test_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./bias_results")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    save_dir = os.path.join(args.save_dir, args.model_name, args.eval_dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    EvalAna(args.base_result, args.base_test_file, args.cf_result, args.cf_test_file, save_dir)