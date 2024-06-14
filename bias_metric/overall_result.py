from collections import OrderedDict, defaultdict
import copy
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

base_results_dir = "./bias_merge_swap_test/merge_bias_probability"
exp_dir = "./bias_merge_swap_test/"
os.makedirs(exp_dir, exist_ok=True)

def read_file(path):
    data = []
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key, value in row.items():
                try:
                    row[key] = float(value)
                except ValueError:
                    pass  
            data.append(row)
    return data


class EvalAna():
    def __init__(self):

        self.bias_types = ["VLbias", "Vbias", "Lbias"]

        self.model_names = ["LLaVA1.5", "LLaVA1.5-13b", "LLaVA1.6-13b", "MiniGPT-4-v2", "mPLUG-Owl2",
               "LLaMA-Adapter-v2", "InstructBLIP", "Otter", "LAMM", 
               "Kosmos2", "QwenVL", 
               "InternLMXComposer", 
               "Shikra", "LLaVARLHF", "RLHFV"] 
        

    def cal_ovl(self):
        result_all = OrderedDict()

        for model_name in self.model_names:
            for bias_type in self.bias_types:
                temp = self.cal_ovl_each(model_name, bias_type)
                result_all[f"{model_name}_{bias_type}"] = temp
        filename = os.path.join(exp_dir, 'overall', f"bias_overall.csv")
        self.write(result_all, filename)

    def write(self, result_all, file_name):
        
        model_data_list = []
        for model_name in self.model_names:
            merge_row = OrderedDict({"model_name": model_name})
            for bias_type in self.bias_types:
                temp = result_all[f"{model_name}_{bias_type}"]
                for key, value in temp.items():
                    merge_row[f"{bias_type}_{key}"] = value
            model_data_list.append(merge_row)

        self.write_csv(file_name, model_data_list)

    def cal_ovl_each(self, model_name, bias_type):
        target_file = os.path.join(base_results_dir, model_name, f"{model_name}_{bias_type}.csv")
        data = read_file(target_file)
        data = sorted(data, key=lambda x: x['bias'], reverse=True)
        acc_mean = np.mean([row['acc'] for row in data])
        bias_mean = np.mean([abs(row['bias']) for row in data])
        bias_max = np.max([row['bias'] for row in data])
        ipst_mean = np.mean([row['ipss'] for row in data])
        acc_delta_mean = np.mean([row['acc_delta'] for row in data])
        result = OrderedDict({
            "ipss_ovl": ipst_mean,
            "bias_ovl": bias_mean,
            "bias_max": bias_max,
            "acc_ovl": acc_mean,
            "acc_delta_ovl": acc_delta_mean
        })
        return result

    def write_csv(self, file_name, data, fieldnames=None):
        write_data = copy.deepcopy(data)
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, mode='w', newline='') as file:
            if fieldnames is None:
                writer = csv.DictWriter(file, fieldnames=list(write_data[0].keys()))
            else:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in write_data:
                for key, value in row.items():
                    if isinstance(value, (int, float)):
                        row[key] = f"{value:.2f}"
                writer.writerow(row)

if __name__ == '__main__':
    
    evalAna = EvalAna()
    evalAna.cal_ovl()
