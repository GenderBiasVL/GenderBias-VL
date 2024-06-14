from collections import OrderedDict
import copy
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

exp_dir = "./bias_merge_swap_test"
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


def write_csv(file_name, data, mode='w', fieldnames=None):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    file_flag = os.path.exists(file_name)

    with open(file_name, mode, newline='') as file:
        if fieldnames is None:
            writer = csv.DictWriter(file, fieldnames=list(data[0].keys()))
        else:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
        # if not file_flag:
        writer.writeheader()
        for row in data:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    row[key] = f"{value:.2f}"
            writer.writerow(row)

def cal_micro_occ_bias(data):
    """
    {
        occ: {
            occ_pairs: [(occ, occ_sim, occtm_bias,occtf_bias,bias), ...]
            measure_bias
        }
    }
    """
    occtm_map = OrderedDict()
    occtf_map = OrderedDict()
    occtm_list = []
    occtf_list = []
    for row in data:
        occ = row['occtm']
        if occ not in occtm_map:
            occtm_map[occ] = {'occ_pairs':[], 'bias':[]}
        occtm_map[occ]['occ_pairs'].append(row)
        occtm_map[occ]['bias'].append(row['occtm_bias'])

        occ = row['occtf']
        if occ not in occtf_map:
            occtf_map[occ] = {'occ_pairs':[], 'bias':[]}
        occtf_map[occ]['occ_pairs'].append(row)
        occtf_map[occ]['bias'].append(row['occtf_bias'])

    for occ in occtm_map:
        mean_bias = np.mean(occtm_map[occ]['bias'])
        temp = OrderedDict({'occ': occ, 'micro_bias': mean_bias})
        occtm_list.append(temp)
        
    for occ in occtf_map:
        mean_bias = np.mean(occtf_map[occ]['bias'])
        temp = OrderedDict({'occ': occ, 'micro_bias': mean_bias})
        occtf_list.append(temp)

    occtm_list = sorted(occtm_list, key=lambda x: x['micro_bias'], reverse=True)
    occtf_list = sorted(occtf_list, key=lambda x: x['micro_bias'], reverse=False)
    return occtm_list, occtf_list

def merge(result, result_swap, model_name, bias_type):
    merge_data = []
    result_data = read_file(result)
    result_swap_data = read_file(result_swap)
    for i in range(0, len(result_data)):
        result_row = result_data[i]
        result_swap_row = result_swap_data[i]
        merge_row = copy.deepcopy(result_row)
        for key, value in result_row.items():
            if isinstance(value, (int, float)):
                merge_row[key] = (value + result_swap_row[key]) / 2
        merge_row['acc_delta'] = abs(result_row['occtm_acc'] - result_swap_row['occtm_acc']) + abs(result_row['occtf_acc'] - result_swap_row['occtf_acc'])
        merge_row['acc_delta'] = merge_row['acc_delta'] / 2
        merge_data.append(merge_row)
    merge_data = sorted(merge_data, key=lambda x: x['bias'], reverse=True)
    
    occtm_list, occtf_list = cal_micro_occ_bias(merge_data)
    filednames = 'occ,micro_bias'.split(',')
    filename = f"{model_name}_{bias_type}_occtm.csv"
    filename = os.path.join(exp_dir, 'merge_micro_occ', model_name, filename)
    write_csv(filename, occtm_list, mode='w', fieldnames=filednames)

    filename = f"{model_name}_{bias_type}_occtf.csv"
    filename = os.path.join(exp_dir, 'merge_micro_occ', model_name, filename)
    write_csv(filename, occtf_list, mode='w', fieldnames=filednames)

    mean_acc_delta = np.mean([row['acc_delta'] for row in merge_data])
    print(f"{model_name} {bias_type} mean acc delta: {mean_acc_delta:.2f}")
    filename = f"{model_name}_{bias_type}.csv"
    filename = os.path.join(exp_dir, 'merge_bias_probability', model_name, filename)
    write_csv(filename, merge_data, mode='w')


def merge_outcome(result, result_swap, model_name, bias_type):
    merge_data = []
    result_data = read_file(result)
    result_swap_data = read_file(result_swap)
    for i in range(0, len(result_data)):
        result_row = result_data[i]
        result_swap_row = result_swap_data[i]
        merge_row = copy.deepcopy(result_row)
        for key, value in result_row.items():
            if isinstance(value, (int, float)):
                merge_row[key] = (value + result_swap_row[key]) / 2
        merge_row['acc_delta'] = abs(result_row['occtm_acc'] - result_swap_row['occtm_acc']) + abs(result_row['occtf_acc'] - result_swap_row['occtf_acc'])
        merge_row['acc_delta'] = merge_row['acc_delta'] / 2
        merge_data.append(merge_row)
    merge_data = sorted(merge_data, key=lambda x: x['bias'], reverse=True)
    

    mean_acc_delta = np.mean([row['acc_delta'] for row in merge_data])
    print(f"{model_name} {bias_type} mean acc delta: {mean_acc_delta:.2f}")
    filename = f"{model_name}_{bias_type}.csv"
    filename = os.path.join(exp_dir, 'merge_bias_outcome', model_name, filename)
    write_csv(filename, merge_data, mode='w')

if __name__ == '__main__':
    result_base_dir = "./bias_results/"

    eval_datasets = ["VLbias", "VLbias_swap", "Vbias", "Vbias_swap", "Lbias", "Lbias_swap"]
    

    model_names = ["LLaVA1.5", "LLaVA1.5-13b", "LLaVA1.6-13b", "MiniGPT-4-v2", "mPLUG-Owl2",
               "LLaMA-Adapter-v2", "InstructBLIP", "Otter", "LAMM", 
               "Kosmos2", "QwenVL", 
               "InternLMXComposer", 
               "Shikra", "LLaVARLHF", "RLHFV"] 
    
    
    for i in range(0, len(eval_datasets), 2):
        all_data = []
        for model_name in model_names:
            model_dir = os.path.join(result_base_dir, model_name)
            target_name = "occ_bias_pair_probablity_difference.csv"
            sub_file = os.path.join(model_dir, eval_datasets[i], target_name)
            sub_file_swap = os.path.join(model_dir, eval_datasets[i+1], target_name)
            merge(sub_file, sub_file_swap, model_name, eval_datasets[i])

            target_name = "occ_bias_pair_outcome_difference.csv"
            sub_file = os.path.join(model_dir, eval_datasets[i], target_name)
            sub_file_swap = os.path.join(model_dir, eval_datasets[i+1], target_name)
            merge_outcome(sub_file, sub_file_swap, model_name, eval_datasets[i])

