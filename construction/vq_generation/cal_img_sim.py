import csv
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import seaborn
import torch
import clip
from PIL import Image
from tqdm import tqdm

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

test_root = "xl_generate_base" # the path of the base image data


def load_img_dirs(img_dirs):
    img_list = []
    for img_path in os.listdir(img_dirs):
        img_path = os.path.join(img_dirs, img_path)
        img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        img_list.append(img)
        del img
    if len(img_list) == 0:
        return None
    img_list = torch.concat(img_list, dim=0)
    return img_list


def read_occ_file():
    path = "../../resources/occ_us.csv"
    data = []
    job_tend_to_male = {} # "job": woman_ratio
    job_tend_to_female = {}
    job_no_tend = {}
    cnt = 0
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row['occupation'])
            job = row['occupation']
            woman_ratio = row['Women']
            if woman_ratio == '-':
                print(f"skip {job} {woman_ratio}")
                job_no_tend[job] = 50
                continue
            woman_ratio = float(woman_ratio)
            if woman_ratio==50:
                print(f"go to female {job} {woman_ratio}")
            if woman_ratio < 50:
                job_tend_to_male[job] = woman_ratio
            elif woman_ratio >= 50:
                job_tend_to_female[job] = woman_ratio
    return data, job_tend_to_male, job_tend_to_female


def get_occ_features(job_list):
    gender_occ_img_features = {}
    male_features = []
    female_features = []
    genders = ["female", "male"]
    with torch.no_grad():
        for occ in tqdm(job_list):
            occ_name = "_".join(occ.split(" "))
            occ_dir = os.path.join(test_root, occ_name)
            for gender in genders:
                sub_dir = os.path.join(occ_dir, gender)
                img_list = load_img_dirs(sub_dir)
                if img_list is None:
                    mean_img_features = torch.zeros(mean_img_features.shape).to(device)
                else:
                    _img_features = model.encode_image(img_list)
                    mean_img_features = _img_features.mean(dim=0)
                    del _img_features
                    mean_img_features = mean_img_features / mean_img_features.norm(dim=0, keepdim=True)
                gender_occ_img_features[(occ, gender)] = mean_img_features
                if gender == "female":
                    female_features.append(mean_img_features)
                else:
                    male_features.append(mean_img_features)
                del img_list
        female_features = torch.stack(female_features, dim=0)
        male_features = torch.stack(male_features, dim=0)

    return gender_occ_img_features, female_features, male_features


def write_file(file_name, data_to_out, fieldnames):
    f_out = open(file_name, 'w', newline='')
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for row in data_to_out:
        writer.writerow(row)

if __name__=='__main__':
    
    feature_path = "features.pkl"
    all_occ_list, job_tend_to_male, job_tend_to_female = read_occ_file()
    
    job_male_text = list(job_tend_to_male.keys())
    print(job_male_text)
    print(job_tend_to_male)
    job_female_text = list(job_tend_to_female.keys())

    if os.path.exists(feature_path):
        with open(feature_path, "rb") as f:
            features = pickle.load(f)
            m_f_fea, m_m_fea = features["m_f_fea"], features["m_m_fea"]
            f_f_fea, f_m_fea = features["f_f_fea"], features["f_m_fea"]
            m_f_fea = m_f_fea.to(device)
            m_m_fea = m_m_fea.to(device)
            f_f_fea = f_f_fea.to(device)
            f_m_fea = f_m_fea.to(device)
    else:
        job_tend_to_male_features, m_f_fea, m_m_fea = get_occ_features(job_male_text)
        job_tend_to_female_features, f_f_fea, f_m_fea = get_occ_features(job_female_text)

        with open(feature_path, "wb") as f:
            pickle.dump({
                "m_f_fea": m_f_fea.cpu(),
                "m_m_fea": m_m_fea.cpu(),
                "f_f_fea": f_f_fea.cpu(),
                "f_m_fea": f_m_fea.cpu()
            }, f)

    # cal the sim of occ 
    with torch.cuda.amp.autocast(dtype=torch.float16):
        logits_sim_f = m_f_fea @ f_f_fea.t()
        logits_sim_m = m_m_fea @ f_m_fea.t()
        
    logits_avg = torch.zeros_like(logits_sim_f)  

    for i in range(logits_avg.shape[0]):
        for j in range(logits_avg.shape[1]):
            if logits_sim_f[i, j] == 0:
                logits_avg[i, j] = logits_sim_m[i, j]
            elif logits_sim_m[i, j] == 0:
                logits_avg[i, j] = logits_sim_f[i, j]
            else:
                logits_avg[i, j] = (logits_sim_f[i, j] + logits_sim_m[i, j]) / 2

    result_map = {}
    for i in range(0, len(job_male_text)):
        for j in range(0, len(job_female_text)):
            result_map[(job_male_text[i], job_female_text[j])] = logits_avg[i,j].item()
    sorted_result = sorted(result_map.items(), key=lambda x: x[1], reverse=True)

    data_to_out = []
    for item in sorted_result:
        data_to_out.append({
            "job_male": item[0][0],
            "job_female": item[0][1],
            "similarity": item[1],
            "job_male_ratio": job_tend_to_male[item[0][0]],
            "job_female_ratio": job_tend_to_female[item[0][1]]
        })
    fieldnames = list(data_to_out[0].keys())
    os.makedirs("./similarity", exist_ok=True)
    write_file(f"./similarity/occ_img_sim.csv", data_to_out, list(data_to_out[0].keys()))

    