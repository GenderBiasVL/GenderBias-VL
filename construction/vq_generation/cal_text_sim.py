import csv
import os
import clip


def setup_clip_model(device):
    clip_model, preprocess = clip.load("ViT-L/14", device="cuda:1")
    print(device)
    clip_model.to(device=device)
    clip_model.eval()
    return clip_model, preprocess


def read_file(path):
    data = []
    job_tend_to_male = {} # "job": woman_ratio
    job_tend_to_female = {}
    job_no_tend = {}
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
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
    print(f"job_no_tend: {len(job_no_tend)}")
    print(f"job_tend_to_male: {len(job_tend_to_male)}")
    print(f"job_tend_to_female: {len(job_tend_to_female)}")
    return data, job_tend_to_male, job_tend_to_female

def get_text_feature(text, clip_model):
    text = clip.tokenize(text).to("cuda")
    text_features = clip_model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

def write_file(file_name, data_to_out, fieldnames):
    f_out = open(file_name, 'w', newline='')
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for row in data_to_out:
        writer.writerow(row)

if __name__=="__main__":
    path = "../../resources/occ_us.csv"
    data, job_tend_to_male, job_tend_to_female = read_file(path)

    clip_model, preprocess = setup_clip_model("cuda")

    job_male_text = list(job_tend_to_male.keys())
    job_female_text = list(job_tend_to_female.keys())

    job_male_feature = get_text_feature(job_male_text, clip_model)
    job_female_feature = get_text_feature(job_female_text, clip_model)

    logits = job_male_feature @ job_female_feature.t()
    print(logits)

    result_map = {}
    for i in range(0, len(job_male_text)):
        for j in range(0, len(job_female_text)):
            result_map[(job_male_text[i], job_female_text[j])] = logits[i,j].item()
    sorted_result = sorted(result_map.items(), key=lambda x: x[1], reverse=True)
    


    data_to_out = []
    for item in sorted_result:
        sub_value = abs(job_tend_to_female[item[0][1]]-job_tend_to_male[item[0][0]])
        
        data_to_out.append({
            "job_male": item[0][0],
            "job_female": item[0][1],
            "similarity": item[1],
            "job_male_ratio": job_tend_to_male[item[0][0]],
            "job_female_ratio": job_tend_to_female[item[0][1]]
        })
    fieldnames = list(data_to_out[0].keys())
    os.makedirs("./similarity", exist_ok=True)
    write_file(f"./similarity/socc_text_sim.csv", data_to_out, list(data_to_out[0].keys()))


