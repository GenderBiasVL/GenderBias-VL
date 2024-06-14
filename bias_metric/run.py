import os
import subprocess

project_dir = "./"
result_base_dir = os.path.join(project_dir, "GenderBias-VL/lvlm_inference/LAMM/results")
base_test_file = os.path.join(project_dir, "GenderBias-VL/construction/vq_generation/test_meta_file/VLbias/occ_test_base_ask_gender.json")
cf_test_file = os.path.join(project_dir, "GenderBias-VL/construction/vq_generation/test_meta_file/VLbias/occ_test_cf_ask_gender.json")


model_names = ["LLaVA1.5", "LLaVA1.5-13b", "LLaVA1.6-13b", "MiniGPT-4-v2", "mPLUG-Owl2",
               "LLaMA-Adapter-v2", "InstructBLIP", "Otter", "LAMM", 
               "Kosmos2", "QwenVL", "InternLMXComposer", 
               "Shikra", "LLaVARLHF", "RLHFV"]


inferencer_type = "PPL"


base_datasets = ["OccBaseAskGender", "OccBaseAskGenderSwapOption", "OccBaseAskPerson", "OccBaseAskPersonSwapOption", "OccTextBaseAskGender", "OccTextBaseAskGenderSwapOption"]
cf_datasets = ["OccCfAskGender", "OccCfAskGenderSwapOption", "OccCfAskPerson", "OccCfAskPersonSwapOption", "OccTextCfAskGender", "OccTextCfAskGenderSwapOption"]
eval_datasets = ["VLbias", "VLbias_swap", "Vbias", "Vbias_swap", "Lbias", "Lbias_swap"]

def get_result_name(sub_dir):
    candidates = []
    for file in os.listdir(sub_dir):
        file = str(file)
        if file.startswith("OccBias_"):
            candidates.append(file)
    assert len(candidates) == 1 or len(candidates) == 0, f"check in {sub_dir}, candidates: {len(candidates)}"
    if len(candidates) == 0:
        return None
    return os.path.join(sub_dir, candidates[0])

if __name__ == "__main__":
    for model_name in model_names:
        for base_dataset, cf_dataset, eval_dataset in zip(base_datasets, cf_datasets, eval_datasets):
            base_sub_dir = os.path.join(result_base_dir, model_name, base_dataset, inferencer_type)
            cf_sub_dir = os.path.join(result_base_dir, model_name, cf_dataset, inferencer_type)
            base_result = get_result_name(base_sub_dir)
            cf_result = get_result_name(cf_sub_dir)
            print(f"model_name: {model_name}, eval_dataset: {eval_dataset}")

            command = [
                "python", "cal_bias.py",
                "--model_name", model_name,
                "--eval_dataset", eval_dataset,
                "--inferencer_type", inferencer_type,
                "--base_result", base_result,
                "--cf_result", cf_result,
                "--base_test_file", base_test_file,
                "--cf_test_file", cf_test_file
            ]

            subprocess.run(command)
        # exit(0)
