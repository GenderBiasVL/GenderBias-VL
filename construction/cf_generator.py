import argparse
import csv
import os
import random
import clip
import numpy as np
import torch

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from tqdm import tqdm
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
)


clip_model = "ViT-L/14"

sd_model_dir = "./" #
sd_model_base = f"{sd_model_dir}/stable-diffusion-xl-base-1.0"
sd_model_refiner = f"{sd_model_dir}/stable-diffusion-xl-refiner-1.0"

sd_model_edit = f"{sd_model_dir}/timbrooks/instruct-pix2pix"

exp_dir = "../exp_flip" #  your target generation directory
fail_exp_dir = "../exp_fail" # your target generation directory
sub_exp = "xl_generate_base"
sub_cf_exp = "xl_generate_cf_via_instructpix2pix"

path = "../resources/occ_us.csv"
prompt_path = "../resources/prompts.txt"


class Generator:
    def __init__(self, args):
        self.args = args
        logfile = args.log_file
        os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
        self.flog = open(os.path.join(exp_dir, "logs", logfile), "w")
        os.makedirs(os.path.join(exp_dir, "prompt_records"), exist_ok=True)
        self.flog_propmpt = open(os.path.join(exp_dir, "prompt_records", logfile), "w")

        os.makedirs(os.path.join(fail_exp_dir, "logs"), exist_ok=True)
        self.fail_flog = open(os.path.join(fail_exp_dir, "logs", logfile), "w")
        os.makedirs(os.path.join(fail_exp_dir, "prompt_records"), exist_ok=True)
        self.fail_flog_propmpt = open(
            os.path.join(fail_exp_dir, "prompt_records", logfile), "w"
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_seed()
        self.init_all_occ()
        self.build_clip_model()
        self.build_stable_diffusion_model()
        print(f"start_index: {args.start_index}, end_index: {args.end_index}")
        self.generate_images(args.start_index, args.end_index)

    def setup_seed(self, seed=0):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def build_clip_model(self):
        self.clip_model, self.clip_model_preprocess = clip.load(
            clip_model, device=self.device
        )
        self.clip_model.eval()

    def build_stable_diffusion_model(self):

        self.base = StableDiffusionXLPipeline.from_pretrained(
            sd_model_base,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.base.to(self.device)
        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            sd_model_refiner,
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to(self.device)
        self.edit_model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            sd_model_edit,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.edit_model.to(self.device)
        self.generator = torch.Generator("cuda").manual_seed(0)


    def read_occ_file(self):
        data = []
        prompt_data = []
        occ_gender_prompts_map = {}

        with open(prompt_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if len(line.strip()) == 0:
                    continue
                str_ = str(line.strip('"').strip())
                prompt_data.append(str_)
        sublists = [prompt_data[i : i + 5] for i in range(0, len(prompt_data), 5)]

        with open(path, "r") as file:
            reader = csv.DictReader(file)
            index = 0
            for row in reader:
                data.append(row["occupation"])

                occ_gender_prompts_map[(row["occupation"], "female")] = sublists[index]
                index += 1
                occ_gender_prompts_map[(row["occupation"], "male")] = sublists[index]
                index += 1
                if index == len(sublists):
                    break

        return data, occ_gender_prompts_map

    def init_all_occ(self):
        self.genders = ["female", "male"]
        self.cf_gender_map = {
            "female": "male",
            "male": "female",
            "her": "him",
            "him": "her",
            "woman": "man",
            "man": "woman",
        }
        self.gender_reso = {
            "female": "her",
            "male": "him",
            "her": "her",
            "him": "him",
            "woman": "her",
            "man": "him",
        }

        self.all_occ_list, self.occ_gender_prompts_map = self.read_occ_file()
        self.neg_list = [
            "((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))) , ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"
        ]

    def generate_images(self, start_index=0, end_index=1):
        each_occ_nums = 20
        each_occ_batch = 4  
        n_steps = 50  
        high_noise_frac = 0.8
        genders = ["female", "male"]
        each_occ_trial = 100
        edit_guidance_scale = 7.5
        img_height, img_width = 768, 768
        
        for occ in tqdm(self.all_occ_list[start_index:end_index]):

            occ_name = "_".join(occ.split(" "))
            occ_dir = os.path.join(exp_dir, sub_exp, occ_name)
            cf_occ_dir = os.path.join(exp_dir, sub_cf_exp, occ_name)

            fail_occ_dir = os.path.join(fail_exp_dir, sub_exp, occ_name)
            fail_cf_occ_dir = os.path.join(fail_exp_dir, sub_cf_exp, occ_name)

            for gender in genders:
                output_dir = os.path.join(occ_dir, gender)
                os.makedirs(output_dir, exist_ok=True)
                cf_output_dir = os.path.join(cf_occ_dir, gender)
                os.makedirs(cf_output_dir, exist_ok=True)

                fail_output_dir = os.path.join(fail_occ_dir, gender)
                os.makedirs(fail_output_dir, exist_ok=True)
                fail_cf_output_dir = os.path.join(fail_cf_occ_dir, gender)
                os.makedirs(fail_cf_output_dir, exist_ok=True)

                prompts = self.occ_gender_prompts_map[(occ, gender)]
                cur_num, fail_cur_num = 0, 0
                trial = 0
                batch_i = 0
                while trial < each_occ_trial:
                    prompt_id = batch_i % len(prompts)
                    prompt = prompts[batch_i % len(prompts)]
                    batch_i += 1  
                    print(f"occ: {occ}, gender: {gender}, prompt: {prompt}")
                    prompt_list = [prompt] * each_occ_batch
                    neg_prompt_list = self.neg_list * each_occ_batch
                    image = self.base(
                        prompt=prompt_list,
                        negative_prompt=neg_prompt_list,
                        num_inference_steps=n_steps,
                        denoising_end=high_noise_frac,
                        output_type="latent",
                        generator=self.generator,
                        height=img_height,
                        width=img_width,
                    ).images
                    base_images = self.refiner(
                        prompt=prompt_list,
                        negative_prompt=neg_prompt_list,
                        num_inference_steps=n_steps,
                        denoising_start=high_noise_frac,
                        image=image,
                        generator=self.generator,
                    ).images

                    cf_gender = self.cf_gender_map[gender]
                    cf_prompt = f"turn {self.gender_reso[gender]} into a {cf_gender}"
                    prompt_list = [cf_prompt] * each_occ_batch

                   
                    cf_images = self.edit_model(
                        prompt=prompt_list,
                        image=base_images,
                        edit_guidance_scale=edit_guidance_scale,
                        generator=self.generator,
                    ).images

                    test_labels = []
                    for _gender in [gender, cf_gender]:
                        test_labels.append(f"{_gender}")
                    answer_base = f"{gender}"
                    answer_cf = f"{cf_gender}"
                    test_label_emb = self.get_clip_text_emb(test_labels)
                    base_img_emb = self.get_clip_image_emb(base_images)
                    cf_img_emb = self.get_clip_image_emb(cf_images)
                    score_base = (100.0 * base_img_emb @ test_label_emb.T).softmax(
                        dim=-1
                    )  
                    score_cf = (100.0 * cf_img_emb @ test_label_emb.T).softmax(dim=-1)

                    for _ind in range(len(base_images)):
                        pred_base = test_labels[score_base[_ind].argmax()]
                        pred_cf = test_labels[score_cf[_ind].argmax()]
                        if pred_base == answer_base and pred_cf == answer_cf:
                            img_base = base_images[_ind]
                            img_base.save(
                                f"{output_dir}/{gender}_{occ_name}_{cur_num}.png"
                            )
                            img_cf = cf_images[_ind]
                            img_cf.save(
                                f"{cf_output_dir}/{gender}_{occ_name}_{cur_num}.png"
                            )

                            print(
                                f"{test_labels}\tbase: {score_base[_ind].tolist()}, cf: {score_cf[_ind].tolist()}"
                            )
                            self.flog.write(
                                f"{test_labels}\tbase: {score_base[_ind].tolist()}, cf: {score_cf[_ind].tolist()}\n"
                            )
                            self.flog.flush()
                            self.flog_propmpt.write(
                                f"{gender}_{occ_name}_{cur_num}.png,{prompt_id},{prompt}\n"
                            )
                            self.flog_propmpt.flush()
                            cur_num += 1
                            print("gender flip")
                        else:
                            img_base = base_images[_ind]
                            img_base.save(
                                f"{fail_output_dir}/{gender}_{occ_name}_{fail_cur_num}.png"
                            )
                            img_cf = cf_images[_ind]
                            img_cf.save(
                                f"{fail_cf_output_dir}/{gender}_{occ_name}_{fail_cur_num}.png"
                            )

                            print(
                                f"{test_labels}\tbase: {score_base[_ind].tolist()}, cf: {score_cf[_ind].tolist()}"
                            )
                            self.fail_flog.write(
                                f"{test_labels}\tbase: {score_base[_ind].tolist()}, cf: {score_cf[_ind].tolist()}\n"
                            )
                            self.fail_flog.flush()
                            self.fail_flog_propmpt.write(
                                f"{gender}_{occ_name}_{cur_num}.png,{prompt_id},{prompt}\n"
                            )
                            self.fail_flog_propmpt.flush()
                            fail_cur_num += 1
                            print("gender fail")
                        trial += 1
                        print(
                            f"answer_base: {answer_base}, pred_base:{pred_base}, answer_cf: {answer_cf}, pred_cf:{pred_cf}"
                        )

    def get_clip_image_emb(self, imgs):
        tensor_list = [self.clip_model_preprocess(img) for img in imgs]
        imgs = torch.stack(tensor_list)

        imgs = imgs.to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(imgs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_clip_text_emb(self, texts):
        with torch.no_grad():
            text = clip.tokenize(texts).to(self.device)
            text_features = self.clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="log.txt",
    )
    args = parser.parse_args()
    Generator(args)
