import argparse
import csv
from functools import partial
from io import TextIOWrapper
import os
import clip
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import DataLoader

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


exp_dir = "./exp_image_filter"


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, transform, root_path=None,
                 occupation_path:str=None):
        self.root_path = root_path # image root path
        self.transform = transform
        self.genders = ["female", "male"]
        self.gender_map = {"man":"woman", "woman":"man", "male": "female", "female":"male"}

        self.all_occ_list = self.read_occ_file(occupation_path)

                
        self.data = []
        self.filenames = []
        for occ in self.all_occ_list:
            occ_name = "_".join(occ.split(" "))
            occ_dir = os.path.join(root_path, occ_name)
            for gender in self.genders:
                sub_dir = os.path.join(occ_dir, gender)
                for img_path in os.listdir(sub_dir):
                    self.data.append(os.path.join(sub_dir, img_path))
                    self.filenames.append(img_path)

    def __len__(self):
        return len(self.data)
    
    def read_occ_file(self, path):
        data = []
        with open(path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row['occupation'])
        return data

    def is_all_black(self, img):
        image = np.array(img)
        return np.all(image == [0, 0, 0])

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        nsfw_label = self.is_all_black(img)
        img, _ = self.transform(img, None)
        return self.filenames[idx], img, nsfw_label

def inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    for i in range(tensor.shape[0]):
        tensor[i] = (tensor[i] * std[i]) + mean[i]
    return tensor

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

class Filter():
    def __init__(self, args):
        self.args = args
        
        os.makedirs(os.path.join(exp_dir, self.args.sub_exp), exist_ok=True)
        logdir = os.path.join(exp_dir, self.args.sub_exp, "logs")
        os.makedirs(logdir, exist_ok=True)

        logfile = args.log_file
        self.log = open(os.path.join(logdir, logfile), "w")
        test_case_path = "test_case.txt"
        self.test_case = open(os.path.join(logdir, test_case_path), "w")
        
        multi_face_path = "multi_face.txt"
        self.multi_face = open(os.path.join(logdir, multi_face_path), "w")
        no_face_path = "no_face.txt"
        self.no_face = open(os.path.join(logdir, no_face_path), "w")
        nsfw_image_path = "nsfw_image.txt"
        self.nsfw_image = open(os.path.join(logdir, nsfw_image_path), "w")
        diff_box_path = "diff_box.txt"
        self.diff_box = open(os.path.join(logdir, diff_box_path), "w")


        self.multi_face_dir = os.path.join(exp_dir, self.args.sub_exp, "multi_face_imgs")
        os.makedirs(self.multi_face_dir, exist_ok=True)
        self.no_face_dir = os.path.join(exp_dir, self.args.sub_exp, "no_face_imgs")
        os.makedirs(self.no_face_dir, exist_ok=True)
        self.nsfw_image_dir = os.path.join(exp_dir, self.args.sub_exp, "nsfw_imgs")
        os.makedirs(self.nsfw_image_dir, exist_ok=True)
        self.diff_box_dir = os.path.join(exp_dir, self.args.sub_exp, "diff_box_imgs")
        os.makedirs(self.diff_box_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_seed()
        self.build_dataset()
        self.build_model()

    def setup_seed(self, seed=0):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def write_msg(self, file:TextIOWrapper, msg):
        file.write(msg)
        file.flush()

    def build_dataset(self):
        self.transform = T.Compose([
                T.RandomResize([800], max_size=1333), #800
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),   
        ])
        self.base_dataset = BaseDataset(self.transform, root_path=self.args.base_root, occupation_path=self.args.occ_path)
        self.base_dataloader = DataLoader(self.base_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        self.cf_dataset = BaseDataset(self.transform, root_path=self.args.cf_root, occupation_path=self.args.occ_path)
        self.cf_dataloader = DataLoader(self.cf_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

    def build_model(self):
        args = self.args
        config_file = args.config  # change the path of the model config file
        grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
        grounding_model = self._load_grounding_dino_model(config_file, grounded_checkpoint, device=self.device)
        self.grounding_model = grounding_model

    def _load_grounding_dino_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        model = model.to(device)
        _ = model.eval()
        return model

    def save_bbox_imgs(self, img, base_bboxes, base_phrases, save_dir, img_name):    
        H, W = img.shape[1], img.shape[2]
        for i in range(base_bboxes.size(0)):
            base_bboxes[i] = base_bboxes[i] * torch.Tensor([W, H, W, H])
            base_bboxes[i][:2] -= base_bboxes[i][2:] / 2
            base_bboxes[i][2:] += base_bboxes[i][:2]
        base_bboxes = base_bboxes.cpu()

        img = inverse_normalize(img)
        img = Image.fromarray(np.uint8(img.transpose(1, 2, 0) * 255))

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        for box, label in zip(base_bboxes, base_phrases):
            show_box(box.numpy(), plt.gca(), label)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, img_name), bbox_inches="tight")


    def main_filter(self):
        self.write_msg(self.log,"Start filtering\n")
        args = self.args
        detect_prompt = args.detect_prompt
        box_threshold = args.box_threshold
        text_threshold = args.text_threshold
        
        for i, (base_data, cf_data) in tqdm(enumerate(zip(self.base_dataloader, self.cf_dataloader)), total=len(self.base_dataloader)):
            base_names, base_imgs, base_nsfw_labels = base_data
            cf_names, cf_imgs, cf_nsfw_labels = cf_data
            base_imgs, cf_imgs = base_imgs.to(self.device), cf_imgs.to(self.device)
            detect_prompts = [detect_prompt] * base_imgs.shape[0]
            
            base_bboxes_batch, base_predicts_batch, base_phrases_batch = self.predict_batch(self.grounding_model, base_imgs, detect_prompts, box_threshold, text_threshold, device=self.device)
            cf_bboxes_batch, cf_predicts_batch, cf_phrases_batch = self.predict_batch(self.grounding_model, cf_imgs, detect_prompts, box_threshold, text_threshold, device=self.device)

            for j in range(base_imgs.shape[0]):
                base_name = base_names[j]
                cf_name = cf_names[j]
                base_phrases = base_phrases_batch[j]
                cf_phrases = cf_phrases_batch[j]
                base_predicts = base_predicts_batch[j]
                cf_predicts = cf_predicts_batch[j]
                base_bboxes = base_bboxes_batch[j]
                cf_bboxes = cf_bboxes_batch[j]
                
                if len(base_bboxes) > 1: # multi faces
                    self.write_msg(self.log, f"{base_name} has multi faces\n")
                    self.save_bbox_imgs(base_imgs[j], base_bboxes, base_phrases, self.multi_face_dir, base_name)
                    self.write_msg(self.multi_face, f"{base_name}\n")
                elif len(base_bboxes) == 0:
                    self.write_msg(self.log, f"{base_name} has no face\n")
                    self.save_bbox_imgs(base_imgs[j], base_bboxes, base_phrases, self.no_face_dir, base_name)
                    self.write_msg(self.no_face, f"{base_name}\n")
                elif len(base_bboxes) != len(cf_bboxes):
                    self.write_msg(self.log, f"{base_name} has different number of faces\n")
                    self.save_bbox_imgs(base_imgs[j], base_bboxes, base_phrases, self.diff_box_dir, 'base_'+base_name)
                    self.save_bbox_imgs(cf_imgs[j], cf_bboxes, cf_phrases, self.diff_box_dir, 'cf_'+cf_name)
                    self.write_msg(self.diff_box, f"{base_name}\n")
                elif base_nsfw_labels[j] or cf_nsfw_labels[j]:
                    self.write_msg(self.log, f"{base_name} is nsfw\n")
                    self.save_bbox_imgs(base_imgs[j], base_bboxes, base_phrases, self.nsfw_image_dir, 'base_'+base_name)
                    self.save_bbox_imgs(cf_imgs[j], cf_bboxes, cf_phrases, self.nsfw_image_dir, 'cf_'+cf_name)
                    self.write_msg(self.nsfw_image, f"{base_name}\n")
                else:
                    self.write_msg(self.test_case, f"{base_name}\n")
        self.write_msg(self.log, "End filtering\n")

    def predict_batch(self, model, image, captions, box_threshold, text_threshold, device="cpu"):
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image, captions=captions)
        _logits = outputs["pred_logits"].cpu().sigmoid()  # torch.Size([num_batch, 900, 256]) 
        _boxes = outputs["pred_boxes"].cpu() # torch.Size([num_batch, 900, 4])
        
        prediction_logits = _logits.clone()
        prediction_boxes = _boxes.clone()
        mask = prediction_logits.max(dim=2)[0] > box_threshold # mask: torch.Size([num_batch, 256])

        bboxes_batch = []
        predicts_batch = []
        phrases_batch = [] # list of lists
        tokenizer = model.tokenizer
        tokenized = tokenizer(captions[0])
        for i in range(prediction_logits.shape[0]):
            logits = prediction_logits[i][mask[i]]  # logits.shape = (n, 256)
            phrases = [
                        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
                        for logit # logit is a tensor of shape (256,) torch.Size([256])
                        in logits # torch.Size([n, 256])
                    ]
            boxes = prediction_boxes[i][mask[i]]  # boxes.shape = (n, 4)
            phrases_batch.append(phrases)
            bboxes_batch.append(boxes)
            predicts_batch.append(logits.max(dim=1)[0])
        
        return bboxes_batch, predicts_batch, phrases_batch
    
if __name__=="__main__":
    occ_path = "../resources/occ_us.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1)
    parser.add_argument("--log_file", type=str, default='log.txt')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_root", type=str, default="")
    parser.add_argument("--cf_root", type=str, default="")
    parser.add_argument("--occ_path", type=str, default=occ_path)
    parser.add_argument("--config", type=str, default="GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounded_checkpoint", type=str, default="groundingdino_swint_ogc.pth")
    parser.add_argument("--detect_prompt", type=str, default="a person.")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--sub_exp", type=str, default="filter")
    args = parser.parse_args()
    image_filter = Filter(args)
    image_filter.main_filter()