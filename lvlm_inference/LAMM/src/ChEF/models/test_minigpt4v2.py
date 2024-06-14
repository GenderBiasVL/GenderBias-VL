import torch
import torch.nn.functional as F
import numpy as np
from .minigpt4_v2.common.config import Config
from .minigpt4_v2.common.registry import registry
from .minigpt4_v2.conversation.conversation import Chat, CONV_VISION_minigptv2, CONV_VISION_Vicuna0

# imports modules for registration
from .minigpt4_v2.models import *
from .minigpt4_v2.processors import *
from .test_base import TestBase
from .utils import get_image
from .minigpt4_v2.common.eval_utils import prepare_texts, eval_parser

class TestMiniGPT4V2(TestBase):
    def __init__(self, 
                 device,
                 cfg_path = 'ChEF/models/minigpt4_v2/minigptv2_eval.yaml',
                 **kwargs
                 ):
        # parser = eval_parser()
        # args = parser.parse_args()
        # args.cfg_path = cfg_path
        cfg = Config(cfg_path, None)
        
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)
        
        vis_processor_cfg = cfg.preprocess_cfg.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        
        model.eval()
        self.model, self.vis_processor = model, vis_processor
        print(self.vis_processor.transform)

        self.model.llama_model = self.model.llama_model.float().to(device)
        self.dtype = torch.float16
        self.device = device
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.chat = Chat(model, vis_processor, device=self.device)
        # self.chat.move_stopping_criteria_device(self.device, dtype=self.dtype)

    def build_input_image(self, image_list):
        images = self.get_image_list(image_list)
        image_tensor = []
        for image in images:
            image_tensor.append(self.vis_processor(image).unsqueeze(0).to(self.device))
        return image_tensor
    
    def build_conversation(self, idx, image_tensor_list, prompt, CoT_answer_list=None, batch_answers=None, **kwargs):
        conv = CONV_VISION_minigptv2.copy()
        # conv = CONV_VISION_Vicuna0.copy()
        conv.append_message(conv.roles[0], " ".join(["<Img><ImageHere></Img>"] * len(image_tensor_list)))
        self.chat.ask(prompt, conv)
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()
        if CoT_answer_list is not None:
            query += CoT_answer_list[idx]
        if batch_answers is not None:
            query += ' ' + batch_answers[idx]
        return query

    @torch.no_grad()
    def batch_generate(self, batch_images, batch_prompt, max_new_tokens, **kwargs):
        outputs = []
        for idx, (image_list, prompt) in enumerate(zip(batch_images, batch_prompt)):
            input_image_list = self.build_input_image(image_list)
            input_prompt = self.build_conversation(idx, input_image_list, prompt, generate=True, **kwargs)
            output = self.do_generate(input_image_list, input_prompt, max_new_tokens)
            outputs.append(output)
        return outputs
    
    @torch.no_grad()
    def ppl_inference(self, batch_images, batch_prompt, batch_options, **kwargs):
        input_images, input_prompts = [], []
        for idx, (image_list, prompt) in \
                enumerate(zip(batch_images, batch_prompt)):
            # import pdb;pdb.set_trace()
            input_image_list = self.build_input_image(image_list)
            input_prompt = self.build_conversation(idx, input_image_list, prompt, generate=False, **kwargs)
            input_prompts.append(input_prompt)
            input_images.append(input_image_list)
        return self.do_ppl(input_images, input_prompts, batch_options, **kwargs)
    
    def do_generate(self, input_image_list: list, input_prompt: str, max_new_tokens, **kwargs):
        img_embes_list = []
        for img_tensor in input_image_list:
            image_emb, _ = self.model.encode_img(img_tensor)
            img_embes_list.append(image_emb)
        embes, _ = self.chat.get_context_emb(input_prompt, img_embes_list)

        outputs = self.model.llama_model.generate(
            inputs_embeds=embes,
            max_new_tokens=20, # max_new_tokens
            stopping_criteria=self.chat.stopping_criteria,
            num_beams=1, # 5
            do_sample=False,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0, # -1.0
            temperature=1.0, 
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        # if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
        #     output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        # output_text = output_text.split('###')[0]  # remove the stop sign '###'
        # output_text = output_text.split('Assistant:')[-1].strip()
        output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
        output_text = output_text.replace("<s>", "")
        output_text = output_text.split(r'[/INST]')[-1].strip()
        return output_text
    
    def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
        embs_list, tokenid_list = [], []
        for image_list, question in zip(batch_images, batch_prompt):
            img_embes_list = []
            for image in image_list:
                image_emb, _ = self.model.encode_img(image)
                img_embes_list.append(image_emb)
            embs, input_ids = self.chat.get_context_emb(question, img_embes_list)
            tokenid_list.append(input_ids.squeeze(0))
            embs_list.append(embs.squeeze(0))
        # left padding
        embs_list = torch.nn.utils.rnn.pad_sequence(
            [embs.flip(dims=[0]) for embs in embs_list],
            batch_first=True,
            padding_value=0.).to(self.device).flip(dims=[1])
        attn_mask = torch.all(embs_list!=0, dim=-1)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [token.flip(dims=[0]) for token in tokenid_list],
            batch_first=True,
            padding_value=0.).to(self.device).flip(dims=[1])
        outputs = self.model.llama_model(
            inputs_embeds = embs_list,
            attention_mask = attn_mask,
            return_dict = True
        )
        logits = outputs['logits'][:,:-1].float()
        labels = input_ids[:,1:]
        batch_option_ids = []
        for option in batch_options:
            batch_option_ids.append(self.model.llama_tokenizer(option, add_special_tokens=False, return_tensors='pt').input_ids.squeeze(0))
        # import pdb; pdb.set_trace()
        results = []
        for idx in range(labels.shape[0]):
            option_len = len(batch_option_ids[idx])
            non_zero_indices = torch.nonzero(labels[idx], as_tuple=False).squeeze()
            start_index = non_zero_indices.max() - option_len + 1
            end_index = start_index + option_len
            if not np.all(labels[idx][start_index: end_index].detach().cpu().numpy() == batch_option_ids[idx].numpy()):
                import ipdb;ipdb.set_trace()
            prob = F.softmax(logits[idx][start_index: end_index], dim=-1)
            rows = torch.arange(0, option_len)
            score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean().item()
            results.append(score)
        return results
