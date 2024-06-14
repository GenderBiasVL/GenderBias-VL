import torch
import numpy as np
import torch.nn.functional as F
from transformers import TextStreamer

from .mplug_owl.builder import load_pretrained_model
from .mplug_owl.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from .mplug_owl.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from .mplug_owl.conversation import conv_templates, SeparatorStyle
from .test_base import TestBase


class TestMplugOwl2(TestBase):
    def __init__(self, device, model_path, **kwargs):
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, self.model_name, load_8bit=False, load_4bit=False, device=device)
        self.device = device
        self.move_to_device(device)
        self.model.eval()
        self.tokenizer.padding_side = 'left'
        if not hasattr(self.tokenizer, 'pad_token_id'):
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_i
    
    def build_input_image(self, image_list):
        images = self.get_image_list(image_list)
        images = process_images(images, self.image_processor)
        images = images.to(self.device, dtype=torch.float16)
        return images
    
    def build_conversation(self, idx, image_list, prompt, CoT_answer_list=None, batch_answers=None, **kwargs):
        conv = conv_templates["mplug_owl2"].copy()
        lenimg = 1 if isinstance(image_list,str) else len(image_list)
        inp = DEFAULT_IMAGE_TOKEN * lenimg + prompt #
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if CoT_answer_list is not None:
            prompt += CoT_answer_list[idx]
        if batch_answers is not None:
            prompt += ' ' + batch_answers[idx]
        return prompt
    
    def do_generate(self, input_image_list: list, input_prompt: str, max_new_tokens):
        input_ids = tokenizer_image_token(input_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        conv = conv_templates["mplug_owl2"].copy()
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=input_image_list,
                do_sample=False,
                temperature=0.7,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        return outputs
    
    @torch.no_grad()
    def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
        batch_images = torch.cat(batch_images, dim=0)
        # print('batch_images: ', batch_images.shape)
        batch_input_ids, attention_mask = batch_tokenizer_image_token(batch_prompt, self.tokenizer)
        batch_option_ids = batch_tokenizer_image_token(batch_options, self.tokenizer, add_special_tokens=False).to(self.device)[:, 1:]
        # print('batch_input_ids: ', batch_input_ids.shape)
        # print('attention_mask: ', attention_mask.shape)
        input_ids, modality_indicators, attention_mask, past_key_values, inputs_embeds, labels = \
            self.model.prepare_inputs_labels_for_multimodal(
                input_ids=batch_input_ids.to(self.device), 
                attention_mask=attention_mask.to(self.device), 
                past_key_values=None,
                labels=batch_input_ids.to(self.device),
                images=batch_images.to(dtype=self.dtype).to(self.device)
            )
        outputs = self.model.base_model(
            input_ids=input_ids,
            modality_indicators=modality_indicators,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
        )
        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        # print('logits: ', logits)
        labels = labels[:, 1:]
        results = []
        for idx in range(labels.shape[0]):
            option_len = torch.sum(batch_option_ids[idx]!=IGNORE_INDEX).item() 
            non_zero_indices = torch.where(labels[idx] != -100)[0]
            start_index = non_zero_indices.max() - option_len + 1
            end_index = start_index + option_len
            prob = F.softmax(logits[idx][start_index: end_index], dim=-1)
            rows = torch.arange(0, option_len)
            score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean().item()
            results.append(score)
            # import pdb; pdb.set_trace()
        return results
    
def batch_tokenizer_image_token(prompts: list, tokenizer, add_special_tokens=True):
    input_ids_list = []
    for prompt in prompts:
        input_ids = tokenizer_image_token(prompt, tokenizer, 
            IMAGE_TOKEN_INDEX, return_tensors='pt').tolist()
        input_ids_list.append(input_ids)
    
    input_tokens_max_length = max([len(x) for x in input_ids_list])
    pad_token_id = tokenizer.pad_token_id if add_special_tokens else IGNORE_INDEX
    input_ids_list = [([pad_token_id] * (input_tokens_max_length - len(_)) + _) for _ in input_ids_list] # pad in the left
    input_ids_list = torch.LongTensor(input_ids_list)
    if add_special_tokens:
        attention_mask = 1 - input_ids_list.eq(pad_token_id).long()
        return input_ids_list, attention_mask
    else:
        return input_ids_list