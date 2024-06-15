# GenderBias-*VL*: Benchmarking Gender Bias in Vision Language Models via Counterfactual Probing

<p align="center">
<br>
  <a href="https://genderbiasvl.github.io/" target="_blank"> Website </a >  •  <a href="https://github.com/GenderBiasVL/GenderBias-VL"> Codebase </a > • <a href="https://huggingface.co/datasets/xiaoyisong/GenderBias-VL"> Dataset </a > <br> 
<br>
</p >

## The Dataset
The GenderBias-*VL* dataset comprises 34,581 visual question counterfactual pairs covering 177 occupations, enabling LVLM bias evaluation in multimodal and unimodal contexts.
The dataset is available here: `https://huggingface.co/datasets/xiaoyisong/GenderBias-VL/tree/main`, which includes occupation images in `*.tar.gz` format and visual questions in a `.json` format within the `test_meta_file` directory.


## The CodeBase
The GenderBias-*VL* codebase is organized into three components: [construction pipeline](./construction), [LVLMs inference](./lvlm_inference), and [bias evaluation](./bias_metric). The LVLMs inference is based on the [LAMM codebase](https://github.com/OpenGVLab/LAMM), which is licensed under a CC-BY-NC 4.0 license and used in our work in compliance with its terms.

## Evaluation

### Installation

You can run the following script to configure the necessary environment.

``` sh
git clone https://github.com/GenderBiasVL/GenderBias-VL.git
cd GenderBias-VL
conda env create -f GenderBiasVL.yaml
conda activate GenderBiasVL
```

### LVLMs Preparation
The evaluated 15 LVLMs are listed as follows.

|         LVLMs         | Config   | Download Link |
|------------------|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [LLaVA1.5-7B](https://github.com/haotian-liu/LLaVA)          | [llava15_7b.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/llava15_7b.yaml)         | [Hugging Face](https://huggingface.co/liuhaotian/llava-v1.5-7b)|
| [LLaVA1.5-13B](https://github.com/haotian-liu/LLaVA)             | [llava15_13b.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/llava15_13b.yaml)       | [Hugging Face](https://huggingface.co/liuhaotian/llava-v1.5-13b) |
| [LLaVA1.6-13B](https://github.com/haotian-liu/LLaVA)                | [llava16_13b.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/llava16_13b.yaml)           | [Hugging Face](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b) |
| [MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4)              | [minigpt4v2.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/minigpt4v2.yaml)               | [MiniGPT-v2 (after stage-3)](https://drive.google.com/file/d/1HkoUUrjzFGn33cSiUkI-KcT-zysCynAz/view) |
| [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)      | [mplug2.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/mplug2.yaml) | [Hugging Face](https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b)  |
| [LLaMA-Adapter-2](https://github.com/ml-lab/LLaMA-Adapter-2) | [llamaadapterv2.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/llamaadapterv2.yaml)                 | [GitHub](https://github.com/ml-lab/LLaMA-Adapter-2) |
| [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)   | [instructblip_vicuna.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/instructblip_vicuna.yaml)                 | [InstructBLIP](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth), [Vicuna v1.1](https://huggingface.co/lmsys/vicuna-7b-v1.1) |
| [Otter](https://github.com/Luodian/Otter)             | [otter.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/otter.yaml)             | [Hugging Face](https://huggingface.co/luodian/OTTER-Image-LLaMA7B-LA-InContext)  |
| [LAMM](https://github.com/OpenGVLab/LAMM)              | [lamm.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/lamm.yaml)            | [Hugging Face](https://huggingface.co/openlamm/lamm_13b_lora32_186k)  |
| [Kosmos-2](https://github.com/microsoft/unilm/tree/master/kosmos-2)             | [kosmos2.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/kosmos2.yaml)          | [GitHub](https://github.com/microsoft/unilm/tree/master/kosmos-2#checkpoints) |
| [Qwen-VL](https://github.com/QwenLM/Qwen-VL)         | [qwen_vl.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/qwen_vl.yaml)   | [Hugging Face](https://huggingface.co/Qwen/Qwen-VL-Chat) |
| [InternLM-XComposer2-VL](https://github.com/InternLM/InternLM-XComposer)            | [internlm_xcomposer.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/internlm_xcomposer.yaml)         | [Hugging Face](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b)    |
| [Shikra](https://github.com/shikras/shikra)           | [shikra.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/shikra.yaml)    | [Hugging Face](https://huggingface.co/shikras/shikra-7b-delta-v1) |
| [LLaVA-RLHF](https://github.com/llava-rlhf/LLaVA-RLHF)          | [llavarlhf.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/llavarlhf.yaml)    | [Hugging Face](https://huggingface.co/zhiqings/LLaVA-RLHF-13b-v1.5-336)    |
| [RLHF-V](https://github.com/RLHF-V/RLHF-V)          | [rlhfv.yaml](./lvlm_inference/LAMM/src/config/ChEF/models/rlhfv.yaml)   | [Hugging Face](https://huggingface.co/openbmb/RLHF-V)  |


### Dataset Preparation

After downloading the [GenderBias-*VL* database](https://huggingface.co/datasets/xiaoyisong/GenderBias-VL/tree/main), you need to setup the config of evaluation datasets as follows: ``base_data_path: /path/to/image_root``, ``meta_file: /path/to/visual_question``.

|         Bias Contexts         | Base VQ   | Counterfactual VQ |
|:--:|:--:|:--:|
| VL-Bias         | [OccBaseAskGender.yaml](./lvlm_inference/LAMM/src/config/Bias/occbias/OccBaseAskGender.yaml)          | [OccCfAskGender.yaml](./lvlm_inference/LAMM/src/config/Bias/occbias/OccCfAskGender.yaml) |
| V-Bias         | [OccBaseAskPerson.yaml](./lvlm_inference/LAMM/src/config/Bias/occbias/OccBaseAskPerson.yaml)     | [OccCfAskPerson.yaml](./lvlm_inference/LAMM/src/config/Bias/occbias/OccCfAskPerson.yaml)     |
| L-Bias         | [OccTextBaseAskGender.yaml](./lvlm_inference/LAMM/src/config/Bias/occbias/OccTextBaseAskGender.yaml)      |  [OccTextCfAskGender.yaml](./lvlm_inference/LAMM/src/config/Bias/occbias/OccTextCfAskGender.yaml)    |

### LVLMs Inference
You can run the following script (an example for LLaVA 1.5-7B) to obtain the inference results of LVLMs on the GenderBias-*VL* database (visual questions). 

``` sh
cd ./lvlm_inference/LAMM/src/evalsh
bash llava15.sh
```

After inference, the results will be saved in `./lvlm_inference/LAMM/results/{modelname}` (in this case, LLaVA1.5-7B). This directory will include responses to each visual question, stored in JSON files that start with `OccBias_`.

### Bias Evaluation
Based on the response json files, you can run the following script to access LVLMs' gender bias.
    ``` sh
    cd ./bias_metric
    ```
1. Calculate bias results for each LVLM and record them in `./bias_results/{modelname}/{bias_context}/`. For example, see the [VL-Bias results of LLaVA1.5-7B](./bias_metric/bias_results/LLaVA1.5/VLbias/).
    ```sh
    python run.py ### this is a script to call cal_bias.py for all LVLMs and all contexts.
    ```

2. Merge the results of option-swapping test.
Bias ($B_{pair}$) base on probability difference is recorded in [merge_bias_probability](./bias_metric/bias_merge_swap_test/merge_bias_probability/), and the occupation-level bias $B_{micro}$ (probability difference) is recored in [merge_micro_occ](./bias_metric/bias_merge_swap_test/merge_micro_occ/). Bias ($B_{pair}^{o}$) based on outcome difference is recorded in [merge_bias_outcome](./bias_metric/bias_merge_swap_test/merge_bias_outcome/).
    ```sh
    python merge_swap_result.py
    ```

3. Overall results: bias $B_{ovl}$ recored in [overall](./bias_metric/bias_merge_swap_test/overall/bias_overall.csv). You can modify the Line 10 to access $B_{ovl}^{o}$ (Section 4.2).

    ```sh
    python overall_result.py 
    ```


## Counterfactual Visual Question Pairs Construction Pipeline

First `cd ./construction`.

### Occupation Image Generation (<u>image</u>)
1. The occupation list we utilize is stored in [occ_us.csv](./resources/occ_us.csv), and prompts generated by ChatGPT are stored in [prompts.txt](./resources/prompts.txt).

2. Generate counterfactual image pairs by running ``python cf_generator.py``. 
  
    You should set the text-to-diffusion models ([Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix)) and the image output directory from Lines 21 to 30 in the python file.

3. Conduct image filtering by running following scripts.
    ```sh
      cd ./runsh
      bash filter.sh
    ```
    You should first set up the [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything).

    By default, it will yield a valid test case file in `./exp_image_filter/image_filter/logs/test_case.txt`.

4. Bleach gender representation in images.
    ```sh
      cd ./runsh
      bash bleach.sh
    ```
    Similar to how you set up `filter.sh`.

### Stereotyped Occupation Pairs Identification (<u>option</u>)

You can run the following script to obtain stereotyped occupation pairs.

```sh
cd ./vq_generation
python cal_img_sim.py  # set image path in Line 15
python cal_text_sim.py 
python merge_img_text_sim.py
```
By default, similarity scores are stored in `./vq_generation/similarity`: 
- [visual similarity](./construction/vq_generation/similarity/occ_img_sim.csv) 
- [textual similarity](./construction/vq_generation/similarity/occ_text_sim.csv)
- [$sim$ scores](./construction/vq_generation/similarity/occ_merge_sim.csv). 

The stereotyped occupation pairs are stored [here](./construction/vq_generation/similarity/occ_merge_filter_sim.csv).

### Visual Question Counterfactuals Creation.

Call `python test_file_generate_json.py` to create visual questions. 

Set the `image_type`, `ask_item`, and `swap` parameter in Lines 19-21 to control the visual question templates, Additionally, set the `exp_dir` on Line 23 to control the output directory.

1. For VL-Bias:
    ```sh
    ## set image_type=base, ask_item=gender, swap=False, exp_dir="./test_meta_file/VLbias"
    python test_file_generate_json.py 

    ## set image_type=cf, ask_item=gender, swap=False, exp_dir="./test_meta_file/VLbias"
    python test_file_generate_json.py 
    ```

2. For V-Bias:
    ``` sh
    ## set image_type=base, ask_item=person, swap=False, exp_dir="./test_meta_file/Vbias"
    python test_file_generate_json.py 

    ## set image_type=cf, ask_item=person, swap=False, exp_dir="./test_meta_file/Vbias"
    python test_file_generate_json.py 
    ```

3. For L-Bias:
    ``` sh
    ## set image_type=base, ask_item=gender, swap=False, exp_dir="./test_meta_file/Lbias"
    python test_file_generate_json_textbias.py 

    ## set image_type=cf, ask_item=gender, swap=False, exp_dir="./test_meta_file/Lbias"
    python test_file_generate_json_textbias.py 
    ```
4. For the option-swapping test file, set `swap=True` and run above scripts.

## License

Please see the [CC BY NC 4.0 licence](./LICENSE)

## Acknowledgement

The [LVLMs inference](./lvlm_inference/) is based on [LAMM](https://github.com/OpenGVLab/LAMM). [Chef in LAMM](https://github.com/OpenGVLab/LAMM/blob/main/docs/ch3ef.md) provides an amazing framework to evaluate LVLMs. We appreciate their contribution.

## Citation

```
@misc{hall2023visogender,
      title={GenderBias-\emph{VL}: Benchmarking Gender Bias in Vision Language Models via Counterfactual Probing}, 
      author={Yisong Xiao, Aishan Liu, QianJia Cheng, Zhenfei Yin, Siyuan Liang, Jiapeng Li, Jing Shao, Xianglong Liu, Dacheng Tao},
      year={2024},
}
```

