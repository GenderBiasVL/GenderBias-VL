cd ..

basedir="../exp_flip"
dir="." # Path to the root directory of the Grounded-Segment-Anything-main
CUDA_VISIBLE_DEVICES=1 python image_bleach.py \
  --config ${dir}/Grounded-Segment-Anything-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ${dir}/Grounded-Segment-Anything-main/model_weights/groundingdino_swint_ogc.pth \
  --sam_checkpoint ${dir}/Grounded-Segment-Anything-main/model_weights/sam_vit_h_4b8939.pth \
  --box_threshold 0.3 --text_threshold 0.25 --batch_size 32 \
  --base_root ${basedir}/xl_generate_base \
  --occ_path "../resources/occ_us.csv" \
  --detect_prompt "a person." --sub_exp xl_bleach_person

