cd ..



basedir="../exp_flip"
dir="." # Path to the root directory of the Grounded-Segment-Anything-main

CUDA_VISIBLE_DEVICES=1 python image_filter.py \
  --config ${dir}/Grounded-Segment-Anything-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ${dir}/Grounded-Segment-Anything-main/model_weights/groundingdino_swint_ogc.pth \
  --box_threshold 0.3 --text_threshold 0.25 --batch_size 32 \
  --base_root ${basedir}/xl_generate_base \
  --cf_root ${basedir}/xl_generate_cf_via_instructpix2pix \
  --occ_path "../resources/occ_us.csv" \
  --detect_prompt "a person." --sub_exp image_filter


### image_filter/logs/test_case.txt are utilized to construct test cases.
