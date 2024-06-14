
cd ../

model_cfg=config/ChEF/models/gemini.yaml
modelname=Gemini

cuda_id=0
suffix=''
swap=''
func(){
    START_TIME=`date +%Y%m%d-%H:%M:%S`   
    export http_proxy="http://127.0.0.1:7890" 
    export https_proxy="http://127.0.0.1:7890"
    LOG_FILE=../logs_close/evaluation_${modelname}_${dataname}${swap}_${suffix}_${START_TIME}.log
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        CUDA_VISIBLE_DEVICES=$cuda_id python eval.py \
                    --save_dir ../results_close \
                    --time ${suffix} \
                    --model_cfg=${model_cfg} \
                    --recipe_cfg=${recipe_cfg} \
                    --batch_size=1 \
                    # --sample_len=1 \
                    # \
                    # 2>&1 | tee -a $LOG_FILE > /dev/null #&
    sleep 10s; # 0.5
    # tail -f ${LOG_FILE}
}



all_modal_bias_direct(){
    ####### confirm suffix, if_swap
    #### VLbias
    dataname=OccBaseAskGender_${suffix}
    recipe_cfg=config/Bias/close/${dataname}.yaml 
    # cuda_id=1
    func

    dataname=OccCfAskGender_${suffix}
    recipe_cfg=config/Bias/close/${dataname}.yaml
    # cuda_id=2
    func

    ### Vbias
    dataname=OccBaseAskPerson_${suffix}
    recipe_cfg=config/Bias/close/${dataname}.yaml
    # cuda_id=3
    func

    dataname=OccCfAskPerson_${suffix}
    recipe_cfg=config/Bias/close/${dataname}.yaml 
    # cuda_id=4
    func

    #### Lbias
    dataname=OccTextBaseAskGender_${suffix}
    recipe_cfg=config/Bias/close/${dataname}.yaml
    # cuda_id=5
    func

    dataname=OccTextCfAskGender_${suffix}
    recipe_cfg=config/Bias/close/${dataname}.yaml 
    # cuda_id=6
    func
}


suffix=direct
swap=''
sub_dir=occbias 
all_modal_bias_direct
