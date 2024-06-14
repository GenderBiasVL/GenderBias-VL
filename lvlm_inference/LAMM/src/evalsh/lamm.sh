
cd ../

model_cfg=config/ChEF/models/lamm.yaml
modelname=lamm

cuda_id=3
suffix=''
swap=''
func(){
    START_TIME=`date +%Y%m%d-%H:%M:%S`   

    LOG_FILE=../logs/evaluation_${modelname}_${dataname}${swap}_${suffix}_${START_TIME}.log
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        CUDA_VISIBLE_DEVICES=$cuda_id python eval.py \
                    --save_dir ../results \
                    --time ${suffix} \
                    --model_cfg=${model_cfg} \
                    --recipe_cfg=${recipe_cfg} \
                    --batch_size=32 \
                    2>&1 | tee -a $LOG_FILE > /dev/null #&
    sleep 10s; # 0.5
    # tail -f ${LOG_FILE}
}

all_modal_bias(){
    ####### confirm suffix, if_swap
    #### VLbias
    dataname=OccBaseAskGender
    recipe_cfg=config/Bias/${sub_dir}/${dataname}.yaml 
    # cuda_id=1
    func

    dataname=OccCfAskGender
    recipe_cfg=config/Bias/${sub_dir}/${dataname}.yaml
    # cuda_id=2
    func

    #### Vbias
    dataname=OccBaseAskPerson
    recipe_cfg=config/Bias/${sub_dir}/${dataname}.yaml
    # cuda_id=3
    func

    dataname=OccCfAskPerson
    recipe_cfg=config/Bias/${sub_dir}/${dataname}.yaml 
    # cuda_id=4
    func

    #### Lbias
    dataname=OccTextBaseAskGender
    recipe_cfg=config/Bias/${sub_dir}/${dataname}.yaml
    # cuda_id=5
    func

    dataname=OccTextCfAskGender
    recipe_cfg=config/Bias/${sub_dir}/${dataname}.yaml 
    # cuda_id=6
    func
}


suffix=PPL
swap=''
sub_dir=occbias # occbias_swap
all_modal_bias

suffix=PPL
swap='_swap'
sub_dir=occbias_swap # occbias_swap
all_modal_bias