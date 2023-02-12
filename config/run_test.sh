#! /bin/bash
#exec 1000>&-;表示关闭文件描述符1000的写
#exec 1000<&-;表示关闭文件描述符1000的读
#trap是捕获中断命令


export CUDA_VISIBLE_DEVICES="0"
log_info='100'

dataset='gowalla'
logs='log_gowalla'
# dataset='yelp'
loss_type=2
lambda=5
epoch=20
batch=4096
lr=0.01
n=200
w=0.01
beta=1

python main_more2.py -m 0 --lambda_w ${lambda} -lr ${lr} --beta ${beta} --data ${dataset} \
                --sampler 0 -s $n -e ${epoch} -b ${batch} --fix_seed --weighted --log_path ${logs} \
                --weight_decay ${w} --loss_type ${loss_type} --log_info ${log_info};



