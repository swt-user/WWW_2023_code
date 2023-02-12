#! /bin/bash
#exec 1000>&-;表示关闭文件描述符1000的写
#exec 1000<&-;表示关闭文件描述符1000的读
#trap是捕获中断命令
trap "exec 1000>&-;exec 1000<&-;exit 0" 2 #接受信号2（ctrl +C）做的操作，表示在脚本运行过程中，如果接收到Ctrl+C中断命令，则关闭文件描述符1000的读写，并正常退出。
FIFO_FILE=$$.fifo  # $表示当前执行文件的PID
mkfifo $FIFO_FILE  #创建管道文件
exec 1000<>$FIFO_FILE #将管道文件和文件操作符绑定

rm -rf $FIFO_FILE	#删除管道文件

# 并行数目
PROCESS_NUM=3
for ((idx=0; idx<$PROCESS_NUM; idx++));
do
    echo>&1000 #对文件操作符进行写入操作，通过for循环写入空行，空行数目为定义的后台线程数量
done

export CUDA_VISIBLE_DEVICES="7"
log_info='73'

dataset='gowalla'
logs='log_gowalla'
# dataset='yelp'
loss_type=12
n=200
epoch=200
batch=4096
beta=1

for lambda in 0.01 0.1 1
do  
    for rescale_hyp in 0.02 0.05 0.1 0.2 0.3
    do
        for lr in 0.01 0.001
        do
            for w in 0.01 0.001 0.0001
            do
                sleep 10
                read -u1000 #从文件描述符读入空行。read -u 后面跟fd，从文件描述符中读入，该文件描述符可以是exec新开启的。
                {
                    # 要执行的命令
                    nohup python main_more2.py -m 0 --lambda_w ${lambda} -lr ${lr} --beta ${beta} --data ${dataset} --rescale_hyp ${rescale_hyp}\
                    --sampler 0 -s $n -e ${epoch} -b ${batch} --fix_seed --weighted --log_path ${logs} \
                    --weight_decay ${w} --loss_type ${loss_type} --log_info ${log_info} >/dev/null 2>&1;

                    echo >&1000 #上一条命令执行完毕后，在文件描述符中写入空行
                } &
            done
        done
    done
done

wait #wait指令等待所有后台进程执行结束
exec 1000>&- #表示关闭文件描述符1000的写