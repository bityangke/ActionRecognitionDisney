spatial_net_prototxt=/home/mscvproject/mscvproject/code/temporal-segment-networks/models/ActNet200/tsn_bn_inception_rgb_deploy.prototxt
spatial_net_model=/home/mscvproject/mscvproject/code/temporal-segment-networks/models/ActNet200/top5BNFreeze/ActNet200_tsn_rgb_bn_inceptionseg5top3BNfreeze_iter_15000.caffemodel
temporal_net_prototxt=/home/mscvproject/mscvproject/code/temporal-segment-networks/models/ActNet200/tsn_bn_inception_flow_deploy.prototxt
temporal_net_model=/home/mscvproject/mscvproject/code/temporal-segment-networks/models/ActNet200/top5BNFreeze/ActNet200_tsn_flow_bn_inceptionseg5top3BNFreeze_iter_20000.caffemodel
num_gpu=7	
gpu_start_idx=9
scores_dump_folder='scores/untrimmed_val/seg5/'
score_layer_name='fc-action200'

# scores meta info
scores_dataset="activity_net"
scores_title="seg5"

python eval_TSN.py ${spatial_net_prototxt} ${spatial_net_model} ${temporal_net_prototxt} ${temporal_net_model} --num_gpu ${num_gpu} --gpu_start_idx ${gpu_start_idx} --scores_dump_folder ${scores_dump_folder} --score_layer_name ${score_layer_name} --scores_dataset ${scores_dataset} --scores_title ${scores_title}
