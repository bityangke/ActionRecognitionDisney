spatial_net_prototxt=/home/mscvproject/mscvproject/code/ActionRecognitionDisney/TSN_eval/models/resnet200_anet_2016_deploy.prototxt
spatial_net_model=/home/mscvproject/mscvproject/code/ActionRecognitionDisney/TSN_eval/models/resnet200_anet_2016.caffemodel
temporal_net_prototxt=/home/mscvproject/mscvproject/code/ActionRecognitionDisney/TSN_eval/models/bn_inception_anet_2016_temporal_deploy.prototxt
temporal_net_model=/home/mscvproject/mscvproject/code/ActionRecognitionDisney/TSN_eval/models/bn_inception_anet_2016_temporal.caffemodel.v5
num_gpu=7
gpu_start_idx=2
scores_dump_folder='scores/untrimmed_val/ref/'
score_layer_name='fc-action'

# scores meta info
scores_dataset="activity_net"
scores_title="reference"

python eval_TSN.py ${spatial_net_prototxt} ${spatial_net_model} ${temporal_net_prototxt} ${temporal_net_model} --num_gpu ${num_gpu} --gpu_start_idx ${gpu_start_idx} --scores_dump_folder ${scores_dump_folder} --score_layer_name ${score_layer_name} --scores_dataset ${scores_dataset} --scores_title ${scores_title}
