
input_folder=/home/mscvproject/mscvproject/data/ActivityNetUntrim/val
output_folder=/home/mscvproject/mscvproject/data/ActivityNetShot/val
num_threads=48
threshold=125

python ShotDetection.py --input_folder ${input_folder} --output_folder ${output_folder} --extension mp4 --num_threads ${num_threads} --threshold ${threshold}
send_email_to_yu 'shot detection for untrimmed validation video is finished.'

