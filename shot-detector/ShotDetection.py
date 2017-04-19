""" Parallel Video Shot Detection
"""
import os
import argparse
import multiprocessing
from multiprocessing import Lock
from multiprocessing import Value


lock = None
count = None
total = None


def process_video(args):
    global count, total, lock
    video_name, output_folder, idx = args
    os.system('./Shotdetect/build/shotdetect-cmd -i {0} -o {1} -s 100 -a {2}'.format(video_name, output_folder, idx))
    lock.acquire()
    count.value += 1
    print('{0}/{1} videos completed.'.format(count.value, total))
    lock.release()


def detect(num_threads, video_list, output_folder):
    """
    detect shots in videos in parallel
    Args:
        num_threads: number of worker threads
        video_list: list of video path
        output_folder: path to dump the result
    Returns:
    """
    global lock, count, total
    os.system('rm -rf {0}'.format(output_folder))
    os.system('mkdir {0}'.format(output_folder))
    lock = Lock()
    count = Value('i', 0)
    total = len(video_list)
    p = multiprocessing.Pool(num_threads)
    p.map(process_video, zip(video_list, [output_folder] * total, range(total)))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str,
                        help='input folder containing the videos')
    parser.add_argument('--extension', default='mp4',
                        help='extension of the video file')
    parser.add_argument('--output_folder', type=str,
                        help='output_folder to dump video shot info (.xml)')
    parser.add_argument('--num_threads', type=int,
                        help='number of worker threads')
    args = parser.parse_args()

    # collect video list
    video_list = []
    for root, dir, files in os.walk(args.input_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.endswith(args.extension):
                video_list.append(file_path)
    print('{0} videos to be processed'.format(len(video_list)))
    detect(num_threads=args.num_threads, video_list=video_list, output_folder=args.output_folder)
