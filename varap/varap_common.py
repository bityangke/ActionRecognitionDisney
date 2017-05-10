""" Common rountines used by VARAP system
"""

import sys
sys.path.append('../VideoRecog')
import VideoRecog.data.activity_net as activity_net
import VideoRecog.data.hmdb51 as hmdb51
import VideoRecog.config as config

def register_dm():
    """
    register all data managers
    """
    dm_acnet = activity_net.DataActivityNet(annotation_file=config.activitynet_annotation_file,
                                            frame_folders=config.activitynet_frame_folder,
                                            trimmed=False)
    dm_reg = {}
    dm_reg['activity_net'] = dm_acnet
    for key, val in dm_reg.iteritems():
        val.init()
    return dm_reg


# intiailize data managers
dm_reg = register_dm()

# Configurations
score_folder = '/home/mscvproject/mscvproject/code/ActionRecognitionDisney/TSN_eval/scores'