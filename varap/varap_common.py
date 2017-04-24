""" Common rountines used by VARAP system
"""

import VideoRecog.data.activity_net as activity_net
import VideoRecog.data.hmdb51 as hmdb51

def register_dm():
    """
    register all data managers
    """
    dm_acnet = activity_net.DataActivityNet(annotation_file='/home/mscvproject/mscvproject/data/ActivityNetTrimflow/.scripts/activity_net.v1-3.min.json',
                                            frame_folders='/home/mscvproject/mscvproject/data/ActivityNetUntrimflow/view',
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