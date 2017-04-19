"""
Video Frame Sampler
"""

import random
import numpy as np


class BaseSampler:
    def sample_minibatch_indices(self, num_frame):
        assert False, 'not implemented'


class UniformSegmentSampler(BaseSampler):
    def __init__(self, num_segments):
        self._num_segments = num_segments

    def sample_minibatch_indices(self, num_frame):
        assert num_frame >= self._num_segments
        result = []
        segment_width = int(np.ceil(float(num_frame)/self._num_segments))
        for i in range(self._num_segments):
            start = i * segment_width
            end = min((i+1) * segment_width, num_frame)
            result.append(random.randrange(start, end))
        return np.array(result)


class DynamicSampler(BaseSampler):
    def sample_minibatch_indices(self, num_frame):
        pass
