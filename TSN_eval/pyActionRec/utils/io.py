import numpy as np


def flow_stack_oversample(flow_stack, crop_dims):
    """
    This function performs oversampling on flow stacks.
    1. corner cropping + center cropping (x5)
    2. horizontal flipping (x2)

    :param flow_stack: (N, H, W)
    :param crop_dims: (crop_h, crop_w)
    :return: cropped_result (10, N, H, W)
    """
    im_shape = np.array(flow_stack.shape[1:])
    stack_depth = flow_stack.shape[0]
    crop_dims = np.array(crop_dims)

    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])

    h_center_offset = (im_shape[0] - crop_dims[0])/2
    w_center_offset = (im_shape[1] - crop_dims[1])/2

    crop_ix = np.empty((5, 4), dtype=int)

    cnt = 0
    for i in h_indices:
        for j in w_indices:
            crop_ix[cnt, :] = (i, j, i+crop_dims[0], j+crop_dims[1])
            cnt += 1
    crop_ix[4, :] = [h_center_offset, w_center_offset,
                     h_center_offset+crop_dims[0], w_center_offset+crop_dims[1]]

    crop_ix = np.tile(crop_ix, (2,1))

    crops = np.empty((10, flow_stack.shape[0], crop_dims[0], crop_dims[1]),
                     dtype=np.float32)

    for ix in xrange(10):
        cp = crop_ix[ix]
        crops[ix] = flow_stack[:, cp[0]:cp[2], cp[1]:cp[3]]

    crops[5:] = crops[5:, :, :, ::-1]

    # TODO: we should contact the author. This doesn't make sense.
    crops[5:, range(0, stack_depth, 2), ...] = 255 - crops[5:, range(0, stack_depth, 2), ...]

    return crops
