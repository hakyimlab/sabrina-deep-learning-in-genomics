import numpy as np

def inverse_transform(y_wt, track_scale, track_transform, clip_soft):

    y_wt_curr = np.array(np.copy(y_wt), dtype=np.float32)

    # undo scale
    y_wt_curr /= track_scale

    # undo soft_clip
    y_wt_curr_unclipped = (y_wt_curr - clip_soft) ** 2 + clip_soft

    unclip_mask_wt = y_wt_curr > clip_soft

    y_wt_curr[unclip_mask_wt] = y_wt_curr_unclipped[unclip_mask_wt]

    # undo sqrt
    y_wt_curr = y_wt_curr ** (1.0 / track_transform)

    return y_wt_curr

def expr_attr(y_wt, track_index, gene_slice):
    y_wt_curr = inverse_transform(y_wt, 0.01, 3./4., 384.)
    y_wt_track = np.mean(y_wt_curr[..., track_index], axis=(0, 1, 3))
    sum_wt = np.sum(y_wt_track[gene_slice])
    return sum_wt