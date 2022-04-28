
import json
import os
import os.path as pt
from typing import Any, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_curve)

import sys

class NumpyEncoder(json.JSONEncoder):
    """ Encoder to correctly use json on numpy arrays """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def single_save(dir: str, name: str, dic: Any, subdir='.'):
    """
    Writes a given dictionary to a json file in the log directory.
    Returns without impact if the size of the dictionary exceeds 10MB.
    :param name: name of the json file
    :param dic: serializable dictionary
    :param subdir: if given, creates a subdirectory in the log directory. The data is written to a file
        in this subdirectory instead.
    """
    outfile = pt.join(dir, subdir, '{}.json'.format(name))
    if not pt.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    if isinstance(dic, dict):
        sz = np.sum([sys.getsizeof(v) for k, v in dic.items()])
        if sz > 10000000:
            print(
                'WARNING: Could not save {}, because size of dict is {}, which exceeded 10MB!'
                .format(pt.join(dir, subdir, '{}.json'.format(name)), sz),
                print=True
            )
            return
        with open(outfile, 'w') as writer:
            json.dump(dic, writer, cls=NumpyEncoder)
    else:
        torch.save(dic, outfile.replace('.json', '.pth'))    
    
    
def _reduce_curve_number_of_points(x, y, npoints) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduces the number of points in the curve by interpolating linearly.
    Suitable for ROC and PR curves.
    """
    func = interp1d(x, y, kind='linear')
    xmin, xmax = np.min(x), np.max(x)
    xs = np.linspace(xmin, xmax, npoints, endpoint=True)
    return xs, func(xs)
  
    
@torch.no_grad()
def compute_gtmap_roc(
    anomaly_scores,
    original_gtmaps,
    net, 
    limit_npoints: int = 3000,
):
    """the scores are upsampled to the images' original size and then the ROC is computed."""
    
    # GTMAPS pixel-wise anomaly detection = explanation performance
    print('Computing ROC score')
    
    # Reduces the anomaly score to be a score per pixel (explanation)
    anomaly_scores = anomaly_scores.mean(1).unsqueeze(1)
    anomaly_scores = net.receptive_upsample(anomaly_scores, std=net.gauss_std)
        
    # Further upsampling for original dataset size
    anomaly_scores = torch.nn.functional.interpolate(anomaly_scores, (original_gtmaps.shape[-2:]))
    flat_gtmaps, flat_ascores = original_gtmaps.reshape(-1).int().tolist(), anomaly_scores.reshape(-1).tolist()
    
    fpr, tpr, ths = roc_curve(
        y_true=flat_gtmaps, 
        y_score=flat_ascores,
        drop_intermediate=True,
    )
    
    # reduce the number of points of the curve
    npoints = ths.shape[0]
    
    if npoints > limit_npoints:
        
        _, fpr = _reduce_curve_number_of_points(
            x=ths, 
            y=fpr, 
            npoints=limit_npoints,
        )
        ths, tpr = _reduce_curve_number_of_points(
            x=ths, 
            y=tpr, 
            npoints=limit_npoints,
        )
    
    auc_score = auc(fpr, tpr)
    
    print(f'##### GTMAP ROC TEST SCORE {auc_score} #####')
    gtmap_roc_res = {'tpr': tpr, 'fpr': fpr, 'ths': ths, 'auc': auc_score}
    
    return gtmap_roc_res


@torch.no_grad()
def compute_gtmap_pr(
    anomaly_scores,
    original_gtmaps,
    net, 
    limit_npoints: int = 3000,
):
    """
    The scores are upsampled to the images' original size and then the PR is computed.
    The scores are normalized between 0 and 1, and interpreted as anomaly "probability".
    """
    
    # GTMAPS pixel-wise anomaly detection = explanation performance
    print('Computing PR score')
    
    # Reduces the anomaly score to be a score per pixel (explanation)
    anomaly_scores = anomaly_scores.mean(1).unsqueeze(1)
    anomaly_scores = net.receptive_upsample(anomaly_scores, std=net.gauss_std)
        
    # Further upsampling for original dataset size
    anomaly_scores = torch.nn.functional.interpolate(anomaly_scores, (original_gtmaps.shape[-2:]))
    flat_gtmaps, flat_ascores = original_gtmaps.reshape(-1).int().tolist(), anomaly_scores.reshape(-1).tolist()
    
    # ths = thresholds
    precision, recall, ths = precision_recall_curve(
        y_true=flat_gtmaps, 
        probas_pred=flat_ascores,
    )
    
    # a (0, 1) point is added to make the graph look better
    # i discard this because it's not useful and there is no 
    # corresponding threshold 
    precision, recall = precision[:-1], recall[:-1]
    
    # recall must be in descending order 
    # recall = recall[::-1]
    
    # reduce the number of points of the curve
    npoints = ths.shape[0]
    
    if npoints > limit_npoints:
        
        _, precision = _reduce_curve_number_of_points(
            x=ths, 
            y=precision, 
            npoints=limit_npoints,
        )
        ths, recall = _reduce_curve_number_of_points(
            x=ths, 
            y=recall, 
            npoints=limit_npoints,
        )
    
    ap_score = average_precision_score(y_true=flat_gtmaps, y_score=flat_ascores)
    
    print(f'##### GTMAP AP TEST SCORE {ap_score} #####')
    gtmap_pr_res = dict(recall=recall, precision=precision, ths=ths, ap=ap_score)
    return gtmap_pr_res

