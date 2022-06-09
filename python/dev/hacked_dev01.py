"""
these are functions that i copied from some other code and modified some minor things so 
i could adapt it to my usage
"""


import abc
import functools
import random
from tkinter import W
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.states import RunningStage
from sklearn.metrics import (roc_auc_score, roc_curve)
from torch import Tensor

from data_dev01 import ANOMALY_TARGET, NOMINAL_TARGET


def roc_curve(
    y_true=None, y_probas=None, labels=None, classes_to_plot=None, title=None
):
    """
    src: wandb.plot.roc_curve.roc_curve()
    i marked "modif" where i added something
    """
    from wandb.plots.utils import test_missing, test_types

    import wandb
    from wandb import util

    chart_limit = wandb.Table.MAX_ROWS
    
    np = util.get_module(
        "numpy",
        required="roc requires the numpy library, install with `pip install numpy`",
    )
    util.get_module(
        "sklearn",
        required="roc requires the scikit library, install with `pip install scikit-learn`",
    )
    from sklearn.metrics import roc_curve

    if test_missing(y_true=y_true, y_probas=y_probas) and test_types(
        y_true=y_true, y_probas=y_probas
    ):
        y_true = np.array(y_true)
        y_probas = np.array(y_probas)
        classes = np.unique(y_true)
        probas = y_probas
        
        # modif
        (nsamples, nscores) = y_probas.shape
        is_single_score = nscores == 1
        if is_single_score:
            assert tuple(classes) == (0, 1), "roc requires binary classification if there is a single score"
            assert classes_to_plot is None, f"classes_to_plot must be None if there is a single score"
            assert labels is None or len(labels) == 1, f"labels must be None or have length 1 if there is a single score"
        # modif
            
        if classes_to_plot is None:
            classes_to_plot = classes 

        fpr_dict = dict()
        tpr_dict = dict()

        indices_to_plot = np.in1d(classes, classes_to_plot)

        data = []
        count = 0
        
        # modif
        # very hacky but who cares
        if is_single_score:  # use only the positive score
            classes_to_plot = [classes_to_plot[1]]
            indices_to_plot = [indices_to_plot[1]] 
            classes = [classes[1]]
            probas = probas.reshape(-1, 1)
        # modif

        for i, to_plot in enumerate(indices_to_plot):
            
            # modif
            if is_single_score and i > 0:
                break
            # modif
            
            fpr_dict[i], tpr_dict[i], _ = roc_curve(
                y_true, probas[:, i], pos_label=classes[i]
            )
                
            if to_plot:
                for j in range(len(fpr_dict[i])):
                    if labels is not None and (
                        isinstance(classes[i], int)
                        or isinstance(classes[0], np.integer)
                    ):
                        class_dict = labels[classes[i]]
                    else:
                        class_dict = classes[i]
                    fpr = [
                        class_dict,
                        round(fpr_dict[i][j], 3),
                        round(tpr_dict[i][j], 3),
                    ]
                    data.append(fpr)
                    count += 1
                    if count >= chart_limit:
                        wandb.termwarn(
                            "wandb uses only the first %d datapoints to create the plots."
                            % wandb.Table.MAX_ROWS
                        )
                        break
        table = wandb.Table(columns=["class", "fpr", "tpr"], data=data)
        title = title or "ROC"
        return wandb.plot_table(
            "wandb/area-under-curve/v0",
            table,
            {"x": "fpr", "y": "tpr", "class": "class"},
            {
                "title": title,
                "x-axis-title": "False positive rate",
                "y-axis-title": "True positive rate",
            },
        )
        
        
        
        
import wandb
from wandb import util
from wandb.plots.utils import test_missing, test_types


chart_limit = wandb.Table.MAX_ROWS


def pr_curve(y_true=None, y_probas=None, labels=None, classes_to_plot=None, title=None):
    """
    src: wandb.plot.pr_curve.pr_curve()
    i marked "modif" where i added something
    """
    np = util.get_module(
        "numpy",
        required="roc requires the numpy library, install with `pip install numpy`",
    )
    preprocessing = util.get_module(
        "sklearn.preprocessing",
        "roc requires the scikit preprocessing submodule, install with `pip install scikit-learn`",
    )

    metrics = util.get_module(
        "sklearn.metrics",
        "roc requires the scikit metrics submodule, install with `pip install scikit-learn`",
    )

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    if test_missing(y_true=y_true, y_probas=y_probas) and test_types(
        y_true=y_true, y_probas=y_probas
    ):
        classes = np.unique(y_true)
        probas = y_probas
        
        # modif
        (nsamples, nscores) = y_probas.shape
        is_single_score = nscores == 1
        if is_single_score:
            assert tuple(classes) == (0, 1), "roc requires binary classification if there is a single score"
            assert classes_to_plot is None, f"classes_to_plot must be None if there is a single score"
            assert labels is None or len(labels) == 1, f"labels must be None or have length 1 if there is a single score"
        # modif

        if classes_to_plot is None:
            classes_to_plot = classes
            
        binarized_y_true = preprocessing.label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            binarized_y_true = np.hstack((1 - binarized_y_true, binarized_y_true))

        pr_curves = {}
        indices_to_plot = np.in1d(classes, classes_to_plot)
        
        # modif
        # very hacky but who cares
        if is_single_score:  # use only the positive score
            classes_to_plot = [classes_to_plot[1]]
            indices_to_plot = [indices_to_plot[1]] 
            classes = [classes[1]]
            probas = probas.reshape(-1, 1)
        # modif
        
        for i, to_plot in enumerate(indices_to_plot):
            
            # modif
            if is_single_score and i > 0:
                break
            # modif
            
            if to_plot:
                precision, recall, _ = metrics.precision_recall_curve(
                    y_true, probas[:, i], pos_label=classes[i]
                )

                samples = 20
                sample_precision = []
                sample_recall = []
                for k in range(samples):
                    sample_precision.append(
                        precision[int(len(precision) * k / samples)]
                    )
                    sample_recall.append(recall[int(len(recall) * k / samples)])

                pr_curves[classes[i]] = (sample_precision, sample_recall)

        data = []
        count = 0
        for class_name in pr_curves.keys():
            precision, recall = pr_curves[class_name]
            for p, r in zip(precision, recall):
                # if class_names are ints and labels are set
                if labels is not None and (
                    isinstance(class_name, int) or isinstance(class_name, np.integer)
                ):
                    class_name = labels[class_name]
                # if class_names are ints and labels are not set
                # or, if class_names have something other than ints
                # (string, float, date) - user class_names
                data.append([class_name, round(p, 3), round(r, 3)])
                count += 1
                if count >= chart_limit:
                    wandb.termwarn(
                        "wandb uses only the first %d datapoints to create the plots."
                        % wandb.Table.MAX_ROWS
                    )
                    break
        table = wandb.Table(columns=["class", "precision", "recall"], data=data)
        title = title or "Precision v. Recall"
        return wandb.plot_table(
            "wandb/area-under-curve/v0",
            table,
            {"x": "recall", "y": "precision", "class": "class"},
            {"title": title},
        )
