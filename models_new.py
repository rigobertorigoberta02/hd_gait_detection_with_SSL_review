import random
from copy import deepcopy
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold,GroupShuffleSplit
import sslmodel
from torch.utils.data import DataLoader
import ipdb


class GaitDetectorSSL:
    def __init__(
            self,
            window_sec=10,
            sample_rate=30,
            walk_percent=0.4,
            pnr=1.0,
            cv=5,
            device='cpu',
            weights_path='state_dict.pt',
            repo_tag='v1.0.0',
            verbose=False,
            n_jobs=1,
            output_logits=True
    ):
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.walk_percent = walk_percent
        self.pnr = pnr
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.device = device
        self.weights_path = weights_path
        self.repo_tag = repo_tag
        self.state_dict = None
        self.verbose = verbose
        self.window_len = int(np.ceil(self.window_sec * self.sample_rate))
        self.cv_scores = None
        self.output_logits = output_logits

    def cross_val(self, X, Y, params, groups=None, return_f1=False):
        # train walk detector & cross-val-predict
        if self.verbose:
            print("Running cross_val_predict...")

        Wlog, Wp, cv_test_idxs = cvp(
            self, X, Y, params, groups=groups,
            fit_predict_groups=True,
            n_splits=self.cv,
            n_jobs=self.n_jobs,
            return_indices=True,
        )


        if self.verbose:
            print("Fitting walk detector...")
        self.fit(X, Y, params, groups=groups)

        sample_weight = calc_sample_weight(Y, self.pnr)

        # TODO Ive changed the Y from 2 dim to 1 dim
        # Convert to one-dimension labels
        Y = np.argmax(Y, axis=1)
        
        if self.output_logits is False:
            # performance -- walk detector

            _, wd_scores = get_cv_scores(
                Y, Wp, cv_test_idxs,
                sample_weight=sample_weight,
                scorer_type='classif'
            )

            self.cv_results = {
                'test_indices': cv_test_idxs,
                'groups': groups,
                'walk_detector': {
                    'scores': wd_scores,
                    'y_true': Y,
                    'y_pred': Wp,
                },
            }

        if return_f1:
            return self.cv_results['walk_detector']['scores']
        else:
            return self, Y, Wp, Wlog

    def fit(self, X, Y,  params, groups=None):
        sslmodel.verbose = self.verbose

        if self.verbose:
            print('Training SSL')

        # prepare training and validation sets
        folds = GroupShuffleSplit(
            1, test_size=0.2, random_state=41
        ).split(X, Y, groups=groups)
        train_idx, val_idx = next(folds)
        x_train = X[train_idx]
        x_val = X[val_idx]

        y_train = Y[train_idx]
        y_val = Y[val_idx]

        group_train = groups[train_idx]
        group_val = groups[val_idx]

        train_dataset = sslmodel.NormalDataset(x_train, y_train, pid=group_train, name="training")
        val_dataset = sslmodel.NormalDataset(x_val, y_val, pid=group_val, name="validation")

        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=1,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=1,
        )

        # balancing to 90% notwalk, 10% walk
        walk = np.sum(y_train[:,1])
        notwalk = np.sum(y_train[:, 0])
        class_weights = [(walk * 9.0) / notwalk, 1.0]

        model = sslmodel.get_sslnet(tag=self.repo_tag, pretrained=True)
        model.to(self.device)


        # # Freeze the parameters of the first n_layers
        # n_layers = params['n_layers']
        # for name, param in model.named_parameters():
        #     # Always fine tune the classifier layers
        #     if name.startswith('classifier'):
        #         param.requires_grad = True
        #     else:
        #         layer = name.split('.')[1] # e.g., convert feature_extractor.layer2.2.bn1.bias -> layer2
        #         layer_num = int(layer[-1]) # e.g., convert layer2 -> 2
        #         if layer_num >= n_layers:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False



        sslmodel.train(model, train_loader, val_loader, self.device, class_weights, weights_path=self.weights_path)
        model.load_state_dict(torch.load(self.weights_path, self.device))

        # move model to cpu to get a device-less state dict (prevents device conflicts when loading on cpu/gpu later)
        model.to('cpu')
        self.state_dict = model.state_dict()

        return self

    def predict(self, X, params, groups=None):
        sslmodel.verbose = self.verbose

        dataset = sslmodel.NormalDataset(X, name='prediction')
        dataloader = DataLoader(
            dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=1,
        )

        model = sslmodel.get_sslnet(tag=self.repo_tag, pretrained=False)
        model.load_state_dict(self.state_dict)
        model.to(self.device)
        _, y_logits,y_pred, _ = sslmodel.predict(model, dataloader, self.device)

        return y_logits,y_pred

    def predict_from_frame(self, data, **kwargs):

        def fn(chunk):
            """ Process the chunk. Apply padding if length is not enough. """
            n = len(chunk)
            x = chunk[['x', 'y', 'z', 'annotation']].to_numpy()
            if n > self.window_len:
                x = x[:self.window_len]
            if n < self.window_len:
                m = self.window_len - n
                x = np.pad(x, ((0, m), (0, 0)), mode='wrap')
            return x

        X, T = make_windows(data, self.window_sec, fn=fn, return_index=True)
        L = X['annotation']
        X = X['x', 'y', 'z']
        X = np.asarray(X)
        Y = self.predict(X, **kwargs)
        Y = pd.Series(Y, index=T)
        return Y, L

def make_windows(data, window_sec, fn=None, return_index=False):
    """ Split data into windows """

    if fn is None:
        def fn(x):
            return x

    X = [fn(x) for _, x in data.resample(f"{window_sec}s", origin="start")]

    if return_index:
        T = (
            data.index
            .to_series()
            .resample(f"{window_sec}s", origin="start")
            .first()
        )
        return X, T

    return X


def cvp(
    model, X, Y, params, groups,
    method='predict',
    fit_predict_groups=False,
    return_indices=False,
    n_splits=5,
    n_jobs=1,
):
    """ Like cross_val_predict with custom tweaks """
    if n_splits == -1:
        n_splits = len(np.unique(groups))

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    def worker(train_idxs, test_idxs):
        X_train, Y_train, groups_train = X[train_idxs], Y[train_idxs], groups[train_idxs]
        X_test, Y_test, groups_test = X[test_idxs], Y[test_idxs], groups[test_idxs]

        m = deepcopy(model)

        if fit_predict_groups:
            m.fit(X_train, Y_train, params, groups=groups_train)
            Y_test_logits,Y_test_pred = getattr(m, method)(X_test, params, groups=groups_test)
        else:
            m.fit(X_train, Y_train)
            Y_test_logits,Y_test_pred = getattr(m, method)(X_test, params)

        return Y_test_logits, Y_test_pred, test_idxs

    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(train_idxs, test_idxs)
        for train_idxs, test_idxs in groupkfold(groups, n_splits)
    )

    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(worker)(train_idxs, test_idxs)
    #     for train_idxs, test_idxs in groupkfold(X, Y, groups, n_splits)
    # )
    
    Y_logits = np.concatenate([r[0] for r in results])
    Y_pred = np.concatenate([r[1] for r in results])
    cv_test_idxs = [r[2] for r in results]

    if return_indices:
        return Y_logits, Y_pred, cv_test_idxs

    return Y_pred

def groupkfold(groups, n_splits=5):
    """ Like GroupKFold but ordered """

    ord_unq_grps = groups[np.sort(np.unique(groups, return_index=True)[1])]
    folds_unq_grps = np.array_split(ord_unq_grps, n_splits)

    for unq_grps in folds_unq_grps:
        mask = np.isin(groups, unq_grps)
        test_idxs = np.nonzero(mask)
        train_idxs = np.nonzero(~mask)
        yield train_idxs, test_idxs




# def groupkfold(groups, n_splits=5):
#     """ Stratified grouping of folds, with all samples of each subject in a single fold """
#     unique_subjects = np.unique(groups)
#     cohort_labels = np.array([subject.split("_")[0] for subject in unique_subjects])
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     fold_subjects = [[] for _ in range(n_splits)]
#
#     for cohort, cohort_idxs in enumerate(skf.split(np.zeros_like(cohort_labels), cohort_labels)):
#         _, cohort_test_idxs = cohort_idxs
#         cohort_subjects = unique_subjects[cohort_idxs[1]]
#         for subject in cohort_subjects:
#             # randomly assign the subject's samples to a fold
#             fold_idx = np.random.choice(range(n_splits))
#             fold_subjects[fold_idx].append(subject)
#
#     for fold_subjects in fold_subjects:
#         fold_idxs = np.concatenate([np.where(groups == subject)[0] for subject in fold_subjects])
#         test_idxs = fold_idxs.astype(int)
#         train_idxs = np.setdiff1d(np.arange(len(groups)), test_idxs)
#         yield train_idxs, test_idxs

# def groupkfold(X, Y, groups, n_splits):
#     y = np.argmax(Y, axis=1) # for 2d to 1d labels
#     sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=42, shuffle=True)
#     for _, (train_idxs, test_idxs) in enumerate(sgkf.split(X, y, groups)):
#         yield train_idxs, test_idxs




def get_cv_scores(yt, yp, cv_test_idxs, sample_weight=None, scorer_type='classif'):
    # ToDo: add , sample_weight=sample_weight
    classif_scorers = {
        'accuracy': lambda yt, yp, sample_weight=None: metrics.accuracy_score(yt, yp),
        'f1': lambda yt, yp, sample_weight=None: metrics.f1_score(yt, yp),
        'precision': lambda yt, yp, sample_weight=None: metrics.precision_score(yt, yp),
        'recall': lambda yt, yp, sample_weight=None: metrics.recall_score(yt, yp),
        'balanced_accuracy': lambda yt, yp, sample_weight=None: metrics.balanced_accuracy_score(yt, yp),
        'auc': lambda yt, yp, sample_weight=None: metrics.roc_auc_score(yt, yp),
        'roc_curve': lambda yt, yp, sample_weight=None: metrics.roc_curve(yt, yp),
        'confusion_matrix': lambda yt, yp, sample_weight=None: metrics.confusion_matrix(yt, yp)
    }

    regress_scorers = {
        'mae': lambda yt, yp, sample_weight: metrics.mean_absolute_error(yt, yp, sample_weight=sample_weight),
        'rmse': lambda yt, yp, sample_weight: metrics.mean_squared_error(yt, yp, sample_weight=sample_weight, squared=False),
        'mape': lambda yt, yp, sample_weight: smooth_mean_absolute_percentage_error(yt, yp, sample_weight=sample_weight),
    }

    def smooth_mean_absolute_percentage_error(yt, yp, sample_weight=None):
        yt, yp = yt.copy(), yp.copy()
        # add 1 where zero to smooth the mape
        whr = yt == 0
        yt[whr] += 1
        yp[whr] += 1
        return metrics.mean_absolute_percentage_error(yt, yp, sample_weight=sample_weight)

    if scorer_type == 'classif':
        scorers = classif_scorers
    elif scorer_type == 'regress':
        scorers = regress_scorers
    else:
        raise ValueError(f"Unknown {scorer_type}")

    raw_scores = defaultdict(list)

    for idxs in cv_test_idxs:
        yt_, yp_, sample_weight_ = yt[idxs], yp[idxs], sample_weight[idxs]
        for scorer_name, scorer_fn in scorers.items():
            raw_scores[scorer_name].append(scorer_fn(yt_, yp_, sample_weight=sample_weight_))

    summary = {}
    for key, val in raw_scores.items():
        q0, q25, q50, q75, q100 = np.quantile(val, (0, .25, .5, .75, 1))
        avg, std = np.mean(val), np.std(val)
        if key=='confusion_matrix':
            avg=np.mean(val,axis=0)
        summary[key] = {
            'min': q0, 'Q1': q25, 'med': q50, 'Q3': q75, 'max': q100,
            'mean': avg, 'std': std,
        }

    return raw_scores, summary

def calc_sample_weight(yt, pnr=None):
    sample_weight = np.ones_like(yt, dtype='float')
    if pnr is None:
        return sample_weight
    sample_weight[yt == 0] = (yt == 1).sum() / (pnr * (yt == 0).sum())
    return sample_weight


