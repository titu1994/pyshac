import os
import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

import warnings

# compatible with both Python 2 and 3
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def get_classifier(num_trees=200, n_jobs=1, seed=None):
    """

    Returns:
         xgb.XGBClassifier
    """
    model = xgb.XGBClassifier(n_estimators=num_trees, random_state=seed, n_jobs=n_jobs)
    return model


def train_single_model(encoded_samples, labels, num_splits=5, n_jobs=1):
    """

    Args:
        encoded_samples (np.ndarray):
        labels (np.ndarray):

    Returns:
         xgb.XGBClassifier
    """
    score = 0.0
    model = None
    model_count = 0

    while score < 0.5:
        random_seed = model_count
        model = get_classifier(n_jobs=n_jobs, seed=random_seed)

        if num_splits > 1:
            kfold = StratifiedKFold(n_splits=num_splits, random_state=0)
            scores = cross_val_score(model, encoded_samples, labels, cv=kfold, n_jobs=1)
            score = np.mean(scores)

            if score >= 0.5:
                model.fit(encoded_samples, labels)
                preds = model.predict(encoded_samples)
                pred_count = np.sum(preds)

                if pred_count == 0. or pred_count == len(preds):  # all same class predictions on fit set
                    score = 0.0  # find another model
                else:
                    break

        else:
            model.fit(encoded_samples, labels)
            preds = model.predict(encoded_samples)
            pred_count = np.sum(preds)

            if pred_count == 0. or pred_count == len(preds):  # all same class predictions on fit set
                score = 0.0  # find another model
            else:
                break

        if model_count > 10:
            warnings.warn("Could not find any model after 10 iterations of cv training ! "
                          "Change the batch size or perhaps the evaluation function needs to be changed !")
            return None
        else:
            model_count += 1

    return model


def evaluate_models(encoded_samples, clfs, relax_checks=False):
    """

    Args:
        encoded_samples (np.ndarray):
        clfs (list(xgb.XGBClassifier)):
        relax_checks (int): If reset, will try to get samples which pass
            through all classifier checks. If given an integer, will allow
            samples who fail R classifier checks.

    Returns:

    """
    preds = []

    for clf in clfs:  # type: xgb.XGBClassifier
        pred = clf.predict(encoded_samples)
        preds.append(pred)

    results = np.array(preds)

    if int(relax_checks) == 0:
        results = np.all(results, axis=0)
        return results
    else:
        min_required = len(clfs) - relax_checks

        results = results.astype(np.int32)
        count = np.sum(results, axis=0)

        return count >= min_required


def save_classifiers(clfs, basepath='shac'):
    if 'classifiers' not in basepath:
        basepath = os.path.join(basepath, 'classifiers')

    if not os.path.exists(basepath):
        os.makedirs(basepath)

    path = os.path.join(basepath, 'classifiers.pkl')
    joblib.dump(clfs, path)
    print("Saved classifier #%d" % (len(clfs)))


def restore_classifiers(basepath='shac'):
    if 'classifiers' not in basepath:
        basepath = os.path.join(basepath, 'classifiers')

    path = os.path.join(basepath, 'classifiers.pkl')

    if os.path.exists(path):
        models = joblib.load(path)
        return models
    else:
        raise FileNotFoundError("Serialized classifier files not found at %s!" % basepath)
