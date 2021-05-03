####################################################
# HOME CREDIT XGBOOST WITH FEATURE SELECTION
####################################################
# combined_csv taken: https://www.kaggle.com/anlgvrk/home-credit-xgboost-dsmlbc4-gr2

# Importing essential libraries
import numpy as np
import pandas as pd
import time
import gc
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


# Defining timer to track progress
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def combined_df(num_rows=None):
    # Read data
    df = pd.read_csv('../input/home-credit-combined-dataset/combined_df.csv', nrows=num_rows)   # come from hc_lgbm.py
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df


# Filter Columns by Correlation
def corr_features(dataframe):
    corr_features = set()
    corr_matrix = dataframe.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                colname = corr_matrix.columns[i]
                corr_features.add(colname)

    return list(corr_features)


# Feature Selection
def drop_corr_columns(dataframe):
    # Divide in training/validation and test data
    train_df = dataframe[dataframe['TARGET'].notnull()]
    test_df = dataframe[dataframe['TARGET'].isnull()]
    drop_col = corr_features(train_df)
    train_df.drop(drop_col, axis=1, inplace=True)
    test_df.drop(drop_col, axis=1, inplace=True)
    return train_df, test_df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


###############################################
# XGB GBDT with KFold or Stratified KFold
###############################################
# Reference from https://www.kaggle.com/tunguz/xgb-simple-features

def kfold_xgb(train_df, test_df, num_folds=10, stratified=False, debug=False):
    print("Starting XGB. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1054)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1054)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        clf = XGBClassifier(
            learning_rate=0.01,
            n_estimators=10000,
            max_depth=4,
            min_child_weight=5,
            subsample=0.7,
            colsample_bytree=0.7,
            objective='binary:logistic',
            nthread=4,
            scale_pos_weight=2.5,
            seed=27,
            reg_lambda=1.2,
            tree_method='gpu_hist'
        )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats])[:, 1] / folds.n_splits  # - Uncomment for K-fold

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

        np.save("xgb_oof_preds_1", oof_preds)
        np.save("xgb_sub_preds_1", sub_preds)

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)
    display_importances(feature_importance_df)
    return feature_importance_df


#####################################
# Main Function
#####################################
def main(debug=False):
    num_rows = 10000 if debug else None
    with timer('Reading Combined Data'):
        df = combined_df(num_rows)
        print("Combined_df shape: ", df.shape)
    with timer("Filter Feature Selection by Correlation Filter"):
        train_df, test_df = drop_corr_columns(df)
        print("Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
        del df
        gc.collect()
    with timer("Run XGBoost with kfold"):
        kfold_xgb(train_df, test_df, num_folds=10, stratified=True, debug=debug)


if __name__ == "__main__":
    submission_file_name = "submission_DSMLBC4_Grp2.csv"
    with timer("Full model run"):
        main(debug=False)

