#
#
# Study of https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm
#
#
# GIVES PUB LB: 0.778
# AUC: 0.785154, 0.778472, 0.781327, 0.786876, 0.782118
# => 0.782675
# LB 778

# 783061, 791012, 774942, 783135, 777430, 786645, 789287, 785450, 779907, 785058
# => 0.785058
# 
# LB 0.777

y, data, test = preprocess()



####################################################################################


from lightgbm import LGBMClassifier
import gc

gc.enable()

num_splits = 5

folds = KFold(n_splits=num_splits, shuffle=True, random_state=11)
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])
feats = [f for f in data.columns if f not in ['SK_ID_CURR']]

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
    trn_x, trn_y = data[feats].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = data[feats].iloc[val_idx], y.iloc[val_idx]
    
    clf = LGBMClassifier(
        n_estimators=4000,
        learning_rate=0.03,
        num_leaves=30,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2,
        silent=-1,
        verbose=-1,
    )
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=100, early_stopping_rounds=100  #30
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
    
print('Full AUC score %.6f' % roc_auc_score(y, oof_preds)) 

test['TARGET'] = sub_preds

test[['SK_ID_CURR', 'TARGET']].to_csv(DATA_DIR + 'a_25MAY_1.csv', index=False)



