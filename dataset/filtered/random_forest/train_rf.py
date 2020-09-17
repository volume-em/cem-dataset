from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, average_precision_score, plot_precision_recall_curve, accuracy_score
from sklearn.model_selection import train_test_split


features = np.load('/data/IASEM/conradrw/data/images224_fpaths_qsf_rf_features.npy')
gt_labels = np.load('/data/IASEM/conradrw/data/images224_fpaths_qsf_rf_gt.npy')

good_indices = np.where(gt_labels == 'good')[0]
bad_indices = np.where(gt_labels == 'bad')[0]
labeled_indices = np.concatenate([good_indices, bad_indices], axis=0)
unlabeled_indices = np.setdiff1d(range(len(features)), labeled_indices)

np.random.seed(1227)
trn_indices, val_indices = train_test_split(labeled_indices, test_size=0.2)


trn_imfiles = imfiles[trn_indices]
#trn_features = features[labeled_indices]
#trn_labels = np.zeros((len(imfiles), ))
#trn_labels[good_indices] = 1
#trn_labels = trn_labels[labeled_indices]
trn_features = features[trn_indices]
labels = np.zeros((len(features), ))
labels[good_indices] = 1
trn_labels = labels[trn_indices]
val_features = features[val_indices]
val_labels = labels[val_indices]

#rf = ExtraTreesClassifier(max_depth=16, class_weight='balanced', min_samples_split=4)
rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced',  min_samples_split=8)
rf = rf.fit(trn_features, trn_labels)