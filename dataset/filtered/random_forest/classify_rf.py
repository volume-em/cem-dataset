"""
Description:
------------

Fits a random forest model to an array of features that describe images and uses that model to
predict whether unlabeled images are "good" or "bad" quality. It's assumed that the calculate_rf_features.py
script has already been run and that some images have been manually labeled using the corrector.py utilities
running in a Jupyter notebook.

The results of this script are the roc curve plot on a randomly chosen validation set of images, the
model object as a .sav file and the model's predictions on all the features in the given features array.

Example usage:
--------------

python filter_rf.py {features_fpath} {labels_fpath} {savedir}

For help with arguments:
------------------------

python filter_rf.py --help
"""

import os, argparse, pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, plot_roc_curve
from sklearn.model_selection import train_test_split


#main function of the script
if __name__ == "__main__":
    
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Classifies a set of images by fitting a random forest to an array of descriptive features')
    parser.add_argument('features_fpath', type=str, metavar='features_fpath', help='Path to array file containing image features')
    parser.add_argument('labels_fpath', type=str, metavar='labels_fpath', help='Path to array file containing image labels (good or bad)')
    parser.add_argument('savedir', type=str, metavar='savedir', help='Directory in which to save model, roc curve, and predictions')
    
    #parse the arguments
    args = parser.parse_args()
    features_fpath = args.features_fpath
    labels_fpath = args.labels_fpath
    savedir = args.savedir

    #make sure the savedir exists
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    #load the features and labels arrays
	features = np.load(features_fpath)
	gt_labels = np.load(labels_fpath)

	#it's expected that the gt_labels were generated within a Jupyter notebook by
	#using the the corrector.py labeling utilities 
	#in that case the labels are text with the possible options of "good", "bad", and "none"
	#those with the label "none" are considered the unlabeled set and we make predictions
	#about their labels using the random forest that we train on the labeled images
	good_indices = np.where(gt_labels == 'good')[0]
	bad_indices = np.where(gt_labels == 'bad')[0]
	labeled_indices = np.concatenate([good_indices, bad_indices], axis=0)
	unlabeled_indices = np.setdiff1d(range(len(features)), labeled_indices)

	#setting the random seed ensures that we're always comparing results against the same
	#validation dataset (otherwise hyperparameter tuning results would be indecipherable)
	np.random.seed(1227)
	trn_indices, val_indices = train_test_split(labeled_indices, test_size=0.15)

	#convert the labels from text to integers (0 = "bad", 1= "good")
	labels = np.zeros((len(features), ))
	labels[good_indices] = 1

	#separate train and validation sets
	trn_features = features[trn_indices]
	trn_labels = labels[trn_indices]
	val_features = features[val_indices]
	val_labels = labels[val_indices]

	#fit the random forest model to the training data
	print(f'Fitting random forest model to {len(trn_features)} images...')
	rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced',  min_samples_split=8)
	rf = rf.fit(trn_features, trn_labels)

	#save the model object
	pickle.dump(rf, open(os.path.join(savedir, 'random_forest.sav'), 'wb'))

	#evaluate the model on the heldout validation test
	print(f'Evaluating model predictions...')
	val_predictions_rf = rf.predict_proba(val_features)[:, 1]
	tn, fp, fn, tp = confusion_matrix(val_labels, val_predictions_rf > 0.5).ravel()
	acc = accuracy_score(val_labels, val_predictions_rf > 0.5)

	print(f'Total validation images: {len(val_features)}')
	print(f'True Positives: {tp}')
	print(f'True Negatives: {tn}')
	print(f'False Positives: {fp}')
	print(f'False Negatives: {fn}')
	print(f'Accuracy: {acc}')

	#plot roc curve and save it
	plot_roc_curve(rf, val_features, val_labels)
	plt.savefig(os.path.join(savedir, "rf_roc_curve.png"))

	print(f'Predicting labels for {len(unlabeled_indices)} unlabeled images...')
	tst_features = features[unlabeled_indices]
	tst_predictions = rf.predict_proba(tst_features)[:, 1]

	#create an array of labels that are all zeros and fill in the values from a combination
	#of the ground truth labels from training and validation sets and the predicted
	#labels for unlabeled indices
	predicted_labels = np.zeros(len(features), dtype=np.uint8)
	predicted_labels[trn_indices] = trn_labels.astype(np.uint8)
	predicted_labels[val_indices] = val_labels.astype(np.uint8)
	predicted_labels[unlabeled_indices] = (tst_predictions > 0.5).astype(np.uint8)

	print(f'Saving results...')
	np.save(os.path.join(savedir, "rf_predictions.npy"), predicted_labels)

	print('Finished.')