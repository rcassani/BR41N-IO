# -*- coding: utf-8 -*-
"""
BCI workshop 2015
Exercise 2: A Basic BCI
Description:
In this second exercise, we will learn how to use an automatic algorithm to 
recognize somebody's mental states from their EEG. We will use a classifier: 
a classifier is an algorithm that, provided some data, learns to recognize 
patterns, and can then classify similar unseen information.
"""


import mules
import numpy as np
import bci_workshop_tools as BCIw
import matplotlib.pyplot as plt
import argparse
from clientmyo import ClientMyo
import time
from ml_utils import prepare_data, train_classifiers, predict_from_list, test_from_list, normalize
from blablabla import blablabla


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='SMC Brain Hackathon')
	parser.add_argument('--training-secs', type = float, default = 5.0, help = 'Duration of each training iteration in seconds')
	parser.add_argument('--win-test-secs', type = float, default = 1.0, help = 'Length of the test window in seconds')
	parser.add_argument('--overlap-secs', type = float, default = 0.7, help = 'Overlap between two consecutive test windows')
	parser.add_argument('--max-feedback-loops', type = int, default = 10, help = 'Maximum number of training loops (feedback loops)')
	parser.add_argument('--training-accuracy-threshold', type = float, default = 0.7, help = 'Minimum accuracy for getting out of the training loop')
	parser.add_argument('--max-trainset-size', type = float, default = 0.7, help = 'Minimum accuracy for getting out of the training loop')
	parser.add_argument('--csp', action = store_true, default = False, help = 'Enables CSP features')
	args = parser.parse_args()

	# MuLES connection paramters
	mules_ip = '127.0.0.1'
	mules_port = 30000

    # Connection paramters
	myo_ip = '127.0.0.1'
	myo_port = 40000

	# Creates a mules_client
	mules_client = mules.MulesClient(mules_ip, mules_port) 
	# Device parameters    
	params = mules_client.getparams()

	# Creates the Myo client and stablish connection with the server
	myo_client = ClientMyo(myo_ip, myo_port) 
	time.sleep(2)

	# Set some parameters
	shift_secs = args.win_test_secs - args.overlap_secs   

	# Training loop (Now it is a loop because we have feedback!!)

	loops_done = 0
	training_acc = 0.0

	while ((loops_done < args.max_feedback_loops) & (training_acc < args.training_accuracty_threshold)):

		# Record data for mental activity 0
		BCIw.beep()
		eeg_data0 = mules_client.getdata(args.training_secs)

		# Record data for mental activity 1
		BCIw.beep()
		eeg_data1 = mules_client.getdata(args.training_secs)    

		# Divide data into epochs
		eeg_epochs0 = BCIw.epoching(eeg_data0, args.win_test_secs * params['sampling frequency'], 
		                                    args.overlap_secs * params['sampling frequency'])
		eeg_epochs1 = BCIw.epoching(eeg_data1, args.win_test_secs * params['sampling frequency'],    
		                                    args.overlap_secs * params['sampling frequency'])

		# Compute features

		if (args.csp):
	
			feat_matrix0 = blablabla
			feat_matrix1 = blablabla


		else:

			feat_matrix0 = BCIw.compute_feature_matrix(eeg_epochs0, params['sampling frequency'])
			feat_matrix1 = BCIw.compute_feature_matrix(eeg_epochs1, params['sampling frequency'])

		data_points, labels = prepare_data(feat_matrix0, feat_matrix1)

		try:
			total_features = np.vstack([total_features, data_points])
			total_labels = np.hstack([total_labels, labels])
		except NameError:
			total_features = data_points
			total_labels = labels

		# Train classifier    
		_, ensemble_model = train_classifiers(total_features, total_labels)


		#list_predictions = test_from_list(data_points, models)
		ensemble_predictions = ensemble_model.predict(total_features)
		training_acc = accuracy_score(total_labels, ensemble_predictions)

		# Feedback
		if (training_acc <= 0.4):
			
			# Send vibration 
			myo_client.vibrate('short')
			myo_client.vibrate('short')
			myo_client.vibrate('short')


		if (training_acc > 0.4 & training_acc < args.training_accuracty_threshold):
			
			# Send vibration 
			myo_client.vibrate('short') 
			myo_client.vibrate('short')


		if (training_acc >= args.training_accuracty_threshold):
			
			# Send vibration 
			myo_client.vibrate('short')

		loops_done += 1

				

	# Initialize the buffers for storing raw EEG and decisions

	mules_client.flushdata()  # Flushes old data from MuLES

	# Start pulling data and classifying in real-time

	BCIw.beep() # Beep sound

	# The try/except structure allows to quit the while loop by aborting the 
	# script with <Ctrl-C>
	print(' Press Ctrl-C in the console to break the While Loop')
	try:    
		
	while True: 
		    
			""" 1- ACQUIRE DATA """
			eeg_data = mules_client.getdata(shift_secs, False) # Obtain EEG data from MuLES  

			""" 2- COMPUTE FEATURES and CLASSIFY"""            
			# Compute features on "eeg_data"

			if (args.csp):
				feat_vector = blablabla
			
			else:
				feat_vector = BCIw.compute_feature_vector(eeg_data, params['sampling frequency'])

			y_hat = model.predict(normalize(feat_vector.reshape(1, -1)))

			# DECISION!!!!!!!!!

			plt.pause(0.001)
     
	except KeyboardInterrupt:
		
		mules_client.disconnect() # Close connection 
