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

'''
TO DO:
1- Decision
2- Check data acquisition

'''


import mules
import subprocess
import numpy as np
import bci_workshop_tools as BCIw
import matplotlib.pyplot as plt
import argparse
from clientmyo import ClientMyo
import time
from ml_utils import prepare_data, train_classifiers, predict_from_list, test_from_list, normalize
#from CSP import CSP_train, CSP_test
from calc_entropy import calc_entropy
from sklearn.metrics import accuracy_score
from clientepuck import ClientEpuck


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SMC Brain Hackathon')
    parser.add_argument('--training-secs', type = float, default = 15.0, help = 'Duration of each training iteration in seconds')
    parser.add_argument('--win-test-secs', type = float, default = 1.0, help = 'Length of the test window in seconds')
    parser.add_argument('--overlap-secs', type = float, default = 0.7, help = 'Overlap between two consecutive test windows')
    parser.add_argument('--max-feedback-loops', type = int, default = 10, help = 'Maximum number of training loops (feedback loops)')
    parser.add_argument('--training-accuracy-threshold', type = float, default = 0.7, help = 'Minimum accuracy for getting out of the training loop')
    parser.add_argument('--max-trainset-size', type = int, default = 2000, help = 'Minimum accuracy for getting out of the training loop')
    parser.add_argument('--csp', action = 'store_true', default = False, help = 'Enables CSP features')
    parser.add_argument('--entropy', action = 'store_true', default = False, help = 'Enables entropy features')
    parser.add_argument('--all-features', action = 'store_true', default = False, help = 'Enables all features')
    args = parser.parse_args()

    # MuLES connection paramters
    mules_ip = '127.0.0.1'
    mules_port = 30000

    # Following line runs the MyoClient.exe with the parameters required
    # Remember a "long" wait is needed
    #subprocess.Popen(['./myoclient_exe/myoserver_vi.exe', '--', 'COM8', 'ED:A2:9A:4A:41:EE'])
    #subprocess.Popen(['./myoclient_exe/myoserver_vi.exe', '--', 'COM27', 'F1:13:CA:63:69:76'])
    #time.sleep(25)

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
    
    # Connection paramters for epuck
    epuck_ip = '127.0.0.1'
    epuck_port = 50000 
       
    # Creates the e-puck client and stablish connection with the server
    epuck_client = ClientEpuck(epuck_ip, epuck_port) 
    time.sleep(2)

    # Set turn parameters for epuck turns
    epuck_client.set_turn_settings(200, 50, 1)

    # Training loop (Now it is a loop because we have feedback!!)

    loops_done = 0
    training_acc = 0.0

    i=3

    while ((loops_done < args.max_feedback_loops) and (training_acc < args.training_accuracy_threshold) and i!=0):

        # Record data for mental activity 0
        BCIw.beep()
        eeg_data0 = mules_client.getdata(args.training_secs)

        # Record data for mental activity 1
        BCIw.beep()
        eeg_data1 = mules_client.getdata(args.training_secs)    

        # Divide data into epochs
        eeg_epochs0 = BCIw.epoching(eeg_data0, round(args.win_test_secs * params['sampling frequency']), 
                                            round(args.overlap_secs * params['sampling frequency']))
        eeg_epochs1 = BCIw.epoching(eeg_data1, round(args.win_test_secs * params['sampling frequency']),    
                                            round(args.overlap_secs * params['sampling frequency']))

        # Compute features

        if (args.csp):
            #feat_matrix0, feat_matrix1, W_csp = CSP_train(eeg_epochs0, eeg_epochs1)
            pass
         
        else:
            feat_matrix0 = BCIw.compute_feature_matrix(eeg_epochs0, params['sampling frequency'])
            feat_matrix1 = BCIw.compute_feature_matrix(eeg_epochs1, params['sampling frequency'])

        if (args.entropy):
            entropy0 = calc_entropy(eeg_epochs0)
            entropy1 = calc_entropy(eeg_epochs1)
            feat_matrix0 = np.hstack([feat_matrix0, entropy0])
            feat_matrix1 = np.hstack([feat_matrix1, entropy1])

        data_points, labels, data_mean, data_std = prepare_data(feat_matrix0, feat_matrix1)

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
        
        print(total_labels.shape,ensemble_predictions.shape)

        # Feedback
        if (training_acc <= 0.4):
            
            # Send vibration 
            myo_client.vibrate('short')
            myo_client.vibrate('short')
            myo_client.vibrate('short')
            myo_client.vibrate('short')
            myo_client.vibrate('short')
            myo_client.vibrate('short')
            myo_client.vibrate('short')


        if (training_acc > 0.4 and training_acc < args.training_accuracy_threshold):
            
            # Send vibration 
            myo_client.vibrate('short') 
            myo_client.vibrate('short')
            myo_client.vibrate('short')
            myo_client.vibrate('short')


        if (training_acc >= args.training_accuracy_threshold):
            
            # Send vibration 
            myo_client.vibrate('short')

        loops_done += 1
        i-=1
        print(training_acc,loops_done)

                

    # Initialize the buffers for storing raw EEG and decisions

    mules_client.flushdata()  # Flushes old data from MuLES

    # Start pulling data and classifying in real-time

    BCIw.beep() # Beep sound

    # The try/except structure allows to quit the while loop by aborting the 
    # script with <Ctrl-C>
    print(' Press Ctrl-C in the console to break the While Loop')
    try:
            while True:
                eeg_data = mules_client.getdata(shift_secs, False) # Obtain EEG data from MuLES  
    
            # Compute features on "eeg_data"
                if args.csp:
                    #feat_vector = CSP_test(eeg_data, W_csp)
                    pass
                else:
                    feat_vector = BCIw.compute_feature_vector(eeg_data, params['sampling frequency'])
                
                if (args.entropy):
                    entropy = calc_entropy(eeg_data)
                    feat_vector = np.hstack([feat_vector, entropy])
                
                y_hat = ensemble_model.predict((feat_vector.reshape(1, -1)-data_mean)/(data_std+1e-15))
    
                if y_hat[0] == 1:
                    # Turn Right 
                    epuck_client.send_instruction('right')
                    # Pause 1 secoonds
                    time.sleep(1)
                elif y_hat[0] == 0:
                    # Turn left 
                    epuck_client.send_instruction('left')
                    # Pause 1 secoonds
                    time.sleep(1)
                else:
                    # stop 
                    epuck_client.send_instruction('stop')
                    # Pause 1 secoonds
                    time.sleep(1)
     
    except KeyboardInterrupt:
        
        mules_client.disconnect() # Close connection 