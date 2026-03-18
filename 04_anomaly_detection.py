import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

#LOAD THE DATA
#Same as before
normal = scipy.io.loadmat('97.mat')
fault  = scipy.io.loadmat('105.mat')

# Extract the vibration signals
normal_signal = normal['X097_DE_time'].flatten()
fault_signal  = fault['X105_DE_time'].flatten()

#EXTRACT FEATURES FROM WINDOWS
#Instead of analysing the entire signal as one block, 
# we split it into small windows of 1000 samples each
#Each window represents a shot snapshot of the machines behaviour 
#This is how real monitoring systems worl - tthey process 
# chunks of data continously, not one giant at a time
#Think of it like taking a photo every few seconds
#Instead of one long video you analyze at the end

def extract_features(signal, window_size=1000):
    features = [] #Empty list to store results

    #Loop through the signal in steps of window_size
    #Each loop iteration processes one window
    for i in range(0, len(signal) - window_size, window_size):

        #Cut out window from the signal
        window = signal[i : i + window_size]

        #Calculate 4 features for this window
        #These are the features we calculated in session 2 
        # but now we calculate them for every window separately

        #RMS - avarage enerygy in the window
        rms = np.sqrt(np.mean(window**2))

        #PEAK - the highest single value in the window
        peak = np.max(np.abs(window))

        #CREST FACTOR - how spiky this window is 
        crest = peak/rms

        #Kurtosis - How extreme are the spikes statistically
        kurtosis = np.mean((window - np.mean(window))**4) / np.std(window)**4

        #Add all 4 features as one row to our list
        features.append([rms, peak, crest, kurtosis])

    #Convert the list to numpy array - required format for scikit-learn
    return np.array(features)

#CALCULATE FEATURES FOR BPTH SIGNALS 
print("Extracting features from healthy bearing signal...")
normal_features = extract_features(normal_signal)

print("Extracting Features form faulty bearing signal...")
fault_features = extract_features(fault_signal)

print(f"Healthy windows : {len(normal_features)}")
print(f"Faulty windows : {len(fault_features)}")

#TRAIN THE ISOLATION FOREST 
#This is the machine learning step
#We train ONLY on healthy data - the model never sees a fault
#contamination = 0.05 means we expect about 5% of readings
#to look unusual even in healthy data - this is normal
#random_state = 42 just means the results are reproducible
#every time you run it - same random seed, same result

print("/n Training Isolation Forest on healthy data only...")
model = IsolationForest(contamination = 0.05, random_state=42)
model.fit(normal_features)
print("Training complete.")

#````SCORE ALL DATA````
#Now we ask the model to score every window
#decision_function() returns a score for each window
#More negative score = more abnormal
#We multiply by -1 so that higher number = more abnormal
#This makes it easier to read and set thresholds

normal_scores = -model.decision_function(normal_features)
fault_scores = -model.decision_function(fault_features)

print(f"/nHealthy data - average anomaly score : {np.mean(normal_scores):.4f}")
print(f"/n Faulty data - average anomaly score: {np.mean(fault_scores):.4f}")
print(f"/n If the model works correctly, faulty score should be higher than healthy score")

#PLOT THE ANOMALY SCORES
#We plot the anomaly scores for every window 
#A good model should show low scores for healthy data
#and clearly higher  scores for faulty data

plt.figure(figsize=(12, 6))

#Top graph - healthy bearing anomaly scores over time
plt.subplot(2,1,1)
plt.plot(normal_scores, color='Steelblue', linewidth=0.8)
plt.axhline(y=0.1, color='red', linestyle='--', label='Alert threshold')
plt.title('Healthy Bearing - Anomaly Scores Over Time')
plt.ylabel('Anomaly Score')
plt.legend()

# Bottom graph — faulty bearing anomaly scores over time
plt.subplot(2, 1, 2)
plt.plot(fault_scores, color='crimson', linewidth=0.8)
plt.axhline(y=0.1, color='red', linestyle='--', label='Alert threshold')
plt.title('Faulty Bearing — Anomaly Scores Over Time')
plt.ylabel('Anomaly Score')
plt.xlabel('Window Number')
plt.legend()

plt.tight_layout()
plt.show()

#COUNT ALERTS
#Count how many windows crossed the alert threshold
#this tells us how often the system would have raised an alarm

threshold = 0.1
normal_alerts = np.sum(normal_scores > threshold)
fault_alerts = np.sum(fault_scores > threshold)

print(f"/n ALERT SUMMARY ")
print(f"Threshold set at : {threshold} ")
print(f"Healthy Bearing : {normal_alerts} alerts out of {len(normal_scores)} windows")
print(f"Faulty Bearing : {fault_alerts} alerts out of {len(fault_scores)} windows")
print(f"/n False alarm rate : {normal_alerts/len(normal_scores)*100:.1f}%")
print(f"/n Fault Detection : {fault_alerts/len(fault_scores)*100:.1f}%")