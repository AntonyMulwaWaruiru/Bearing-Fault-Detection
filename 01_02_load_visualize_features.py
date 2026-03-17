#We need these tools to load and work with the data
import scipy.io 
import numpy as np
import matplotlib.pyplot as plt

#STEP 1: Load both files 
normal = scipy.io.loadmat('97.mat')
fault = scipy.io.loadmat('105.mat')
#scipy.io.loadmat reads .mat files into Python

#STEP 2: See what is inside the files
#A .mat file is like a box with laballed compartments 
#This prints the labels so we know what is inside
print("Normal file contains:", list(normal.keys()))
print("fault file contains:", list(fault.keys()))

#Pull out the actual vibration signal from each file 
normal_signal = normal['X097_DE_time'].flatten()
fault_signal = fault['X105_FE_time'].flatten()
# .flatten() converts it from a table shape into a simple list of numbers

#Print some basic information so that we can understand the data
print("Normal signal length:", len(normal_signal)) #how many data points
print("Fault signal length:", len(fault_signal))
print("Normal first 5 values:", normal_signal[:5]) #first 5 vibration readings
print("Fault first 5 values:", fault_signal[:5])

#Create a figure that is 12 inches wide and 5 inches tall
plt.figure(figsize=(12, 5))

#Draw the FIRST graph in a 2-row layout - top position (2,1,1)
plt.subplot(2, 1, 1)
plt.plot(normal_signal[:1000], color='steelblue') #plot first 1000 readings only
plt.title('Healthy Bearing - Vibration Signal')
plt.ylabel('Acceleration (g)') #unit of vibration measurement

#Draw the SECOND graph - bottom postion (2,1,2)
plt.subplot(2, 1, 2)
plt.plot(fault_signal[:1000], color='crimson') #same - first 1000 readings
plt.title('Faulty Bearing - Vibration Signal')
plt.ylabel('Acceleration (g)')

#Automatically adjust spacing so graphs do not overlap
plt.tight_layout()

#Open the window and display the graphs
plt.show(block=False)

#STEP 3: Calculate Health Indicators
#These are called "Statistical features" - single numbers that 
# summarize the entire signal. This is what machine laerning models actually use - not raw signal, but these summaries.

#RMS = Root Mean Square
#Think of it as the average energy in the signal
#A rising RMS means the machine is working harder than normal 
normal_rms = np.sqrt(np.mean(normal_signal**2))
fault_rms = np.sqrt(np.mean(fault_signal**2))

#PEAK = the single highest vibration value recorded
normal_peak = np.max(np.abs(normal_signal))
fault_peak = np.max(np.abs(fault_signal))

#CREST factor = Peak divided by rms
#Measures how spiky the sinal is
#A healthy bearing has a crest factor around 2-3
#A damaged bearing spikes above 6 - classic fault indicator
normal_crest = normal_peak / normal_rms
fault_crest = fault_peak / fault_rms

#KURTOSIS = measures how extreme the spikes are statistically 
#Healthy bearing: Kurtosis around 3
#damaged bearing: Kurtosis rises sharply - sometimes above 10
normal_kurtosis = float(
    np.mean((normal_signal - np.mean(normal_signal))**4) /
    np.std(normal_signal)**4
)
fault_kurtosis = float(
    np.mean((fault_signal - np.mean(fault_signal))**4) /
    np.std(fault_signal)**4
)

#STEP 4: Print a Comparison Table
print("=" * 45)
print(f"{'FEATURE':<20} {'HEALTHY':>10} {'FAULTY':>10}")
print("=" * 45)
print(f"{'RMS':<20} {normal_rms:>10.4f} {fault_rms:>10.4f}")
print(f"{'Peak':<20} {normal_peak:>10.4f} {fault_peak:>10.4f}")
print(f"{'Crest Factor':<20} {normal_crest:>10.4f} {fault_crest:>10.4f}")
print(f"{'Kurtosis':<20} {normal_kurtosis:>10.4f} {fault_kurtosis:>10.4f}")
print("=" * 45)


