import scipy.io
import numpy as np
import matplotlib.pyplot as plt

#Load the data files 
normal = scipy.io.loadmat('97.mat')
fault = scipy.io.loadmat('105.mat')

#Extract Vibration Signals
# .flatten() converts the data into a simple list of numbers
normal_signal = normal['X097_DE_time'].flatten()
fault_signal = fault['X105_DE_time'].flatten()

#SAMPLING FREQUENCY
#These two files were collected at different frequencies
#Normal baseline at 48kHz
#Fault file at 12kHz
fs_normal = 48000
fs_fault = 12000

#COMPUTE FFT FOR NORMAL SIGNAL
#np.fft.fft() converts the time signal into frequency components
#abs() gives the magnitude - how strong each frequency is
#We only keep the first half - the second half is always a mirror image
N_normal = len(normal_signal)
normal_fft = np.abs(np.fft.fft(normal_signal))[:N_normal//2]
normal_freqs = np.fft.fftfreq(N_normal, 1/fs_normal)[:N_normal//2]

#COMPUTE FFT FOR FAULT SIGNAL
#same process - but using fs_fault = 12000
#This ensures the Hz values on the graph area are accurate for this file
N_fault = len(fault_signal)
fault_fft = np.abs(np.fft.fft(fault_signal))[:N_fault//2]
fault_freqs = np.fft.fftfreq(N_fault, 1/fs_fault)[:N_fault//2]

#CALCULATE BPFI
#BPFI = Ball Pass Frequency Inner Race
#The exact frequency at which rolling elements strike the inner race defect 
#Calculated purely from bearing geometry and shaft speed
#If inner race fault exists, a spike must appear at this frequency 

#FORMULA: BPFI = (Nb/2) * (RPM/60) * (1 + (Bd/Pd) * cos(angle))

#CWRU bearing model 6205-2RS JEM SKF specifications:
#Nb = 9       number of rolling elements
#Bd = 0.3126  ball diameter in inches
#Pd = 1.537   pitch diameter in inches
#angle = 0    contact angle in degrees
#RPM = 1797   shaft speed during test

Nb = 9
Bd = 0.3126
Pd = 1.537
RPM = 1797
angle = 0

#np.radians() converts degrees to radians - required by np.cos()
BPFI = (Nb / 2) * (RPM / 60) * (1 + (Bd / Pd)) * np.cos(np.radians(angle))

print(f"Calculated BPFI : {BPFI:.2f} Hz")
print(f"Physical meaning: the inner race defect is struck {BPFI:.0f} times per second")
print(f"On the fault spectrum - look for a spike at this frequecy")

#PLOT THE FREQUENCY SPECTRA
#Zooming into 0 to 2000Hz - bearing fault frequencies live here
#The full spectrum goes 24,000 Hz but is mostly empty aove 2000 Hz

plt.figure(figsize = (12,6))

#TOP GRAPH - HEALTHY BEARING
#A healthy bearing should show No spike at the BPFI frequency
#The spectrum should look relatively flat with no dominant spikes
plt.subplot(2, 1, 1)
plt.plot(normal_freqs, normal_fft, color='Steelblue', linewidth=0.5)
plt.xlim(0, 2000)
plt.title('Healthy Bearing Frequency Spectrum')
plt.ylabel('Magnitude')
plt.axvline(x=BPFI, color='green', linestyle='--', label=f'BPFI = {BPFI:.1f} Hz')
plt.legend()

#BOTTOM GRAPH - FAULTY BEARING
#A damaged inner race bearing MUST show a spike at the BPFI frequency 
#That spike is the physical impact of rolling element hitting the crack
#You may also see the smaller spikes at 2*BPFI, 3*BPFI - called harmonics
plt.subplot(2, 1, 2)
plt.plot(fault_freqs, fault_fft, color='crimson', linewidth=0.5)
plt.xlim(0, 2000)
plt.title('Faulty Bearing(Inner Race) Frequency Spectrum')
plt.ylabel('Magnitude')
plt.xlabel('Frequency (Hz)')
plt.axvline(x=BPFI, color='green', linestyle='--', label=f'BPFI = {BPFI:.1f} Hz')
plt.legend()

#Also mark the second and third harmoics - multiples of BPFI
#Harmonics appearing confirms the fault is real, not random noise
plt.axvline(x=BPFI*2, color='Orange', linestyle='--',
            label=f'2* BPFI = {BPFI*2:.1f} Hz')
plt.axvline(x=BPFI*3, color='red', linestyle='--',
            label=f'3× BPFI = {BPFI*3:.1f} Hz')
plt.legend()

plt.tight_layout()
plt.show()

print("/n WHAT TO LOOK FOR")
print("Green line = BPFI - Where the fault frequency should appear")
print("Orange line = 2*BPFI - First harmonic")
print("Red line = 3*BPFI - Second harmonic")
print("Healthy Bearing : No spikes at these locations")
print("Faulty bearing : Spikes visible at green, possibly orange and red")
print("Harmonics appearing together is strong confirmation of inner race fault")