import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis

# ── LOAD DATA ────────────────────────────────────────────────────────
# Same files we have been using throughout the course
# 97.mat  = healthy bearing, 48kHz sampling rate
# 105.mat = faulty bearing (inner race), 12kHz sampling rate
normal = scipy.io.loadmat('97.mat')
fault  = scipy.io.loadmat('105.mat')

# Extract raw vibration signals
normal_signal = normal['X097_DE_time'].flatten()
fault_signal  = fault['X105_DE_time'].flatten()

# Sampling frequencies — verified in session 3
fs_normal = 48000
fs_fault  = 12000

#BEARING SPECIFICATIONS 
#CWRU Bearing model 6205-2RS JEM SKF 
#These values are used to calculate fault frequencies
Nb = 9
Bd = 0.3126
Pd = 1.537
RPM = 1797
angle = 0
 
#Convert RPM to Hz - shaft frequency in cycles per second
shaft_freq = RPM / 60   #= 29.95Hz

#Calculate the bearing fault frequencies
#BPFI = Ball Pass Frequency Inner Race
#Every time a rolling element passes over the inner race defect 
# it produces an impact at this frequency
BPFI = (Nb/2) * shaft_freq * (1 + (Bd/Pd) * np.cos(np.radians(angle)))

#BPFO 
#Same like BPFI but for outerrace defects
BPFO = (Nb/2) * shaft_freq * (1 - (Bd/Pd) * np.cos(np.radians(angle)))

#BSF = Ball spin frequency 
#The Frequency at which the ball itself rotates
BSF = (Pd/(2*Bd)) * shaft_freq * (1 - (Bd/Pd)**2 * np.cos(np.radians(angle))**2)

print("BEARING FAULT FREQUENCIES")
print(f"Shaft frequency : {shaft_freq:.2} Hz")
print(f"BPFI            : {BPFI:.2} Hz (inner race fault frequency)")
print(f"BPFO            : {BPFO:.2} Hz (outer race fault frequency)")
print(f"BSF            : {BSF:.2} Hz (ball spin fault frequency)")

#TIME DOMAIN FEATURE EXTRACTION FUNCTION 
#This function takes a raw signal and returns a dictionary of features
#A dictionary in python is a labelled collection - like a table with
#column names. Each feature has a name and a value.
#We use a dictionary so the features are self-describing -
#you always know which number corresponds to which feature

def extract_time_features(signal):
    features = {}

    #RMS Root Mean Square
    #The average energy level of the vibration
    #A rising RMS is the earliest and most reliable sign of
    #increasing machine stress or developing fault
    #Formula: square each value, take the maen, take the square root
    features['rms'] = np.sqrt(np.mean(signal**2))

    #PEAK - Maximum absolute value in the signal
    #Physical meaning: Single hardest impact recorded
    #Useful for detecting sudden shock events
    features['peak'] = np.max(np.abs(signal))

    #Pea-to-peak - the difference between the highest and lowest value
    #Physical menaing: the total swing range of vibration 
    #Large peak to peka with low RMS suggests intermittent impacts
    features['peak_to_peak'] = np.max(signal) - np.min(signal)

    #Crest Factor - Peak divided by RMS
    #How spiky the signal is relative to its energy
    #Healthy Bearing: typically 2 to 3
    #Early bearing fault: rises above 6
    #Last stage fault: May drop back as RMS catches up to peak
    #This is why crest factor is the best for EARLY fault detection
    features['crest_factor'] = features['peak'] / features['rms']

    #Kurtosis - statistical measure of spike extremity 
    #How impulsive are the peaks
    #Healthy signal follows a normal distribution - kurtosis=3
    #Bearing fault introduces non-Gaussian impulses - kurtosis rises
    #Very sensetive to ealry stage faults
    #scipy.stats.kurtosis() returns excess kurtosis (subtract 3)
    #so we add 3 to get the raw kurtosis value
    features['kurtosis'] = kurtosis(signal) + 3

    #Skewness - asymmetry of the signal distribution
    #Physical meaning: whether posistive or negative peaks dominate
    #A healthy bearing signal should be roughly symmetric (skewness near 0)
    #Asymmetric loading or shaft bow can cause non-zero skewness 
    features['skewness'] = skew(signal)

    #Shape Factor - RMS divided by mean absolute value
    #Physical meaning: describes the waveform shape
    #Sensitive to changes in signal pattern even when amplitude is stable
    features['shape_factor'] = features['rms'] / np.mean(np.abs(signal))

    #Impulse Factor - Peak divided by mean absolute value
    #Physical Meaning: combines the peak seinsitivity with average level
    #More sensitive to impulses than crest factor at early fault stages
    features['impulse_factor'] = features['peak'] / np.mean(np.abs(signal))

    #Variance - average squared deviation from the mean \
    #Physical Meaning: How spread out the vibrations values are
    #Directly related to RMS but without the square root
    features['variance'] = np.var(signal)

    #Standard deviation - square root of variance
    #Physical meaning: average deviation from the mean vibration level
    #For a zero-mean signal this equals RMS
    features['std_dev'] = np.std(signal)

    return features

#FREQUENCY DOMAIN FEATURE EXTRACTION FUNNCTION
# This function computes the FFT and extracts features from the spectrum
# Frequency domain features capture information that time domain misses
# specifically WHICH frequencies are elevated and by how much

def extract_frequency_features(signal, fs, bpfi, bpfo, bsf, shaft_freq):
    features = {}

    N = len(signal)

    # Compute FFT — convert time signal to frequency components
    spectrum  = np.abs(fft(signal))[:N//2] / N 
    freqs     = fftfreq(N, 1/fs)[:N//2]

    # Frequency resolution — how many Hz between each FFT bin
    # Higher resolution means we can pinpoint fault frequencies more precisely
    freq_resolution = fs / N
    features['freq_resolution'] = freq_resolution

    # Total spectral power — sum of all frequency component magnitudes
    # Physical meaning: total energy distributed across all frequencies
    features['total_power'] = np.sum(spectrum**2)

    # Helper function to get magnitude at a specific frequency
    # We find the FFT bin closest to our target frequency
    def magnitude_at_freq(target_freq):
        idx = np.argmin(np.abs(freqs - target_freq))
        return spectrum[idx]
    
    # Magnitude at shaft frequency
    # Physical meaning: how much vibration at the rotation speed
    # High shaft frequency component suggests imbalance or misalignment
    features['shaft_magnitude'] = magnitude_at_freq(shaft_freq)

    # Magnitude at BPFI — inner race fault frequency
    # Physical meaning: if this is elevated, inner race fault is present
    # This is the exact frequency we found in session 3
    features['bpfi_magnitude'] = magnitude_at_freq(bpfi)

    # Magnitude at BPFO — outer race fault frequency
    # Physical meaning: if this is elevated, outer race fault is present
    features['bpfo_magnitude'] = magnitude_at_freq(bpfo)

    # Magnitude at BSF — ball spin frequency
    # Physical meaning: if this is elevated, ball fault is present
    features['bsf_magnitude'] = magnitude_at_freq(bsf)

    # Magnitude at 2x BPFI — second harmonic of inner race fault
    # Physical meaning: harmonics confirm that the BPFI peak is a real fault
    # not random noise. If BPFI and 2xBPFI are both elevated — high confidence
    features['bpfi_2x_magnitude'] = magnitude_at_freq(bpfi * 2)

    # Magnitude at 3x BPFI — third harmonic
    features['bpfi_3x_magnitude'] = magnitude_at_freq(bpfi * 3)

    # Spectral centroid — the "center of mass" of the spectrum
    # Physical meaning: where most of the spectral energy is concentrated
    # A shift in spectral centroid indicates a change in the frequency
    # characteristics of the machine — useful for tracking degradation
    features['spectral_centroid'] = np.sum(freqs * spectrum) / np.sum(spectrum)

    # Band power — total power in specific frequency ranges
    # We divide the spectrum into three bands and measure energy in each
    # Low band:  0 to 1000 Hz  — shaft and low frequency components
    # Mid band:  1000 to 5000 Hz — bearing fault frequencies typically here
    # High band: 5000 Hz and above — high frequency resonance

    def band_power(low, high):
        mask = (freqs >= low) & (freqs < high)
        return np.sum(spectrum[mask]**2)

    features['low_band_power']  = band_power(0, 1000)
    features['mid_band_power']  = band_power(1000, 5000)
    features['high_band_power'] = band_power(5000, fs/2)

    return features

#EXTRACT ALL FEATURES
print("\n── EXTRACTING FEATURES ──────────────────────────────────────")

# Time domain features for both signals
normal_time_features = extract_time_features(normal_signal)
fault_time_features  = extract_time_features(fault_signal)

# Frequency domain features for both signals
normal_freq_features = extract_frequency_features(
    normal_signal, fs_normal, BPFI, BPFO, BSF, shaft_freq)
fault_freq_features  = extract_frequency_features(
    fault_signal, fs_fault, BPFI, BPFO, BSF, shaft_freq)

print("Feature extraction complete.")

#PRINT COMPARISON TABLE
# This table shows every feature side by side for healthy and faulty
# The ratio column shows how many times larger the fault value is
# A high ratio means that feature is very useful for fault detection

print("\n── TIME DOMAIN FEATURES ─────────────────────────────────────")
print(f"{'FEATURE':<20} {'HEALTHY':>12} {'FAULTY':>12} {'RATIO':>8}")
print("-" * 56)

for feature_name in normal_time_features:
    normal_val = normal_time_features[feature_name]
    fault_val  = fault_time_features[feature_name]

    # Calculate ratio — how much larger is the fault value
    # We use abs() to handle negative values like skewness
    if abs(normal_val) > 0:
        ratio = abs(fault_val) / abs(normal_val)
    else:
        ratio = 0

    print(f"{feature_name:<20} {normal_val:>12.4f} {fault_val:>12.4f} {ratio:>8.2f}x")

print("\n── FREQUENCY DOMAIN FEATURES ────────────────────────────────")
print(f"{'FEATURE':<22} {'HEALTHY':>14} {'FAULTY':>14} {'RATIO':>8}")
print("-" * 62)

for feature_name in normal_freq_features:
    normal_val = normal_freq_features[feature_name]
    fault_val  = fault_freq_features[feature_name]

    if abs(normal_val) > 0:
        ratio = abs(fault_val) / abs(normal_val)
    else:
        ratio = 0

    print(f"{feature_name:<22} {normal_val:>14.2f} {fault_val:>14.2f} {ratio:>8.2f}x")

#VISUALIZE FEATURE COMPARISON
#Bar chart comparing time domain features side by side
# Features where the bars are very different = useful for fault detection
# Features where bars look similar = less useful

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# ── TOP CHART — TIME DOMAIN FEATURES ─────────────────────────────────
time_feature_names  = list(normal_time_features.keys())
normal_time_values  = [normal_time_features[f] for f in time_feature_names]
fault_time_values   = [fault_time_features[f]  for f in time_feature_names]

x = np.arange(len(time_feature_names))
width = 0.35   # width of each bar

bars1 = axes[0].bar(x - width/2, normal_time_values,
                     width, label='Healthy', color='steelblue', alpha=0.8)
bars2 = axes[0].bar(x + width/2, fault_time_values,
                     width, label='Faulty',  color='crimson',   alpha=0.8)

axes[0].set_title('Time Domain Features — Healthy vs Faulty Bearing')
axes[0].set_xticks(x)
axes[0].set_xticklabels(time_feature_names, rotation=45, ha='right')
axes[0].set_ylabel('Feature Value')
axes[0].legend()

# ── BOTTOM CHART — FAULT FREQUENCY MAGNITUDES ────────────────────────
# We specifically compare the magnitudes at fault frequencies
# This shows directly whether the fault frequencies are elevated

freq_names = ['shaft', 'bpfi', 'bpfo', 'bsf', 'bpfi_2x', 'bpfi_3x']
freq_keys  = [f + '_magnitude' for f in freq_names]

normal_freq_vals = [normal_freq_features[k] for k in freq_keys]
fault_freq_vals  = [fault_freq_features[k]  for k in freq_keys]

x2 = np.arange(len(freq_names))

axes[1].bar(x2 - width/2, normal_freq_vals,
            width, label='Healthy', color='steelblue', alpha=0.8)
axes[1].bar(x2 + width/2, fault_freq_vals,
            width, label='Faulty',  color='crimson',   alpha=0.8)

axes[1].set_title('Fault Frequency Magnitudes — Healthy vs Faulty Bearing')
axes[1].set_xticks(x2)
axes[1].set_xticklabels([
    f'Shaft\n{shaft_freq:.1f}Hz',
    f'BPFI\n{BPFI:.1f}Hz',
    f'BPFO\n{BPFO:.1f}Hz',
    f'BSF\n{BSF:.1f}Hz',
    f'2xBPFI\n{BPFI*2:.1f}Hz',
    f'3xBPFI\n{BPFI*3:.1f}Hz'
], rotation=0, ha='center')
axes[1].set_ylabel('Magnitude')
axes[1].legend()

plt.tight_layout()
plt.show()

#FEATURE IMPORTANCE RANKING
# We rank features by their ratio — highest ratio = most useful
# for distinguishing healthy from faulty bearing
# This tells you which features to prioritize in your ML models

print("\n── FEATURE IMPORTANCE RANKING (by ratio) ───────────────────")
print("Higher ratio = better at separating healthy from faulty")
print("-" * 45)

all_features = {}
all_features.update(normal_time_features)

all_ratios = {}
for name in normal_time_features:
    n = normal_time_features[name]
    f = fault_time_features[name]
    all_ratios[name] = abs(f)/abs(n) if abs(n) > 0 else 0

# Sort by ratio — highest first
sorted_features = sorted(all_ratios.items(),
                          key=lambda x: x[1], reverse=True)

for rank, (name, ratio) in enumerate(sorted_features, 1):
    bar = '█' * min(int(ratio * 5), 40)
    print(f"{rank}. {name:<20} {ratio:>6.2f}x  {bar}")

print("\nSession 5 complete.")
print("You now have a complete feature engineering library.")
print("These functions can be reused in any future project.")