import matplotlib.pyplot as plt
import wave
import numpy as np
import struct

def bez2(t,w):
  t2 = t * t
  mt = 1-t
  mt2 = mt * mt
  return w[0]*mt2 + w[1]*2*mt*t + w[2]*t2

def bez3(t,w):
  t2 = t * t
  t3 = t2 * t
  mt = 1-t
  mt2 = mt * mt
  mt3 = mt2 * mt
  return w[0]*mt3 + 3*w[1]*mt2*t + 3*w[2]*mt*t2 + w[3]*t3

def microstep(ms):
	return [ms * x for x in range(int(ms**-1)+1)]

wx = [10, 52.5, 95]
wy = [80, 10, 80]
length_seconds = 2
sample_rate = 48_000
bits = 16
type_bits = np.int16
format_bits = '<h'
byte_width = int(bits / 8)
length = sample_rate * length_seconds
amplitude = (2**(bits-1))-1 # max for n-bit audio
print(f"B: {bits}")
print(f"BW: {byte_width}")
print(f"MAX: {amplitude}")
#y = amplitude * np.sin(freq * 2 * np.pi * length)
#y = y.astype(np.int16)
time = microstep((sample_rate*length_seconds)**-1)
frequencies = np.array([bez2(t, wy) for t in time]) * 8

phases = np.pi * np.cumsum(frequencies) / sample_rate
sine_wave = amplitude * np.sin(phases)
sine_wave = sine_wave.astype(type_bits)

wav = wave.open("out.wav", 'wb')
wav.setnchannels(1)
wav.setframerate(sample_rate)
wav.setsampwidth(byte_width)
wav.writeframes(sine_wave)

# Plot the Bézier curve (frequency modulation)
plt.figure(figsize=(10, 4))
plt.plot(time, frequencies)
plt.title("Frequency Modulation using Bézier Curve")
plt.xlabel("Time (normalized)")
plt.ylabel("Frequency (Hz)")
plt.grid()

# Plot the first part of the sine wave
plt.figure(figsize=(10, 4))
plt.plot(time[:2000], sine_wave[:2000])  # Show the first 1000 samples
plt.title("Generated Sine Wave (Zoomed In)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
