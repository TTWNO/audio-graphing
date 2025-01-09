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
freq = 500
length = np.linspace(0, length_seconds, sample_rate * length_seconds)
amplitude = (2**16)-1 # max for 16-bit audio
y = amplitude * np.sin(freq * 2 * np.pi * length)
y = y.astype(np.int16)
data = np.array(list([1000-(bez2(t, wy)*10) for t in microstep((sample_rate*length_seconds)**-1)]))

phase = 0
sine_wave = []
for i in range(len(length)):
	freq = data[i]
	phase += 2 * np.pi * freq / sample_rate
	sine_wave.append(amplitude * np.sin(phase))
sine_wave = np.array(sine_wave).astype(np.int16)

wav = wave.open("out.wav", 'wb')
wav.setnchannels(1)
wav.setframerate(sample_rate)
wav.setsampwidth(2)

#data = np.array()

#wav.writeframes(data.astype(np.float16))
for ys in sine_wave:
	wav.writeframes(struct.pack('<h', ys))
