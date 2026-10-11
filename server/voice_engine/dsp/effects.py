"""Stateful creative modules shared by recording and live CPU processing.

Mix/drive zero bypasses a module. The echo buffer is bounded to one second.
The pipeline keeps its final limiter after these modules.
"""
import math
import numpy as np


class Metallic:
    def __init__(self, fs):
        self.fs, self.phase = fs, 0.0

    def process(self, audio, p):
        mix = p['metallicMix']
        if not mix:
            return
        step = 2 * math.pi * p['metallicHz'] / self.fs
        for i in range(len(audio)):
            audio[i] *= 1 - mix + mix * math.sin(self.phase)
            self.phase = (self.phase + step) % (2 * math.pi)


class Distortion:
    def __init__(self, fs):
        pass

    def process(self, audio, p):
        drive = p['distortionDrive']
        if drive:
            gain = 1 + drive * 15
            audio[:] = np.tanh(audio.astype(np.float64) * gain) / gain


class Echo:
    def __init__(self, fs):
        self.fs, self.position = fs, 0
        self.buffer = np.zeros(int(fs) + 1, dtype=np.float32)

    def process(self, audio, p):
        mix = p['echoMix']
        if not mix:
            self.buffer.fill(0)
            return
        delay = max(1, int(self.fs * p['echoMs'] / 1000 + 0.5))
        for i in range(len(audio)):
            wet = float(self.buffer[(self.position - delay) % len(self.buffer)])
            dry = float(audio[i])
            self.buffer[self.position] = dry + wet * p['echoFeedback']
            audio[i] = dry * (1 - mix) + wet * mix
            self.position = (self.position + 1) % len(self.buffer)


MODULES = {'metallic': Metallic, 'distortion': Distortion, 'echo': Echo}


class EffectsChain:
    def __init__(self, fs):
        self.modules = {name: factory(fs) for name, factory in MODULES.items()}

    def process(self, audio, p):
        order = ('echo', 'metallic', 'distortion') if p['echoBeforeTexture'] else ('metallic', 'distortion', 'echo')
        for name in order:
            self.modules[name].process(audio, p)
