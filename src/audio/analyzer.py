import numpy as np

class AudioAnalyzer:
    def analyze(self,audio,sample_rate):

        if audio.ndim==1:
            channels=1
            num_samples=len(audio)
        else:
            num_samples=audio.shape[0]
            channels=audio.shape[1]

        duration=num_samples/sample_rate

        return {
            "sample_rate":sample_rate,
            "channels":channels,
            "num_samples":num_samples,
            "duration":duration,
            "dtype":str(audio.dtype),
            "min":float(np.min(audio)),
            "max":float(np.max(audio))
        }