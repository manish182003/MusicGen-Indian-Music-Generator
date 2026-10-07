import numpy as np


class AudioValidator:

    def validate(self, audio, sample_rate):

        errors = []

        if audio is None:
            errors.append("Audio is None")

        if sample_rate <= 0:
            errors.append("Invalid sample rate")

        if np.isnan(audio).any():
            errors.append("NaN values found")

        if np.isinf(audio).any():
            errors.append("Infinite values found")

        if len(audio) == 0:
            errors.append("Audio is empty")

        return errors