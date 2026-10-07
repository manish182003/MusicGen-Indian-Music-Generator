import soundfile as sf

class Audioloader:
    
    def load(self,path):
        audio,sample_rate=sf.read(path)
        return audio,sample_rate

# loader=Audioloader()
# audio,sr=loader.load("data/raw/song1.wav")
# print("Shape:", audio.shape)
# print("Sample rate:", sr)