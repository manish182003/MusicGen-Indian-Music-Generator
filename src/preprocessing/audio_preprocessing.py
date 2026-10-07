from pathlib import Path
import hashlib

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
}


# =============================================================
# TRACK METADATA
# =============================================================

class MetadataPreprocessor:

    def __init__(
        self,
        raw_dir="data/raw",
        metadata_dir="data/metadata",
        processed_dir="data/processed",
        min_duration=60.0,
        max_duration=300.0,
    ):

        self.raw_dir = Path(raw_dir)

        self.metadata_dir = Path(
            metadata_dir
        )

        self.processed_dir = Path(
            processed_dir
        )

        self.min_duration = (
            float(min_duration)
        )

        self.max_duration = (
            float(max_duration)
        )

        self.raw_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.metadata_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.processed_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # =========================================================
    # FIND FILES
    # =========================================================

    def find_audio_files(self):

        files = []

        for path in self.raw_dir.rglob("*"):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):
                files.append(path)

        return sorted(files)

    # =========================================================
    # LOAD
    # =========================================================

    def load_audio(self, path):

        audio, sample_rate = sf.read(
            path,
            dtype="float32"
        )

        return audio, sample_rate

    # =========================================================
    # HASH
    # =========================================================

    def calculate_file_hash(self, path):

        sha256 = hashlib.sha256()

        with open(path, "rb") as file:

            for chunk in iter(
                lambda:
                file.read(1024 * 1024),
                b""
            ):

                sha256.update(chunk)

        return sha256.hexdigest()

    # =========================================================
    # ANALYZE
    # =========================================================

    def analyze_audio(
        self,
        audio,
        sample_rate
    ):

        if audio.ndim == 1:

            channels = 1

            num_samples = (
                audio.shape[0]
            )

        elif audio.ndim == 2:

            channels = (
                audio.shape[1]
            )

            num_samples = (
                audio.shape[0]
            )

        else:

            raise ValueError(
                f"Unexpected audio shape: "
                f"{audio.shape}"
            )

        duration = (
            num_samples /
            sample_rate
        )

        peak = float(
            np.max(
                np.abs(audio)
            )
        )

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(
                        audio.astype(
                            np.float64
                        )
                    )
                )
            )
        )

        samples_above_one = int(
            np.sum(audio > 1.0)
        )

        samples_below_minus_one = int(
            np.sum(audio < -1.0)
        )

        total_samples = audio.size

        if total_samples:

            overs_percentage = (
                (
                    samples_above_one
                    + samples_below_minus_one
                )
                / total_samples
                * 100
            )

        else:

            overs_percentage = 0.0

        return {

            "sample_rate":
                int(sample_rate),

            "channels":
                int(channels),

            "num_samples":
                int(num_samples),

            "duration":
                float(duration),

            "dtype":
                str(audio.dtype),

            "min_amplitude":
                float(np.min(audio)),

            "max_amplitude":
                float(np.max(audio)),

            "peak_amplitude":
                peak,

            "rms":
                rms,

            "samples_above_1":
                samples_above_one,

            "samples_below_minus_1":
                samples_below_minus_one,

            "overs_percentage":
                float(overs_percentage),
        }

    # =========================================================
    # VALIDATE
    # =========================================================

    def validate_audio(
        self,
        audio,
        sample_rate
    ):

        errors = []

        if sample_rate <= 0:

            errors.append(
                "Invalid sample rate"
            )

        if audio is None:

            errors.append(
                "Audio is None"
            )

            return errors

        if audio.size == 0:

            errors.append(
                "Audio is empty"
            )

            return errors

        if np.isnan(audio).any():

            errors.append(
                "NaN values found"
            )

        if np.isinf(audio).any():

            errors.append(
                "Infinite values found"
            )

        if audio.ndim not in [1, 2]:

            errors.append(
                f"Unexpected number of "
                f"dimensions: {audio.ndim}"
            )

        if audio.ndim in [1, 2]:

            duration = (
                audio.shape[0] /
                sample_rate
            )

            if (
                duration <
                self.min_duration
            ):

                errors.append(
                    f"Audio too short: "
                    f"{duration:.2f}s"
                )

            if (
                duration >
                self.max_duration
            ):

                errors.append(
                    f"Audio too long: "
                    f"{duration:.2f}s"
                )

        return errors

    # =========================================================
    # PROCESS ONE TRACK
    # =========================================================

    def process_file(self, path):

        try:

            audio, sample_rate = (
                self.load_audio(path)
            )

            info = self.analyze_audio(
                audio,
                sample_rate
            )

            errors = self.validate_audio(
                audio,
                sample_rate
            )

            file_hash = (
                self.calculate_file_hash(
                    path
                )
            )

            status = (
                "valid"
                if not errors
                else "invalid"
            )

            return {

                "audio_path":
                    path.as_posix(),

                "filename":
                    path.name,

                "status":
                    status,

                "validation_errors":
                    " | ".join(errors),

                "file_hash":
                    file_hash,

                **info,
            }

        except Exception as error:

            return {

                "audio_path":
                    path.as_posix(),

                "filename":
                    path.name,

                "status":
                    "error",

                "validation_errors":
                    str(error),

                "file_hash":
                    "",
            }

    # =========================================================
    # CREATE METADATA
    # =========================================================

    def create_metadata(self):

        files = (
            self.find_audio_files()
        )

        if not files:

            print(
                "No audio files found."
            )

            return pd.DataFrame()

        records = []

        for index, path in enumerate(
            files,
            start=1
        ):

            print(
                f"[{index}/{len(files)}] "
                f"{path.name}"
            )

            records.append(
                self.process_file(path)
            )

        df = pd.DataFrame(
            records
        )

        return df

    # =========================================================
    # SAVE
    # =========================================================

    def save_metadata(
        self,
        df,
        filename="tracks.csv"
    ):

        output_path = (
            self.metadata_dir /
            filename
        )

        df.to_csv(
            output_path,
            index=False
        )

        print(
            f"\nMetadata saved: "
            f"{output_path}"
        )

        return output_path


# =============================================================
# AUDIO PREPROCESSOR
# =============================================================

class AudioPreprocessor:

    def __init__(
        self,
        processed_dir="data/processed",
        target_sample_rate=32000,
        chunk_duration=30.0,
        normalize_peak=0.95,
    ):

        self.processed_dir = Path(
            processed_dir
        )

        self.target_sample_rate = (
            int(target_sample_rate)
        )

        self.chunk_duration = (
            float(chunk_duration)
        )

        self.normalize_peak = (
            float(normalize_peak)
        )

        self.train_dir = (
            self.processed_dir /
            "train"
        )

        self.validation_dir = (
            self.processed_dir /
            "validation"
        )

        self.test_dir = (
            self.processed_dir /
            "test"
        )

        for directory in [
            self.train_dir,
            self.validation_dir,
            self.test_dir,
        ]:

            directory.mkdir(
                parents=True,
                exist_ok=True
            )

    # =========================================================
    # LOAD
    # =========================================================

    def load_audio(self, path):

        audio, sample_rate = sf.read(
            path,
            dtype="float32"
        )

        return audio, sample_rate

    # =========================================================
    # RESAMPLE
    # =========================================================

    def resample_audio(
        self,
        audio,
        original_sr
    ):

        if (
            original_sr ==
            self.target_sample_rate
        ):

            return audio

        return librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=self.target_sample_rate,
            res_type="kaiser_fast",
            axis=0,
        )

    # =========================================================
    # MONO
    # =========================================================

    def convert_to_mono(
        self,
        audio
    ):

        if audio.ndim == 1:

            return audio

        if audio.ndim == 2:

            return np.mean(
                audio,
                axis=1
            )

        raise ValueError(
            f"Unsupported audio shape: "
            f"{audio.shape}"
        )

    # =========================================================
    # SAFE NORMALIZATION
    # =========================================================

    def normalize_audio(
        self,
        audio
    ):

        peak = np.max(
            np.abs(audio)
        )

        if peak == 0:

            return audio

        if (
            peak >
            self.normalize_peak
        ):

            audio = (
                audio *
                (
                    self.normalize_peak /
                    peak
                )
            )

        return audio

    # =========================================================
    # FULL PREPROCESSING
    # =========================================================

    def preprocess_audio(
        self,
        audio,
        original_sr
    ):

        audio = self.resample_audio(
            audio,
            original_sr
        )

        audio = self.convert_to_mono(
            audio
        )

        audio = self.normalize_audio(
            audio
        )

        return audio.astype(
            np.float32
        )

    # =========================================================
    # CHUNK
    # =========================================================

    def create_chunks(
        self,
        audio
    ):

        chunk_samples = int(
            self.target_sample_rate *
            self.chunk_duration
        )

        chunks = []

        for start in range(
            0,
            len(audio),
            chunk_samples
        ):

            end = (
                start +
                chunk_samples
            )

            chunk = audio[
                start:end
            ]

            # Drop incomplete final chunk
            if (
                len(chunk) ==
                chunk_samples
            ):

                chunks.append(chunk)

        return chunks

    # =========================================================
    # PROCESS DATASET
    # =========================================================

    def process_dataset(
        self,
        splits_csv="data/metadata/splits.csv",
        chunks_csv="data/metadata/chunks.csv",
        resume=True,
        force=False,
    ):

        split_df = pd.read_csv(
            splits_csv
        )

        if split_df.empty:

            raise ValueError(
                "splits.csv is empty."
            )

        required_columns = {
            "track_id",
            "audio_path",
            "split",
        }

        missing = (
            required_columns -
            set(split_df.columns)
        )

        if missing:

            raise ValueError(
                f"Missing columns in "
                f"splits.csv: {missing}"
            )

        chunks_csv = Path(
            chunks_csv
        )

        chunks_csv.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        existing_chunks = (
            pd.DataFrame()
        )

        processed_track_ids = set()

        if (
            resume
            and chunks_csv.exists()
        ):

            try:

                existing_chunks = (
                    pd.read_csv(
                        chunks_csv
                    )
                )

                if (
                    "track_id"
                    in existing_chunks.columns
                ):

                    processed_track_ids = set(
                        existing_chunks[
                            "track_id"
                        ]
                        .astype(str)
                    )

            except Exception:

                existing_chunks = (
                    pd.DataFrame()
                )

                processed_track_ids = set()

        new_chunk_metadata = []

        for _, row in split_df.iterrows():

            track_id = str(
                row["track_id"]
            )

            split = str(
                row["split"]
            )

            audio_path = Path(
                str(row["audio_path"])
            )

            if split not in {
                "train",
                "validation",
                "test",
            }:

                print(
                    f"Skipping {track_id}: "
                    f"split={split}"
                )

                continue

            if (
                resume
                and not force
                and track_id
                in processed_track_ids
            ):

                print(
                    f"SKIP {track_id}: "
                    f"already processed"
                )

                continue

            if not audio_path.exists():

                print(
                    f"WARNING: audio missing: "
                    f"{audio_path}"
                )

                continue

            print(
                f"\nProcessing "
                f"{track_id} | {split}"
            )

            audio, original_sr = (
                self.load_audio(
                    audio_path
                )
            )

            audio = (
                self.preprocess_audio(
                    audio,
                    original_sr
                )
            )

            chunks = (
                self.create_chunks(
                    audio
                )
            )

            print(
                f"Chunks created: "
                f"{len(chunks)}"
            )

            if split == "train":

                output_dir = (
                    self.train_dir
                )

            elif split == "validation":

                output_dir = (
                    self.validation_dir
                )

            else:

                output_dir = (
                    self.test_dir
                )

            # Remove old chunks if forcing
            if force:

                for old_file in (
                    output_dir.glob(
                        f"{track_id}_chunk_*.wav"
                    )
                ):

                    old_file.unlink(
                        missing_ok=True
                    )

            for chunk_index, chunk in enumerate(
                chunks
            ):

                chunk_id = (
                    f"{track_id}_"
                    f"chunk_"
                    f"{chunk_index:03d}"
                )

                filename = (
                    f"{chunk_id}.wav"
                )

                output_path = (
                    output_dir /
                    filename
                )

                sf.write(
                    output_path,
                    chunk,
                    self.target_sample_rate
                )

                new_chunk_metadata.append({

                    "chunk_id":
                        chunk_id,

                    "track_id":
                        track_id,

                    "split":
                        split,

                    "audio_path":
                        output_path.as_posix(),

                    "duration":
                        self.chunk_duration,

                    "sample_rate":
                        self.target_sample_rate,

                    "channels":
                        1,
                })

        new_df = pd.DataFrame(
            new_chunk_metadata
        )

        if force:

            final_df = new_df

        else:

            frames = []

            if not existing_chunks.empty:

                frames.append(
                    existing_chunks
                )

            if not new_df.empty:

                frames.append(
                    new_df
                )

            if frames:

                final_df = pd.concat(
                    frames,
                    ignore_index=True
                )

            else:

                final_df = pd.DataFrame()

        if not final_df.empty:

            final_df = (
                final_df
                .drop_duplicates(
                    subset=["chunk_id"],
                    keep="last"
                )
                .sort_values(
                    ["split", "track_id", "chunk_id"]
                )
                .reset_index(drop=True)
            )

        final_df.to_csv(
            chunks_csv,
            index=False
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "CHUNKING COMPLETE"
        )

        print(
            "=" * 60
        )

        print(
            f"Total chunks: "
            f"{len(final_df)}"
        )

        if not final_df.empty:

            print(
                "\nChunks by split:"
            )

            print(
                final_df[
                    "split"
                ].value_counts()
            )

        print(
            f"\nSaved: "
            f"{chunks_csv}"
        )

        return final_df


# =============================================================
# DIRECT EXECUTION
# =============================================================

if __name__ == "__main__":

    metadata_processor = (
        MetadataPreprocessor(
            raw_dir="data/raw",
            metadata_dir="data/metadata",
            processed_dir="data/processed",
            min_duration=60,
            max_duration=300,
        )
    )

    df = (
        metadata_processor
        .create_metadata()
    )

    if df.empty:

        raise SystemExit(
            "No audio files found."
        )

    metadata_processor.save_metadata(
        df
    )

    preprocessor = (
        AudioPreprocessor(
            processed_dir="data/processed",
            target_sample_rate=32000,
            chunk_duration=30,
            normalize_peak=0.95,
        )
    )

    preprocessor.process_dataset(
        splits_csv="data/metadata/splits.csv"
    )