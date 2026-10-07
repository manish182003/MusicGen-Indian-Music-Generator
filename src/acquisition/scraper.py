import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Set

import soundfile as sf
import yt_dlp


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class AudioScraper:
    """
    Robust audio acquisition helper.

    Responsibilities:
        - Search YouTube for audio tracks matching category queries
        - Filter by min/max duration before download
        - Convert and validate audio into clean 32kHz 16-bit WAV format using FFmpeg
        - Resume sequential track indexing without filename collisions or overwrites
        - Maintain download history to prevent duplicate downloads
        - Save verified audio into data/raw/
    """

    def __init__(
        self,
        output_dir: str = "data/raw",
        preferred_codec: str = "wav",
        min_duration_sec: int = 60,
        max_duration_sec: int = 300,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.preferred_codec = preferred_codec
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec

        self.history_file = self.output_dir / ".download_history.json"
        self.downloaded_ids: Set[str] = self._load_history()

        self.current_index = self._get_next_index()

    # =========================================================
    # HISTORY MANAGEMENT
    # =========================================================

    def _load_history(self) -> Set[str]:
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception as e:
                logging.warning(f"Failed to load download history: {e}")
        return set()

    def _save_history(self):
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(sorted(list(self.downloaded_ids)), f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save download history: {e}")

    # =========================================================
    # RESUME & COLLISION-PROOF INDEXING
    # =========================================================

    def _get_next_index(self) -> int:
        existing_indices = []
        pattern = re.compile(r"^(?:song|track)_?(\d+)\.", re.IGNORECASE)

        for file in self.output_dir.iterdir():
            if not file.is_file():
                continue
            match = pattern.match(file.name)
            if match:
                existing_indices.append(int(match.group(1)))

        if existing_indices:
            next_idx = max(existing_indices) + 1
            logging.info(f"Existing raw dataset detected. Resuming index at song{next_idx}.{self.preferred_codec}")
            return next_idx

        return 1

    def _get_unique_target_path(self) -> Path:
        while True:
            target_filename = f"song{self.current_index}.{self.preferred_codec}"
            target_path = self.output_dir / target_filename
            if not target_path.exists():
                return target_path
            self.current_index += 1

    # =========================================================
    # AUDIO CONVERSION & VALIDATION
    # =========================================================

    def _convert_to_valid_wav(self, input_path: Path, output_path: Path) -> bool:
        """
        Converts audio to clean PCM WAV using FFmpeg and validates with SoundFile.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "32000",
            "-ac",
            "2",
            str(output_path),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode == 0 and output_path.exists():
                # Verify valid audio
                info = sf.info(output_path)
                if info.duration >= self.min_duration_sec and info.duration <= (self.max_duration_sec + 30):
                    return True
                else:
                    logging.warning(f"Converted audio duration ({info.duration:.1f}s) outside expected range.")
                    output_path.unlink(missing_ok=True)
                    return False
            else:
                logging.warning(f"FFmpeg conversion failed: {res.stderr[:200]}")
                output_path.unlink(missing_ok=True)
                return False
        except Exception as exc:
            logging.warning(f"Error during audio conversion: {exc}")
            output_path.unlink(missing_ok=True)
            return False

    # =========================================================
    # YT-DLP OPTIONS
    # =========================================================

    def _get_ydl_options(self, temp_filename_template: str) -> Dict:
        def duration_filter(info_dict, *, incomplete):
            duration = info_dict.get("duration")
            if duration is None:
                return None
            if duration < self.min_duration_sec:
                return f"Duration {duration}s < min {self.min_duration_sec}s"
            if duration > self.max_duration_sec:
                return f"Duration {duration}s > max {self.max_duration_sec}s"
            return None

        return {
            "format": "bestaudio/best",
            "outtmpl": temp_filename_template,
            "match_filter": duration_filter,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": self.preferred_codec,
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
        }

    # =========================================================
    # SEARCH & DOWNLOAD
    # =========================================================

    def scrape_from_search(self, search_query: str, max_results: int = 5) -> List[Path]:
        saved_files = []
        search_pool = max(max_results * 4, 15)
        query_url = f"ytsearch{search_pool}:{search_query}"

        temp_template = str(self.output_dir / "temp_%(id)s.%(ext)s")
        ydl_opts = self._get_ydl_options(temp_template)

        logging.info(f"Downloading for query: '{search_query}' (Target: {max_results} files)")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                search_results = ydl.extract_info(query_url, download=True)
            except Exception as exc:
                logging.error(f"yt_dlp search failed for '{search_query}': {exc}")
                return saved_files

            if not search_results or "entries" not in search_results:
                return saved_files

            for entry in search_results["entries"]:
                if len(saved_files) >= max_results:
                    break
                if entry is None:
                    continue

                video_id = entry.get("id")
                if not video_id or video_id in self.downloaded_ids:
                    continue

                # Locate downloaded temp file
                candidates = list(self.output_dir.glob(f"temp_{video_id}.*"))
                if not candidates:
                    continue

                temp_file = candidates[0]
                final_path = self._get_unique_target_path()

                # Convert & validate audio
                success = self._convert_to_valid_wav(temp_file, final_path)
                temp_file.unlink(missing_ok=True)

                if success:
                    saved_files.append(final_path)
                    self.downloaded_ids.add(video_id)
                    self._save_history()
                    logging.info(f"Saved [{len(saved_files)}/{max_results}]: {final_path.name} ({sf.info(final_path).duration:.1f}s)")
                    self.current_index += 1
                else:
                    logging.warning(f"Discarded invalid/corrupt audio for video {video_id}")

        self._cleanup_temp_files()
        return saved_files

    # =========================================================
    # DIVERSE DATASET ACQUISITION
    # =========================================================

    def scrape_diverse_dataset(
        self, category_queries: Dict[str, str], tracks_per_category: int = 5
    ) -> Dict[str, List[Path]]:
        dataset_summary = {}
        for category, query in category_queries.items():
            logging.info("\n" + "=" * 60)
            logging.info(f"CATEGORY: {category.upper()}")
            logging.info("=" * 60)

            downloaded = self.scrape_from_search(search_query=query, max_results=tracks_per_category)
            dataset_summary[category] = downloaded
            logging.info(f"Category '{category}': {len(downloaded)} new tracks acquired.")

        self._cleanup_temp_files()
        return dataset_summary

    # =========================================================
    # CLEANUP
    # =========================================================

    def _cleanup_temp_files(self):
        for temp_file in self.output_dir.glob("temp_*"):
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass


# =============================================================
# DIRECT EXECUTION TEST
# =============================================================

if __name__ == "__main__":
    scraper = AudioScraper(
        output_dir="data/raw",
        preferred_codec="wav",
        min_duration_sec=60,
        max_duration_sec=300,
    )

    queries = {
        "sitar": "indian classical sitar solo instrumental -vocal",
        "flute": "meditative bansuri bamboo flute instrumental indian classical",
        "sarod": "sarod instrumental fast drut gat classical indian",
        "santoor": "kashmiri santoor instrumental acoustic traditional",
        "tabla": "indian classical tabla instrumental percussion solo",
        "veena": "indian classical veena instrumental",
        "shehnai": "indian classical shehnai instrumental",
        "bansuri": "indian classical bansuri flute instrumental",
        "sarangi": "indian classical sarangi solo instrumental",
        "violin": "indian classical violin instrumental solo",
        "harmonium": "indian classical harmonium solo instrumental",
        "guitar": "indian slide guitar mohan veena instrumental",
    }

    results = scraper.scrape_diverse_dataset(queries, tracks_per_category=5)
    print("\nFinished acquisition test.")
    print(f"Next index: song{scraper.current_index}.wav")