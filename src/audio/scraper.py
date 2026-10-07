import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List
import yt_dlp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AudioScraper:
    """
    Scrapes and extracts audio streams with auto-resuming sequential numbering
    (song1.wav, song2.wav, etc.) and pre-download duration filtering.
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
        self.current_index = self._get_next_index()

    def _get_next_index(self) -> int:
        """
        Scans output_dir for existing files matching 'song<number>.<ext>'
        and returns the next unused integer index.
        """
        pattern = re.compile(rf"^song(\d+)\.{self.preferred_codec}$")
        existing_indices = []

        for file in self.output_dir.iterdir():
            match = pattern.match(file.name)
            if match:
                existing_indices.append(int(match.group(1)))

        if existing_indices:
            next_idx = max(existing_indices) + 1
            logging.info(f"Found existing files. Resuming numbering from: song{next_idx}.{self.preferred_codec}")
            return next_idx
        return 1

    def _get_ydl_options(self, temp_filename_template: str) -> Dict:
        """Returns yt-dlp configuration with pre-download duration filter."""

        def duration_filter(info_dict, *, incomplete):
            duration = info_dict.get("duration")
            if duration is None:
                return None
            if duration < self.min_duration_sec:
                return f"Duration {duration}s is shorter than min {self.min_duration_sec}s"
            if duration > self.max_duration_sec:
                return f"Duration {duration}s exceeds max {self.max_duration_sec}s"
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
            "quiet": False,
            "no_warnings": True,
            "ignoreerrors": True,
        }

    def scrape_from_search(self, search_query: str, max_results: int = 5) -> List[Path]:
        """
        Searches, filters by duration, downloads, and renames sequentially.
        """
        saved_files: List[Path] = []
        search_pool = max_results * 3
        query_url = f"ytsearch{search_pool}:{search_query}"

        # Temporary download path pattern using video ID to avoid conflicts
        temp_template = str(self.output_dir / "temp_%(id)s.%(ext)s")
        ydl_opts = self._get_ydl_options(temp_template)

        logging.info(f"Query: '{search_query}' | Target: {max_results} files (under {self.max_duration_sec}s)")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_results = ydl.extract_info(query_url, download=True)
            if not search_results or "entries" not in search_results:
                return saved_files

            for entry in search_results["entries"]:
                if len(saved_files) >= max_results:
                    break
                if entry is None:
                    continue

                video_id = entry.get("id")
                temp_audio_path = self.output_dir / f"temp_{video_id}.{self.preferred_codec}"

                # If download passed duration filtering and was extracted
                if temp_audio_path.exists():
                    target_filename = f"song{self.current_index}.{self.preferred_codec}"
                    final_path = self.output_dir / target_filename

                    # Atomic rename from temp file to sequential name
                    shutil.move(str(temp_audio_path), str(final_path))
                    saved_files.append(final_path)
                    logging.info(f"Saved: {final_path.name} (Duration: {entry.get('duration')}s)")

                    self.current_index += 1

        return saved_files

    def scrape_diverse_dataset(
        self, category_queries: Dict[str, str], tracks_per_category: int = 2
    ) -> Dict[str, List[Path]]:
        """
        Executes searches across multiple musical categories, keeping a unified numbering sequence.
        """
        dataset_summary: Dict[str, List[Path]] = {}

        for category, query in category_queries.items():
            logging.info(f"\n--- Scraping Category: {category.upper()} ---")
            downloaded = self.scrape_from_search(
                search_query=query,
                max_results=tracks_per_category,
            )
            dataset_summary[category] = downloaded

        return dataset_summary


if __name__ == "__main__":
    scraper = AudioScraper(
        output_dir="data/raw",
        preferred_codec="wav",
        min_duration_sec=60,   # Minimum 1 minute
        max_duration_sec=300,  # Maximum 5 minutes
    )

    diversity_queries = {
        "sitar": "indian classical sitar solo instrumental -vocal",
        "flute": "meditative bansuri bamboo flute instrumental indian classical",
        "sarod": "sarod instrumental fast drut gat classical indian",
        "santoor": "kashmiri santoor instrumental acoustic traditional",
    }

    results = scraper.scrape_diverse_dataset(diversity_queries, tracks_per_category=2)
    print(f"\nFinished run. Next index will start at: song{scraper.current_index}.wav")