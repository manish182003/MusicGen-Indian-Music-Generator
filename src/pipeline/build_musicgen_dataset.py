import argparse
import json
import sys
from pathlib import Path


# =============================================================
# PROJECT PATH
# =============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

SRC_ROOT = (
    PROJECT_ROOT /
    "src"
)

if str(SRC_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_ROOT)
    )


# =============================================================
# IMPORT PIPELINE COMPONENTS
# =============================================================

from acquisition.scraper import (
    AudioScraper
)

from preprocessing.audio_preprocessing import (
    MetadataPreprocessor,
    AudioPreprocessor,
)

from preprocessing.chunk_audio_caption import (
    run_caption_pipeline,
)

from dataset.create_split import (
    create_track_split,
)

from dataset.build_dataset_metadata import (
    build_dataset_metadata,
)


# =============================================================
# CONFIG
# =============================================================

DEFAULT_CONFIG = (
    PROJECT_ROOT /
    "config" /
    "musicgen_pipeline.json"
)


# =============================================================
# LOAD CONFIG
# =============================================================

def load_config(
    config_path
):

    config_path = Path(
        config_path
    )

    if not config_path.exists():

        raise FileNotFoundError(
            f"Config not found: "
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# =============================================================
# PATHS
# =============================================================

def make_paths(config):

    paths = config[
        "paths"
    ]

    return {

        "raw":
            PROJECT_ROOT /
            paths["raw_dir"],

        "metadata":
            PROJECT_ROOT /
            paths["metadata_dir"],

        "processed":
            PROJECT_ROOT /
            paths["processed_dir"],
    }


# =============================================================
# ACQUISITION
# =============================================================

def run_acquisition(
    config,
    paths
):

    acquisition = (
        config[
            "acquisition"
        ]
    )

    if not acquisition.get(
        "enabled",
        False
    ):

        print(
            "\n[ACQUISITION]"
        )

        print(
            "Disabled."
        )

        return

    print(
        "\n" + "=" * 70
    )

    print(
        "STAGE 1: AUDIO ACQUISITION"
    )

    print(
        "=" * 70
    )

    scraper = AudioScraper(

        output_dir=str(
            paths["raw"]
        ),

        preferred_codec="wav",

        min_duration_sec=(
            config[
                "audio"
            ][
                "min_duration_seconds"
            ]
        ),

        max_duration_sec=(
            config[
                "audio"
            ][
                "max_duration_seconds"
            ]
        ),
    )

    results = (
        scraper.scrape_diverse_dataset(

            acquisition[
                "queries"
            ],

            tracks_per_category=(
                acquisition[
                    "tracks_per_category"
                ]
            )
        )
    )

    total_downloaded = sum(
        len(files)
        for files
        in results.values()
    )

    print(
        f"\nDownloaded this run: "
        f"{total_downloaded}"
    )


# =============================================================
# METADATA
# =============================================================

def run_metadata(
    config,
    paths
):

    print(
        "\n" + "=" * 70
    )

    print(
        "STAGE 2: TRACK METADATA"
    )

    print(
        "=" * 70
    )

    audio_config = (
        config[
            "audio"
        ]
    )

    processor = (
        MetadataPreprocessor(

            raw_dir=str(
                paths["raw"]
            ),

            metadata_dir=str(
                paths["metadata"]
            ),

            processed_dir=str(
                paths["processed"]
            ),

            min_duration=(
                audio_config[
                    "min_duration_seconds"
                ]
            ),

            max_duration=(
                audio_config[
                    "max_duration_seconds"
                ]
            ),
        )
    )

    df = (
        processor
        .create_metadata()
    )

    if df.empty:

        raise RuntimeError(
            "No audio files found."
        )

    processor.save_metadata(
        df
    )

    valid = int(
        (
            df["status"]
            == "valid"
        ).sum()
    )

    invalid = int(
        (
            df["status"]
            == "invalid"
        ).sum()
    )

    errors = int(
        (
            df["status"]
            == "error"
        ).sum()
    )

    print(
        f"\nTotal: {len(df)}"
    )

    print(
        f"Valid: {valid}"
    )

    print(
        f"Invalid: {invalid}"
    )

    print(
        f"Errors: {errors}"
    )

    return df


# =============================================================
# SPLIT
# =============================================================

def run_split(
    config,
    paths
):

    print(
        "\n" + "=" * 70
    )

    print(
        "STAGE 3: TRACK-LEVEL SPLIT"
    )

    print(
        "=" * 70
    )

    split_config = (
        config[
            "split"
        ]
    )

    return create_track_split(

        metadata_path=str(
            paths["metadata"] /
            "tracks.csv"
        ),

        output_path=str(
            paths["metadata"] /
            "splits.csv"
        ),

        train_ratio=(
            split_config[
                "train_ratio"
            ]
        ),

        validation_ratio=(
            split_config[
                "validation_ratio"
            ]
        ),

        test_ratio=(
            split_config[
                "test_ratio"
            ]
        ),
    )


# =============================================================
# AUDIO PREPROCESSING
# =============================================================

def run_preprocessing(
    config,
    paths,
    force=False
):

    print(
        "\n" + "=" * 70
    )

    print(
        "STAGE 4: AUDIO PREPROCESSING"
    )

    print(
        "=" * 70
    )

    audio_config = (
        config[
            "audio"
        ]
    )

    processor = (
        AudioPreprocessor(

            processed_dir=str(
                paths["processed"]
            ),

            target_sample_rate=(
                audio_config[
                    "target_sample_rate"
                ]
            ),

            chunk_duration=(
                audio_config[
                    "chunk_duration_seconds"
                ]
            ),

            normalize_peak=(
                audio_config[
                    "normalize_peak"
                ]
            ),
        )
    )

    return processor.process_dataset(

        splits_csv=str(
            paths["metadata"] /
            "splits.csv"
        ),

        chunks_csv=str(
            paths["metadata"] /
            "chunks.csv"
        ),

        resume=not force,

        force=force,
    )


# =============================================================
# CAPTIONING
# =============================================================

def run_captioning(
    config,
    paths
):

    print(
        "\n" + "=" * 70
    )

    print(
        "STAGE 5: CHUNK-LEVEL CAPTIONING"
    )

    print(
        "=" * 70
    )

    caption_config = (
        config[
            "captioning"
        ]
    )

    return run_caption_pipeline(

        chunks_csv=str(
            paths["metadata"] /
            "chunks.csv"
        ),

        output_csv=str(
            paths["metadata"] /
            "chunk_captions.csv"
        ),

        project_root=str(
            PROJECT_ROOT
        )

        # gemini_model=(
        #     caption_config[
        #         "gemini_model"
        #     ]
        # ),

        # groq_model=(
        #     caption_config[
        #         "groq_model"
        #     ]
        # ),

        # groq_max_completion_tokens=(
        #     caption_config[
        #         "groq_max_completion_tokens"
        #     ]
        # ),

        # delay_seconds=(
        #     caption_config[
        #         "request_delay_seconds"
        #     ]
        # ),

        # enable_gemini=(
        #     caption_config[
        #         "enable_gemini"
        #     ]
        # ),

        # enable_groq=(
        #     caption_config[
        #         "enable_groq_fallback"
        #     ]
        # ),
    )


# =============================================================
# FINAL DATASET
# =============================================================

def run_dataset_build(
    paths
):

    print(
        "\n" + "=" * 70
    )

    print(
        "STAGE 6: BUILD FINAL DATASET"
    )

    print(
        "=" * 70
    )

    return build_dataset_metadata(

        chunks_path=str(
            paths["metadata"] /
            "chunks.csv"
        ),

        captions_path=str(
            paths["metadata"] /
            "chunk_captions.csv"
        ),

        output_path=str(
            paths["metadata"] /
            "dataset.csv"
        ),
    )


# =============================================================
# FINAL VALIDATION
# =============================================================

def validate_dataset(
    dataset_df,
    paths
):

    print(
        "\n" + "=" * 70
    )

    print(
        "STAGE 7: FINAL VALIDATION"
    )

    print(
        "=" * 70
    )

    errors = []

    # ---------------------------------------------------------
    # Empty dataset
    # ---------------------------------------------------------

    if dataset_df.empty:

        errors.append(
            "Final dataset is empty."
        )

    # ---------------------------------------------------------
    # Duplicate chunks
    # ---------------------------------------------------------

    duplicate_chunks = (
        dataset_df[
            "chunk_id"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_chunks:

        errors.append(
            f"Duplicate chunks: "
            f"{duplicate_chunks}"
        )

    # ---------------------------------------------------------
    # Missing prompts
    # ---------------------------------------------------------

    missing_prompts = (
        dataset_df[
            "musicgen_prompt"
        ]
        .isna()
        |
        (
            dataset_df[
                "musicgen_prompt"
            ]
            .astype(str)
            .str.strip()
            .eq("")
        )
    ).sum()

    if missing_prompts:

        errors.append(
            f"Missing prompts: "
            f"{missing_prompts}"
        )

    # ---------------------------------------------------------
    # Failed captions
    # ---------------------------------------------------------

    if "status" in dataset_df.columns:

        failed = (
            dataset_df[
                "status"
            ]
            .astype(str)
            .str.lower()
            .eq("failed")
            .sum()
        )

        if failed:

            errors.append(
                f"Failed captions: "
                f"{failed}"
            )

    # ---------------------------------------------------------
    # Track leakage
    # ---------------------------------------------------------

    track_split_counts = (
        dataset_df
        .groupby(
            "track_id"
        )[
            "split"
        ]
        .nunique()
    )

    leaked_tracks = (
        track_split_counts[
            track_split_counts > 1
        ]
    )

    if not leaked_tracks.empty:

        errors.append(
            "Track leakage detected: "
            +
            ", ".join(
                leaked_tracks.index
                .astype(str)
                .tolist()
            )
        )

    # ---------------------------------------------------------
    # Audio paths
    # ---------------------------------------------------------

    missing_audio = 0

    for path in dataset_df[
        "audio_path"
    ]:

        if not Path(
            str(path)
        ).exists():

            missing_audio += 1

    if missing_audio:

        errors.append(
            f"Missing audio files: "
            f"{missing_audio}"
        )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print(
        f"Rows: "
        f"{len(dataset_df)}"
    )

    print(
        f"Train: "
        f"{(
            dataset_df['split']
            == 'train'
        ).sum()}"
    )

    print(
        f"Validation: "
        f"{(
            dataset_df['split']
            == 'validation'
        ).sum()}"
    )

    print(
        f"Test: "
        f"{(
            dataset_df['split']
            == 'test'
        ).sum()}"
    )

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    if errors:

        print(
            "\nVALIDATION FAILED"
        )

        for error in errors:

            print(
                f"  ❌ {error}"
            )

        raise RuntimeError(
            "Dataset validation failed."
        )

    print(
        "\nVALIDATION PASSED"
    )

    print(
        "Dataset is ready for Kaggle."
    )


# =============================================================
# MAIN
# =============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build scalable MusicGen dataset"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(
            DEFAULT_CONFIG
        )
    )

    parser.add_argument(
        "--acquire",
        action="store_true",
        help=(
            "Enable audio acquisition "
            "for this run."
        )
    )

    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help=(
            "Reprocess existing audio "
            "chunks."
        )
    )

    parser.add_argument(
        "--skip-captioning",
        action="store_true",
        help=(
            "Skip AI captioning."
        )
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    paths = make_paths(
        config
    )

    # ---------------------------------------------------------
    # Acquisition
    # ---------------------------------------------------------

    if args.acquire:

        config[
            "acquisition"
        ][
            "enabled"
        ] = True

    run_acquisition(
        config,
        paths
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    run_metadata(
        config,
        paths
    )

    # ---------------------------------------------------------
    # Split
    # ---------------------------------------------------------

    run_split(
        config,
        paths
    )

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    run_preprocessing(
        config,
        paths,
        force=args.force_preprocess
    )

    # ---------------------------------------------------------
    # Captioning
    # ---------------------------------------------------------

    if not args.skip_captioning:

        run_captioning(
            config,
            paths
        )

    else:

        print(
            "\nCaptioning skipped."
        )

    # ---------------------------------------------------------
    # Final dataset
    # ---------------------------------------------------------

    dataset_df = (
        run_dataset_build(
            paths
        )
    )

    if "status" in dataset_df.columns:

        failed_captions = int(
            dataset_df[
                "status"
            ]
            .astype(str)
            .str.lower()
            .eq("failed")
            .sum()
        )

    else:

        failed_captions = 0


    missing_prompts = int(
        (
            dataset_df[
                "musicgen_prompt"
            ].isna()
            |
            (
                dataset_df[
                    "musicgen_prompt"
                ]
                .astype(str)
                .str.strip()
                .eq("")
            )
        ).sum()
    )


    if (
        failed_captions > 0
        or missing_prompts > 0
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            "DATASET BUILD INCOMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            f"Total chunks: "
            f"{len(dataset_df)}"
        )

        print(
            f"Successful captions: "
            f"{len(dataset_df) - failed_captions}"
        )

        print(
            f"Remaining failed captions: "
            f"{failed_captions}"
        )

        print(
            f"Missing prompts: "
            f"{missing_prompts}"
        )

        print(
            "\nProgress has been saved."
        )

        print(
            "Run the pipeline again after "
            "the API quota resets."
        )

        return


    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    validate_dataset(
        dataset_df,
        paths
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MUSICGEN DATASET PIPELINE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nFinal files:"
    )

    print(
        f"Tracks: "
        f"{paths['metadata'] / 'tracks.csv'}"
    )

    print(
        f"Splits: "
        f"{paths['metadata'] / 'splits.csv'}"
    )

    print(
        f"Chunks: "
        f"{paths['metadata'] / 'chunks.csv'}"
    )

    print(
        f"Captions: "
        f"{paths['metadata'] / 'chunk_captions.csv'}"
    )

    print(
        f"Dataset: "
        f"{paths['metadata'] / 'dataset.csv'}"
    )


if __name__ == "__main__":

    main()