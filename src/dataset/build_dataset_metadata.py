from pathlib import Path

import pandas as pd


def build_dataset_metadata(
    chunks_path="data/metadata/chunks.csv",
    captions_path="data/metadata/chunk_captions.csv",
    output_path="data/metadata/dataset.csv",
):

    chunks_path = Path(
        chunks_path
    )

    captions_path = Path(
        captions_path
    )

    output_path = Path(
        output_path
    )

    if not chunks_path.exists():

        raise FileNotFoundError(
            f"Missing chunks.csv: "
            f"{chunks_path}"
        )

    if not captions_path.exists():

        raise FileNotFoundError(
            f"Missing chunk_captions.csv: "
            f"{captions_path}"
        )

    chunks_df = pd.read_csv(
        chunks_path
    )

    captions_df = pd.read_csv(
        captions_path
    )

    if chunks_df.empty:

        raise ValueError(
            "chunks.csv is empty."
        )

    if captions_df.empty:

        raise ValueError(
            "chunk_captions.csv is empty."
        )

    # =========================================================
    # VALIDATE CHUNK IDS
    # =========================================================

    if (
        chunks_df["chunk_id"]
        .duplicated()
        .any()
    ):

        duplicates = (
            chunks_df.loc[
                chunks_df[
                    "chunk_id"
                ].duplicated(),
                "chunk_id"
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate chunk IDs found: "
            f"{duplicates[:10]}"
        )

    if (
        captions_df["chunk_id"]
        .duplicated()
        .any()
    ):

        duplicates = (
            captions_df.loc[
                captions_df[
                    "chunk_id"
                ].duplicated(),
                "chunk_id"
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate caption chunk IDs "
            f"found: {duplicates[:10]}"
        )

    # =========================================================
    # CAPTION COLUMNS
    # =========================================================

    caption_columns = [

        "chunk_id",

        "track_id",

        "split",

        "status",

        "caption_source",

        "genre",

        "style",

        "mood",

        "instruments",

        "tempo_bpm",

        "rhythm",

        "structure",

        "full_description",

        "musicgen_prompt",

        "error",
    ]

    missing_caption_columns = (
        set(caption_columns)
        -
        set(captions_df.columns)
    )

    if missing_caption_columns:

        raise ValueError(
            "Missing columns in "
            "chunk_captions.csv: "
            f"{missing_caption_columns}"
        )

    captions_df = (
        captions_df[
            caption_columns
        ]
        .copy()
    )

    # =========================================================
    # MERGE
    # =========================================================

    dataset_df = chunks_df.merge(
        captions_df,
        on="chunk_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_caption"
        )
    )

    # =========================================================
    # VALIDATE
    # =========================================================

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
    )

    failed_captions = (
        dataset_df[
            "status"
        ]
        .astype(str)
        .str.lower()
        .eq("failed")
    )

    missing_audio = (
        ~dataset_df[
            "audio_path"
        ].apply(
            lambda path:
            Path(str(path)).exists()
        )
    )

    # =========================================================
    # REPORT
    # =========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL MUSICGEN DATASET METADATA"
    )

    print(
        "=" * 70
    )

    print(
        f"Total chunks: "
        f"{len(dataset_df)}"
    )

    print(
        f"Missing prompts: "
        f"{int(missing_prompts.sum())}"
    )

    print(
        f"Failed captions: "
        f"{int(failed_captions.sum())}"
    )

    print(
        f"Missing audio files: "
        f"{int(missing_audio.sum())}"
    )

    print(
        "\nChunks by split:"
    )

    print(
        dataset_df[
            "split"
        ].value_counts()
    )

    if (
        "caption_source"
        in dataset_df.columns
    ):

        print(
            "\nCaption source:"
        )

        print(
            dataset_df[
                "caption_source"
            ].value_counts()
        )

    # =========================================================
    # SAVE
    # =========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved dataset metadata:"
        f"\n{output_path}"
    )

    return dataset_df


if __name__ == "__main__":

    build_dataset_metadata()