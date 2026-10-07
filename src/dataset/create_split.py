from pathlib import Path
import hashlib
import re

import pandas as pd


def generate_new_track_id(
    used_ids
):

    numbers = []

    pattern = re.compile(
        r"^track_(\d+)$"
    )

    for track_id in used_ids:

        match = pattern.match(
            str(track_id)
        )

        if match:

            numbers.append(
                int(match.group(1))
            )

    next_number = (
        max(numbers) + 1
        if numbers
        else 1
    )

    return (
        f"track_{next_number:03d}"
    )


def deterministic_bucket(
    track_id
):

    digest = hashlib.sha256(
        str(track_id)
        .encode("utf-8")
    ).hexdigest()

    number = int(
        digest[:8],
        16
    )

    return (
        number /
        0xFFFFFFFF
    )



def ensure_track_ids(
    tracks_df,
    previous_splits
):

    df = tracks_df.copy()

    previous_map = {}

    if (
        previous_splits is not None
        and not previous_splits.empty
        and "filename"
        in previous_splits.columns
        and "track_id"
        in previous_splits.columns
    ):

        previous_map = dict(
            zip(
                previous_splits[
                    "filename"
                ].astype(str),

                previous_splits[
                    "track_id"
                ].astype(str)
            )
        )

    existing_ids = set(
        previous_map.values()
    )

    if (
        "track_id"
        not in df.columns
    ):

        df["track_id"] = ""

    for index, row in df.iterrows():

        filename = str(
            row["filename"]
        )

        existing_id = (
            previous_map.get(
                filename
            )
        )

        if existing_id:

            df.at[
                index,
                "track_id"
            ] = existing_id

            continue

        if not str(
            row["track_id"]
        ).strip():

            new_id = (
                generate_new_track_id(
                    existing_ids
                )
            )

            df.at[
                index,
                "track_id"
            ] = new_id

            existing_ids.add(
                new_id
            )

    return df



def assign_split_for_new_track(
    track_id,
    train_ratio,
    validation_ratio,
    test_ratio
):

    total = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if abs(total - 1.0) > 1e-6:

        raise ValueError(
            "Split ratios must sum to 1.0"
        )

    bucket = deterministic_bucket(
        track_id
    )

    if bucket < train_ratio:

        return "train"

    if bucket < (
        train_ratio +
        validation_ratio
    ):

        return "validation"

    return "test"


def create_track_split(
    metadata_path="data/metadata/tracks.csv",
    output_path="data/metadata/splits.csv",
    train_ratio=0.80,
    validation_ratio=0.10,
    test_ratio=0.10,
):

    metadata_path = Path(
        metadata_path
    )

    output_path = Path(
        output_path
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Metadata not found: "
            f"{metadata_path}"
        )

    tracks_df = pd.read_csv(
        metadata_path
    )

    if tracks_df.empty:

        raise ValueError(
            "tracks.csv is empty."
        )

    required = {
        "filename",
        "status",
        "audio_path",
    }

    missing = (
        required -
        set(tracks_df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    previous_splits = (
        pd.DataFrame()
    )

    if output_path.exists():

        try:

            previous_splits = (
                pd.read_csv(
                    output_path
                )
            )

        except Exception:

            previous_splits = (
                pd.DataFrame()
            )

    df = ensure_track_ids(
        tracks_df,
        previous_splits
    )


    previous_split_map = {}

    if (
        not previous_splits.empty
        and "track_id"
        in previous_splits.columns
        and "split"
        in previous_splits.columns
    ):

        previous_split_map = dict(
            zip(
                previous_splits[
                    "track_id"
                ].astype(str),

                previous_splits[
                    "split"
                ].astype(str)
            )
        )


    duplicate_hashes = set()

    if "file_hash" in df.columns:

        valid_hash_df = df[
            df["file_hash"]
            .notna()
            &
            (
                df["file_hash"]
                .astype(str)
                .str.len()
                > 0
            )
        ]

        duplicated_mask = (
            valid_hash_df[
                "file_hash"
            ].duplicated(
                keep="first"
            )
        )

        duplicate_hashes = set(
            valid_hash_df.loc[
                duplicated_mask,
                "file_hash"
            ].astype(str)
        )


    split_values = []

    for _, row in df.iterrows():

        track_id = str(
            row["track_id"]
        )

        status = str(
            row["status"]
        ).lower()

        file_hash = str(
            row.get(
                "file_hash",
                ""
            )
        )

        # Invalid/error tracks
        if status != "valid":

            split_values.append(
                "excluded"
            )

            continue

        # Exact duplicate
        if file_hash in duplicate_hashes:

            split_values.append(
                "duplicate"
            )

            continue

        # Preserve existing split
        if track_id in previous_split_map:

            old_split = (
                previous_split_map[
                    track_id
                ]
            )

            if old_split in {
                "train",
                "validation",
                "test",
            }:

                split_values.append(
                    old_split
                )

                continue

        # New track
        new_split = (
            assign_split_for_new_track(
                track_id,
                train_ratio,
                validation_ratio,
                test_ratio,
            )
        )

        split_values.append(
            new_split
        )

    df["split"] = split_values


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )


    print(
        "\n" + "=" * 60
    )

    print(
        "TRACK-LEVEL DATASET SPLIT"
    )

    print(
        "=" * 60
    )

    print(
        f"Total tracks: {len(df)}"
    )

    print(
        "\nSplit distribution:"
    )

    print(
        df["split"].value_counts()
    )

    print(
        "\nTrack assignments:"
    )

    print(
        df[
            [
                "track_id",
                "filename",
                "status",
                "split",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {output_path}"
    )

    return df



if __name__ == "__main__":

    create_track_split()