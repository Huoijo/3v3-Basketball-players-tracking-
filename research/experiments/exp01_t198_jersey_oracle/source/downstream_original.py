identity_df = df.copy()

identity_df["jersey_number"] = identity_df["id"].map(
    track_to_jersey
)

identity_df["cluster_id"] = identity_df["id"].map(
    track_to_cluster
)

identity_df["team_id"] = identity_df["cluster_id"].map(
    cluster_to_team
)

# Remove rows without either a team or a jersey
identity_df = identity_df[
    identity_df["jersey_number"].notna()
    &
    identity_df["team_id"].notna()
].copy()

identity_df["jersey_number"] = identity_df[
    "jersey_number"
].astype(int)

# Keep only roster-compatible identities
identity_df = identity_df[
    identity_df.apply(
        lambda row:
            row["jersey_number"]
            in team_rosters[row["team_id"]],
        axis=1
    )
].copy()

identity_df["identity"] = (
    identity_df["team_id"]
    +
    identity_df["jersey_number"].astype(str)
)

print("\nTracklet identities:")

for track_id in sorted(identity_df["id"].unique()):

    tmp = identity_df[
        identity_df["id"] == track_id
    ]

    identity = tmp["identity"].iloc[0]

    print(
        f"Track {track_id} -> {identity}"
    )


# --------------------------------------------------
# Merge fragmented tracklets through semantic identity
# --------------------------------------------------

identity_df = (
    identity_df
    .sort_values(
        "conf",
        ascending=False
    )
    .drop_duplicates(
        subset=["frame_id", "identity"],
        keep="first"
    )
)

recognized_identities = set(
    identity_df["identity"].unique()
)

required_identities = {
    team_id + str(jersey_number)
    for team_id, jersey_number
    in identity_order
}

missing_identities = (
    required_identities
    - recognized_identities
)

if missing_identities:
    raise RuntimeError(
        "The following player identities were not recognized: "
        + ", ".join(sorted(missing_identities))
    )


# --------------------------------------------------
# Build the final prediction table
# --------------------------------------------------

# The final prediction is the single source of truth for:
#   1. TI-HOTA evaluation
#   2. prediction minimap
#   3. video overlay
#
# Challenge submission fields:
#   frame_id, court_x, court_y, team_id, jersey_number
#
# video_id and bbox columns are retained internally only
# for bookkeeping and visualization.

prediction_tables = []

video_id = os.path.splitext(
    os.path.basename(video_path)
)[0]

# Court position is the evaluated prediction.
# Bbox width/height are auxiliary visualization quantities.
# Bbox x/y will be reconstructed from the final court position
# through the inverse homography, so the video overlay and
# minimap refer to exactly the same predicted position.

interpolation_columns = [
    "court_x",
    "court_y",
    "width",
    "height",
]

H_inv = np.linalg.inv(H)

for team_id, jersey_number in identity_order:

    identity = (
        team_id
        + str(jersey_number)
    )

    player = identity_df[
        identity_df["identity"] == identity
    ].copy()

    player = (
        player
        .sort_values("frame_id")
        .set_index("frame_id")
    )

    # Only predict between this player's first and last
    # observed frames. Do not extend the track to the
    # beginning or end of the entire video.
    full_frame_index = range(
        int(player.index.min()),
        int(player.index.max()) + 1
    )

    player = player.reindex(
        full_frame_index
    )

    # Interpolate gaps between observations.
    player[interpolation_columns] = (
        player[interpolation_columns]
        .interpolate(
            method="linear",
            limit_direction="both"
        )
    )

    player = (
        player
        .reset_index()
        .rename(
            columns={"index": "frame_id"}
        )
    )

    # Map the final evaluated court positions back to
    # image-space foot points. Width/height are retained
    # from the tracking output (with interpolation),
    # because inverse homography alone cannot recover bbox size.

    court_points_final = player[
        ["court_x", "court_y"]
    ].to_numpy(
        dtype=np.float32
    ).reshape(
        -1,
        1,
        2
    )

    image_foot_points = cv2.perspectiveTransform(
        court_points_final,
        H_inv.astype(np.float32)
    ).reshape(
        -1,
        2
    )

    bbox_width = player[
        "width"
    ].to_numpy(
        dtype=float
    )

    bbox_height = player[
        "height"
    ].to_numpy(
        dtype=float
    )

    bbox_x = (
        image_foot_points[:, 0]
        - bbox_width / 2.0
    )

    bbox_y = (
        image_foot_points[:, 1]
        - bbox_height
    )

    output = pd.DataFrame({
        "video_id": video_id,
        "frame_id": player["frame_id"].astype(int),
        "court_x": player["court_x"],
        "court_y": player["court_y"],
        "team_id": team_id,
        "jersey_number": jersey_number,
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
    })

    prediction_tables.append(output)

predictions_df = pd.concat(
    prediction_tables,
    ignore_index=True
)

predictions_df = predictions_df.sort_values(
    [
        "frame_id",
        "team_id",
        "jersey_number"
    ]
).reset_index(
    drop=True
)


# --------------------------------------------------
# Save the final prediction
# --------------------------------------------------

predictions_file = os.path.join(
    out_dir,
    "predictions.csv"
)

# Challenge submission format
submission_columns = [
    "frame_id",
    "court_x",
    "court_y",
    "team_id",
    "jersey_number",
]

predictions_df[
    submission_columns
].to_csv(
    predictions_file,
    index=False,
    float_format="%.5f"
)

print("\nFinal player identities:")

print(
    predictions_df[
        ["team_id", "jersey_number"]
    ]
    .drop_duplicates()
    .to_string(index=False)
)

print(
    "\nPrediction file:",
    predictions_file
)

print(
    "Rows:",
    len(predictions_df)
)

print(
    "Frames:",
    predictions_df["frame_id"].nunique()
)