# ==========================================================
# 7. Team identification and jersey-number recognition
# ==========================================================

# --------------------------------------------------
# Fixed roster for this tutorial clip
# --------------------------------------------------

team_rosters = {
    "A": {4, 10, 12},
    "B": {6, 9, 11},
}

all_valid_jerseys = set().union(
    *team_rosters.values()
)

identity_order = [
    ("A", 4),
    ("A", 10),
    ("A", 12),
    ("B", 6),
    ("B", 9),
    ("B", 11),
]


# --------------------------------------------------
# OCR and team-color settings
# --------------------------------------------------

reader = easyocr.Reader(
    ["en"],
    gpu=True
)

MAX_SAMPLED_FRAMES_PER_TRACK = 40
MIN_OCR_CONFIDENCE = 0.10

# Histogram settings for team identification
H_BINS = 16
S_BINS = 8


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_upper_body_crop(frame, x, y, w, h):
    """
    Crop the upper-body region from a player bounding box.
    """

    x1 = int(max(0, x))
    y1 = int(max(0, y))

    x2 = int(min(frame.shape[1], x + w))
    y2 = int(min(frame.shape[0], y + h))

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    H = crop.shape[0]
    W = crop.shape[1]

    # Center-upper torso region
    tx1 = int(0.20 * W)
    tx2 = int(0.80 * W)
    ty1 = int(0.12 * H)
    ty2 = int(0.62 * H)

    torso = crop[ty1:ty2, tx1:tx2]

    if torso.size == 0:
        return None

    return torso


def extract_color_feature(image):
    """
    Extract an HSV histogram feature from a torso crop.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [H_BINS, S_BINS],
        [0, 180, 0, 256]
    )

    hist = cv2.normalize(
        hist,
        None
    ).flatten()

    return hist.astype(np.float32)


def infer_roster_team_from_jersey(jersey_number):
    """
    Infer the team from the roster if the jersey number
    belongs to exactly one team.
    """

    candidate_teams = [
        team_id
        for team_id, roster in team_rosters.items()
        if jersey_number in roster
    ]

    if len(candidate_teams) == 1:
        return candidate_teams[0]

    return None


# --------------------------------------------------
# Sample frames from each ByteTrack tracklet
# --------------------------------------------------

sampled_groups = []

for track_id, group in df.groupby(
    "id"
):
    group = group.sort_values(
        "frame_id"
    )

    n_samples = min(
        MAX_SAMPLED_FRAMES_PER_TRACK,
        len(group)
    )

    sample_indices = np.linspace(
        0,
        len(group) - 1,
        n_samples,
        dtype=int
    )

    sample_indices = np.unique(
        sample_indices
    )

    sampled_groups.append(
        group.iloc[sample_indices]
    )

if not sampled_groups:
    raise RuntimeError(
        "No tracklets remain after court filtering."
    )

sample_df = pd.concat(
    sampled_groups,
    ignore_index=True
)

samples_by_frame = {
    int(frame_id): group.to_dict("records")
    for frame_id, group
    in sample_df.groupby("frame_id")
}


# --------------------------------------------------
# Run OCR and color-feature extraction
# --------------------------------------------------

ocr_scores = defaultdict(
    lambda: defaultdict(float)
)

track_color_features = defaultdict(list)

cap = cv2.VideoCapture(
    video_path
)

frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    if frame_id not in samples_by_frame:
        continue

    for row in samples_by_frame[frame_id]:

        track_id = int(
            row["id"]
        )

        x = row["x"]
        y = row["y"]
        w = row["width"]
        h = row["height"]

        # --------------------------
        # OCR on full player bbox
        # --------------------------

        x1 = int(max(0, x))
        y1 = int(max(0, y))
        x2 = int(min(frame.shape[1], x + w))
        y2 = int(min(frame.shape[0], y + h))

        player_crop = frame[
            y1:y2,
            x1:x2
        ]

        if player_crop.size != 0:

            ocr_crop = cv2.resize(
                player_crop,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_CUBIC
            )

            results = reader.readtext(
                ocr_crop,
                allowlist="0123456789",
                detail=1,
                paragraph=False
            )

            for _, text, confidence in results:

                if confidence < MIN_OCR_CONFIDENCE:
                    continue

                digits = re.sub(
                    r"\D",
                    "",
                    text
                )

                if digits == "":
                    continue

                jersey_number = int(digits)

                if jersey_number not in all_valid_jerseys:
                    continue

                ocr_scores[
                    track_id
                ][
                    jersey_number
                ] += float(confidence)

        # --------------------------
        # Team feature on upper body
        # --------------------------

        torso = get_upper_body_crop(
            frame,
            x,
            y,
            w,
            h
        )

        if torso is not None:

            feature = extract_color_feature(
                torso
            )

            track_color_features[
                track_id
            ].append(feature)

cap.release()


# --------------------------------------------------
# Assign a jersey number to each tracklet
# --------------------------------------------------

track_to_jersey = {}

print("Jersey OCR results:")

for track_id in sorted(df["id"].unique()):

    scores = ocr_scores.get(
        track_id,
        {}
    )

    if not scores:
        track_to_jersey[
            track_id
        ] = None

        print(
            f"Track {track_id}: "
            "no valid jersey number recognized"
        )

        continue

    jersey_number = max(
        scores,
        key=scores.get
    )

    track_to_jersey[
        track_id
    ] = jersey_number

    score_string = ", ".join(
        f"{number}:{score:.2f}"
        for number, score
        in sorted(
            scores.items(),
            key=lambda item: -item[1]
        )
    )

    print(
        f"Track {track_id} -> "
        f"jersey {jersey_number} "
        f"({score_string})"
    )


# --------------------------------------------------
# Team identification by jersey-color clustering
# --------------------------------------------------

track_feature_vectors = {}

for track_id, features in track_color_features.items():

    if len(features) == 0:
        continue

    track_feature_vectors[
        track_id
    ] = np.mean(
        np.stack(features, axis=0),
        axis=0
    ).astype(np.float32)

if len(track_feature_vectors) < 2:
    raise RuntimeError(
        "Not enough tracklets with color features to run "
        "team identification."
    )

cluster_track_ids = sorted(
    track_feature_vectors.keys()
)

feature_matrix = np.stack(
    [
        track_feature_vectors[track_id]
        for track_id in cluster_track_ids
    ],
    axis=0
).astype(np.float32)

criteria = (
    cv2.TERM_CRITERIA_EPS
    + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1.0
)

compactness, labels, centers = cv2.kmeans(
    feature_matrix,
    2,
    None,
    criteria,
    20,
    cv2.KMEANS_PP_CENTERS
)

track_to_cluster = {
    track_id: int(label[0])
    for track_id, label
    in zip(cluster_track_ids, labels)
}

print("\nTeam-cluster assignment:")

for track_id in sorted(track_to_cluster):

    print(
        f"Track {track_id} -> cluster {track_to_cluster[track_id]}"
    )


# --------------------------------------------------
# Map clusters to Team A / Team B
# using the initial offense position
# --------------------------------------------------

# In the original TrackID3x3 Outdoor baseline,
# for a check-ball segment starting at the top,
# the player closest to the midpoint of the outer court line
# parallel to the end line is identified as the initial offense.
#
# In this tutorial:
#   Team A = initial offense
#   Team B = initial defense
#
# Team separation itself has already been performed by
# jersey-color clustering. Here, the initial position is used
# only to resolve which of the two arbitrary cluster labels
# corresponds to Team A.

eligible_track_ids = set(
    track_to_cluster.keys()
)

eligible_df = df[
    df["id"].isin(eligible_track_ids)
].copy()


# --------------------------------------------------
# Use the first available frame
# --------------------------------------------------

initial_frame = int(
    eligible_df["frame_id"].min()
)

initial_df = eligible_df[
    eligible_df["frame_id"] == initial_frame
].copy()

if len(initial_df) == 0:
    raise RuntimeError(
        "No player detections were available in the initial frame."
    )


# --------------------------------------------------
# Find the initial offensive player
# --------------------------------------------------

# Midpoint of the outer court line parallel to the end line
# [cm]

top_reference = np.array(
    [
        (min_x + max_x) / 2.0,
        max_y
    ],
    dtype=np.float32
)

initial_positions = initial_df[
    [
        "court_x",
        "court_y"
    ]
].to_numpy(
    dtype=np.float32
)

distances_to_top = np.linalg.norm(
    initial_positions - top_reference,
    axis=1
)

offense_row = initial_df.iloc[
    int(np.argmin(distances_to_top))
]

offense_track_id = int(
    offense_row["id"]
)

offense_cluster = track_to_cluster[
    offense_track_id
]


# --------------------------------------------------
# Find the initial defensive player
# --------------------------------------------------

# Following the original Outdoor baseline,
# the player closest to the identified offensive player
# is treated as the initial defender.

other_players = initial_df[
    initial_df["id"] != offense_track_id
].copy()

if len(other_players) == 0:
    raise RuntimeError(
        "No candidate defender was available in the initial frame."
    )

offense_position = np.array(
    [
        offense_row["court_x"],
        offense_row["court_y"]
    ],
    dtype=np.float32
)

other_positions = other_players[
    [
        "court_x",
        "court_y"
    ]
].to_numpy(
    dtype=np.float32
)

distances_to_offense = np.linalg.norm(
    other_positions - offense_position,
    axis=1
)

defense_row = other_players.iloc[
    int(np.argmin(distances_to_offense))
]

defense_track_id = int(
    defense_row["id"]
)

defense_cluster = track_to_cluster[
    defense_track_id
]


# --------------------------------------------------
# Resolve the arbitrary k-means cluster labels
# --------------------------------------------------

clusters = sorted(
    set(track_to_cluster.values())
)

if len(clusters) != 2:
    raise RuntimeError(
        f"Expected 2 team clusters, but found {len(clusters)}."
    )

if offense_cluster == defense_cluster:
    raise RuntimeError(
        "The initial offense and defense were assigned to the "
        "same color cluster. Team-cluster alignment is ambiguous."
    )

cluster_to_team = {
    offense_cluster: "A",
    defense_cluster: "B",
}

print(
    "\nInitial team anchors:"
)

print(
    f"Frame {initial_frame}: "
    f"Track {offense_track_id} -> initial offense -> Team A"
)

print(
    f"Frame {initial_frame}: "
    f"Track {defense_track_id} -> initial defense -> Team B"
)

print(
    "\nCluster-to-team mapping:"
)

for cluster_id in sorted(cluster_to_team):

    print(
        f"Cluster {cluster_id} "
        f"-> Team {cluster_to_team[cluster_id]}"
    )


# --------------------------------------------------
# Build semantic identities
# --------------------------------------------------

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