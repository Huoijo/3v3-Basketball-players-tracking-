# ==========================================
# 6. Outdoor court filtering
# ==========================================

columns = [
    "frame_id",
    "id",
    "x",
    "y",
    "width",
    "height",
    "conf",
    "class",
    "visibility",
    "empty",
]

df = pd.read_csv(
    mot_txt_path,
    names=columns,
    sep=","
)


# --------------------------------------------------
# Convert bbox bottom centers to court coordinates
# --------------------------------------------------

df["bot_cx"] = (
    df["x"]
    + df["width"] / 2
)

df["bot_cy"] = (
    df["y"]
    + df["height"]
)

df[
    ["court_x", "court_y"]
] = df.apply(
    lambda row:
        homographic_transformation(
            row["bot_cx"],
            row["bot_cy"],
            H
        ),
    axis=1,
    result_type="expand"
)


# --------------------------------------------------
# Track-level Outdoor filtering
# --------------------------------------------------

paint_line_y = 1002.5
paint_x_min = 502.5
paint_x_max = 1002.5

valid_ids = []

for track_id, group in df.groupby(
    "id"
):

    total = group[
        "frame_id"
    ].nunique()

    inside = group[
        (group["court_x"] >= min_x)
        &
        (group["court_x"] <= max_x)
        &
        (group["court_y"] >= min_y)
        &
        (group["court_y"] <= max_y)
    ]

    # Longest consecutive run inside the court
    frames_inside = sorted(
        inside[
            "frame_id"
        ].unique()
    )

    max_run = 0
    current_run = 0
    previous_frame = None

    for frame_id in frames_inside:

        if (
            previous_frame is not None
            and
            frame_id
            == previous_frame + 1
        ):
            current_run += 1

        else:
            current_run = 1

        max_run = max(
            max_run,
            current_run
        )

        previous_frame = frame_id


    # Near-side endline region
    c1 = (
        group["court_y"].between(
            -300,
            100
        )
        &
        group["court_x"].between(
            0,
            1505
        )
    ).sum()


    # Far-side region outside the paint extension
    c2 = (
        (group["court_y"] > paint_line_y)
        &
        (
            (group["court_x"] < paint_x_min)
            |
            (group["court_x"] > paint_x_max)
        )
    ).sum()


    if (
        max_run >= 10
        and
        c1 <= total / 2
        and
        c2 <= total / 2
    ):
        valid_ids.append(
            track_id
        )


df = df[
    df["id"].isin(
        valid_ids
    )
].copy()


# --------------------------------------------------
# Detection-level filtering
# --------------------------------------------------

buffer_cm = 300

df = df[
    (df["court_x"] >= min_x - buffer_cm)
    &
    (df["court_x"] <= max_x + buffer_cm)
    &
    (df["court_y"] >= min_y - buffer_cm)
    &
    (df["court_y"] <= max_y + buffer_cm)
].copy()


df["frame_id"] = (
    df["frame_id"].astype(int)
)

df["id"] = (
    df["id"].astype(int)
)


# --------------------------------------------------
# Read the actual video length
# --------------------------------------------------

cap = cv2.VideoCapture(
    video_path
)

total_video_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

cap.release()


# --------------------------------------------------
# Save filtered tracklets
# --------------------------------------------------

out_bbox = os.path.join(
    out_dir,
    "filtered_mot.txt"
)

df[
    [
        "frame_id",
        "id",
        "x",
        "y",
        "width",
        "height",
        "conf",
    ]
].to_csv(
    out_bbox,
    index=False,
    header=False
)


out_court = os.path.join(
    out_dir,
    "transformed_court.txt"
)

df[
    [
        "frame_id",
        "id",
        "court_x",
        "court_y",
    ]
].to_csv(
    out_court,
    index=False,
    header=False
)


print("Outdoor filtering completed.")

print(
    "Remaining ByteTrack IDs:",
    sorted(
        df["id"].unique()
    )
)

print(
    "Number of tracklets:",
    df["id"].nunique()
)

print(
    "Number of video frames:",
    total_video_frames
)

print(
    "Filtered MOT file:",
    out_bbox
)

print(
    "Court-coordinate file:",
    out_court
)