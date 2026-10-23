# ==============================================================
# 4. Convert YOLO tracking output to MOT format
# ==============================================================


def get_frame_id_from_filename(path):
    """
    Extract the trailing frame number from a label filename.

    Example:
        000123.txt -> 123
        IMG_0104_1_123.txt -> 123
    """

    base = os.path.splitext(os.path.basename(path))[0]

    match = re.search(r"(\d+)$", base)

    if not match:
        raise ValueError(
            f"No frame number found in filename: {base}"
        )

    return int(match.group(1))


def parse_yolo_tracking_line(parts):
    """
    Parse one YOLO tracking-label line.

    Returns:
        track_id, confidence, xc, yc, width, height
    """

    xc, yc, w_rel, h_rel = map(float, parts[1:5])

    if len(parts) >= 7:
        a, b = parts[5], parts[6]

        try:
            track_id = int(a)
            confidence = float(b)

        except ValueError:
            confidence = float(a)
            track_id = int(b)

    elif len(parts) == 6:

        track_id = int(parts[5])
        confidence = 1.0

    else:
        raise ValueError(
            f"Cannot parse YOLO tracking line: {parts}"
        )

    return (
        track_id,
        confidence,
        xc,
        yc,
        w_rel,
        h_rel
    )


# --------------------------------------------------
# Create MOT-format file
# --------------------------------------------------

mot_txt_path = os.path.join(
    out_dir,
    "mot_results.txt"
)

label_dir = os.path.join(
    out_dir,
    "labels"
)

label_files = sorted(
    glob.glob(
        os.path.join(label_dir, "*.txt")
    ),
    key=get_frame_id_from_filename
)

# Read video properties
cap = cv2.VideoCapture(video_path)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

cap.release()


with open(mot_txt_path, "w") as mot_file:

    for label_file in label_files:

        frame_id = get_frame_id_from_filename(
            label_file
        )

        with open(label_file, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) < 6:
                    continue

                (
                    track_id,
                    confidence,
                    xc,
                    yc,
                    w_rel,
                    h_rel
                ) = parse_yolo_tracking_line(parts)

                # Relative coordinates -> pixel coordinates
                w_pix = w_rel * width
                h_pix = h_rel * height

                x_pix = (
                    xc * width
                    - w_pix / 2
                )

                y_pix = (
                    yc * height
                    - h_pix / 2
                )

                # MOTChallenge-style format
                mot_file.write(
                    f"{frame_id},"
                    f"{track_id},"
                    f"{x_pix:.2f},"
                    f"{y_pix:.2f},"
                    f"{w_pix:.2f},"
                    f"{h_pix:.2f},"
                    f"{confidence:.2f},"
                    "-1,-1,-1\n"
                )

print("MOT file:", mot_txt_path)


# --------------------------------------------------
# Load MOT data for visualization
# --------------------------------------------------

mot_data = defaultdict(list)

with open(mot_txt_path, "r") as f:

    for line in f:

        (
            frame_id,
            track_id,
            x,
            y,
            w,
            h,
            confidence,
            *_
        ) = line.strip().split(",")

        mot_data[
            int(frame_id)
        ].append({
            "id": int(track_id),
            "bbox": (
                float(x),
                float(y),
                float(w),
                float(h)
            ),
            "conf": float(confidence)
        })


# --------------------------------------------------
# Assign colors to ByteTrack IDs
# --------------------------------------------------

unique_ids = sorted({
    detection["id"]
    for detections in mot_data.values()
    for detection in detections
})

id_to_color = {}

n_ids = max(
    len(unique_ids),
    1
)

for index, track_id in enumerate(unique_ids):

    hue = index / n_ids

    r, g, b = colorsys.hsv_to_rgb(
        hue,
        1.0,
        1.0
    )

    id_to_color[track_id] = (
        int(b * 255),
        int(g * 255),
        int(r * 255)
    )


# --------------------------------------------------
# Create tracking visualization
# --------------------------------------------------

output_tracking_video = os.path.join(
    out_dir,
    "tracked_visualization.mp4"
)

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

cap = cv2.VideoCapture(
    video_path
)

writer = cv2.VideoWriter(
    output_tracking_video,
    fourcc,
    fps,
    (width, height)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id = int(
        cap.get(
            cv2.CAP_PROP_POS_FRAMES
        )
    )

    for detection in mot_data.get(
        frame_id,
        []
    ):

        track_id = detection["id"]

        x, y, w, h = detection[
            "bbox"
        ]

        color = id_to_color[
            track_id
        ]

        cv2.rectangle(
            frame,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            color,
            2
        )

        cv2.putText(
            frame,
            str(track_id),
            (int(x), int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    writer.write(frame)

cap.release()
writer.release()

print(
    "Tracking visualization:",
    output_tracking_video
)