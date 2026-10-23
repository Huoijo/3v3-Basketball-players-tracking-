# ==============================
# 11. Create GT and prediction minimaps
# ==============================

# Common settings
base_name = os.path.splitext(
    os.path.basename(
        video_path
    )
)[0]


# --------------------------------------------------
# Load and resize Outdoor.png
# --------------------------------------------------

base_image = cv2.imread(
    court_image_path
)

if base_image is None:

    raise FileNotFoundError(
        f"Could not read court image: "
        f"{court_image_path}"
    )


# Original TrackID3x3 Outdoor minimap dimensions
resized_image = cv2.resize(
    base_image,
    (1505, 1105)
)

height, width = (
    resized_image.shape[:2]
)

print(
    f"Image size: "
    f"width={width}, "
    f"height={height}, "
    f"drawing area={height}px"
)


# --------------------------------------------------
# Ground Truth minimap
# --------------------------------------------------

pts = load_gt_points_from_file(
    gt_file
)

if pts:

    scaled = scale_points_swapped(
        pts,
        width,
        height
    )

    frames = {}

    for f, *data in scaled:

        frames.setdefault(
            f,
            []
        ).append(
            [f, *data]
        )


    cap = cv2.VideoCapture(
        video_path
    )

    fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 10
    )

    cap.release()

    print(
        f"GT Using FPS: {fps}"
    )


    gt_minimap_path = os.path.join(
        "/content",
        f"{base_name}_GT.mp4"
    )

    vw = cv2.VideoWriter(
        gt_minimap_path,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height
        )
    )


    for f in sorted(
        frames
    ):

        img = (
            resized_image.copy()
        )

        img = draw_points(
            img,
            frames[f]
        )

        vw.write(
            img
        )


    vw.release()

    print(
        f"GT video saved: "
        f"{gt_minimap_path}"
    )


# --------------------------------------------------
# Prediction minimap
# --------------------------------------------------

pts = prediction_points_from_dataframe(
    predictions_df
)

if pts:

    scaled = scale_points_swapped(
        pts,
        width,
        height
    )

    frames = {}

    for f, *data in scaled:

        frames.setdefault(
            f,
            []
        ).append(
            [f, *data]
        )


    mapping = (
        build_attribute_color_mapping_from_file(
            gt_file
        )
    )


    cap = cv2.VideoCapture(
        video_path
    )

    fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 10
    )

    cap.release()

    print(
        f"Prediction Using FPS: {fps}"
    )


    pred_minimap_path = os.path.join(
        "/content",
        f"{base_name}_pred.mp4"
    )

    vw = cv2.VideoWriter(
        pred_minimap_path,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height
        )
    )


    for f in sorted(
        frames
    ):

        img = (
            resized_image.copy()
        )

        img = draw_points_with_mapping(
            img,
            frames[f],
            mapping
        )

        vw.write(
            img
        )


    vw.release()

    print(
        f"Prediction video saved: "
        f"{pred_minimap_path}"
    )
