# ==============================
# 12. Overlay minimaps
# ==============================

import os
import cv2
import numpy as np


# --------------------------------------------------
# File paths
# --------------------------------------------------

input_video_path = video_path

pred_video_path = pred_minimap_path
gt_video_path = gt_minimap_path

output_video_path = os.path.join(
    "/content",
    f"{base_name}_with_minimap.mp4"
)


# --------------------------------------------------
# Use the final prediction table for video drawing
# --------------------------------------------------

prediction_rows_by_frame = {
    int(frame_id): group.copy()
    for frame_id, group
    in predictions_df.groupby("frame_id")
}


# --------------------------------------------------
# Open videos
# --------------------------------------------------

cap = cv2.VideoCapture(
    input_video_path
)

pred_cap = cv2.VideoCapture(
    pred_video_path
)

gt_cap = cv2.VideoCapture(
    gt_video_path
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video: "
        f"{input_video_path}"
    )

main_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

main_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

fps = (
    cap.get(
        cv2.CAP_PROP_FPS
    )
    or 10
)

fourcc = (
    cv2.VideoWriter_fourcc(
        *"mp4v"
    )
)

out = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (
        main_width,
        main_height
    )
)


# --------------------------------------------------
# Original TrackID3x3 minimap sizing
# --------------------------------------------------

M_base = (
    main_width / 5.0
)

L = (
    4 / 3.0
) * M_base

# Read first prediction minimap frame
ret_pred, first_pred = (
    pred_cap.read()
)

if not ret_pred:

    raise RuntimeError(
        "Failed to read prediction minimap."
    )

orig_pred_h, orig_pred_w = (
    first_pred.shape[:2]
)

r = (
    orig_pred_h
    / orig_pred_w
)

# Original code enlarges the minimap by 1.1
M = (
    M_base * 1.1
)

H = (
    M * r
)

# Reset minimap frame positions
pred_cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)

gt_cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)

# Original placement
label_space = 20

overlay_y = int(
    label_space
)

pred_x = int(
    L
)

gt_x = int(
    L
    + M_base
    + (
        M_base / 3.0
    )
)


# --------------------------------------------------
# Label settings
# --------------------------------------------------

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_thickness = 5
label_color = (
    255,
    255,
    255
)


# --------------------------------------------------
# Process video
# --------------------------------------------------

frame_index = 1

while True:

    ret_main, main_frame = (
        cap.read()
    )

    if not ret_main:
        break

    current_frame = frame_index


    # --------------------------------------------------
    # Draw the same final predictions on the original video
    # --------------------------------------------------

    if current_frame in prediction_rows_by_frame:

        frame_predictions = prediction_rows_by_frame[
            current_frame
        ]

        for row in frame_predictions.itertuples(index=False):

            x = float(row.bbox_x)
            y = float(row.bbox_y)
            w = float(row.bbox_width)
            h = float(row.bbox_height)

            if not np.isfinite(
                [x, y, w, h]
            ).all():
                continue

            a = w / 2
            b = a / 5

            if (
                a <= 1
                or
                b <= 1
            ):
                continue

            cx = int(
                x + w / 2
            )

            cy = int(
                y
                + h
                + b / 2
            )

            attribute = (
                str(row.team_id)
                + str(int(row.jersey_number))
            )

            color = get_color(
                attribute
            )

            if color == (
                0,
                0,
                0
            ):
                continue

            cv2.ellipse(
                main_frame,
                (
                    cx,
                    cy
                ),
                (
                    int(a),
                    int(b)
                ),
                0,
                330,
                570,
                color,
                3
            )

            text_pos = (
                int(x + 25),
                int(y - 10)
            )

            cv2.putText(
                main_frame,
                attribute,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                color,
                3,
                cv2.LINE_AA
            )


    # --------------------------------------------------
    # Overlay minimaps at the top
    # --------------------------------------------------

    ret_pred, frame_pred = (
        pred_cap.read()
    )

    ret_gt, frame_gt = (
        gt_cap.read()
    )

    if (
        ret_pred
        and
        ret_gt
    ):

        frame_pred_resized = (
            cv2.resize(
                frame_pred,
                (
                    int(M),
                    int(H)
                ),
                interpolation=(
                    cv2.INTER_LINEAR
                )
            )
        )

        frame_gt_resized = (
            cv2.resize(
                frame_gt,
                (
                    int(M),
                    int(H)
                ),
                interpolation=(
                    cv2.INTER_LINEAR
                )
            )
        )

        alpha = 0.9

        # Prediction minimap
        roi_pred = main_frame[
            overlay_y:
                overlay_y + int(H),
            pred_x:
                pred_x + int(M)
        ]

        main_frame[
            overlay_y:
                overlay_y + int(H),
            pred_x:
                pred_x + int(M)
        ] = cv2.addWeighted(
            roi_pred,
            1 - alpha,
            frame_pred_resized,
            alpha,
            0
        )

        # Ground Truth minimap
        roi_gt = main_frame[
            overlay_y:
                overlay_y + int(H),
            gt_x:
                gt_x + int(M)
        ]

        main_frame[
            overlay_y:
                overlay_y + int(H),
            gt_x:
                gt_x + int(M)
        ] = cv2.addWeighted(
            roi_gt,
            1 - alpha,
            frame_gt_resized,
            alpha,
            0
        )

        # --------------------------------------------------
        # Labels
        # --------------------------------------------------

        pred_label_x = (
            pred_x
            + int(M / 2)
        )

        gt_label_x = (
            gt_x
            + int(M / 2)
        )

        label_y = (
            overlay_y
            + int(H)
            + label_space
            + 42
        )

        # Same geometry as the English fallback
        # in the original draw_minimap.ipynb.
        cv2.putText(
            main_frame,
            "Prediction",
            (
                pred_label_x - 150,
                label_y
            ),
            font,
            font_scale,
            label_color,
            font_thickness
        )

        cv2.putText(
            main_frame,
            "Ground Truth",
            (
                gt_label_x - 195,
                label_y
            ),
            font,
            font_scale,
            label_color,
            font_thickness
        )

    out.write(
        main_frame
    )

    frame_index += 1


# --------------------------------------------------
# Release resources
# --------------------------------------------------

cap.release()
pred_cap.release()
gt_cap.release()
out.release()

print(
    "Processing is complete. "
    f"Output: {output_video_path}"
)
