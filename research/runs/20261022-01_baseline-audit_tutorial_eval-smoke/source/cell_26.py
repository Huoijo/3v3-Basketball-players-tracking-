# ==============================
# 10. Minimap helper functions
# ==============================

import cv2
import glob
import os


def load_gt_points_from_file(file_path):
    """
    Read Challenge-format GT data and return:
    (frame, x_cm, y_cm, attribute)
    """

    gt_df = pd.read_csv(
        file_path
    )

    points = []

    for row in gt_df.itertuples(index=False):

        frame = int(
            row.frame_id
        )

        x = float(
            row.court_x
        )

        y = float(
            row.court_y
        )

        attr = (
            str(row.team_id)
            + str(int(row.jersey_number))
        )

        points.append(
            (
                frame,
                x,
                y,
                attr
            )
        )

    return points


def prediction_points_from_dataframe(
    predictions_df
):
    """
    Build minimap points from the same final prediction rows
    used for evaluation and video visualization.
    """

    points = []

    for row in predictions_df.itertuples(index=False):

        points.append(
            (
                int(row.frame_id),
                float(row.court_x),
                float(row.court_y),
                str(row.team_id)
                + str(int(row.jersey_number))
            )
        )

    return points


def scale_points_swapped(
    points,
    img_width,
    draw_height
):
    """
    Follow the original TrackID3x3 Outdoor minimap
    coordinate transformation.

    x is horizontally flipped.
    y is unchanged.
    """

    flipped_points = []

    for point in points:

        frame, x, y, attr = point

        flipped_x = (
            img_width - x
        )

        flipped_points.append(
            (
                frame,
                flipped_x,
                y,
                attr
            )
        )

    return flipped_points


def get_color(attr):
    """
    Fixed Challenge team colors.

    A = initial Offense = purple
    B = initial Defense = green

    O/D are also accepted for compatibility.
    """

    if (
        attr.startswith("A")
        or
        attr.startswith("O")
    ):
        # purple
        return (128, 0, 128)

    elif (
        attr.startswith("B")
        or
        attr.startswith("D")
    ):
        # green
        return (0, 200, 0)

    else:
        return (0, 0, 0)


def build_attribute_color_mapping_from_file(
    gt_file
):
    """
    Build attribute -> color mapping from GT.
    """

    mapping = {}

    pts = load_gt_points_from_file(
        gt_file
    )

    for p in pts:

        _, _, _, attr = p

        if attr not in mapping:

            mapping[attr] = get_color(
                attr
            )

    return mapping


def draw_points(
    image,
    points
):
    """
    Draw GT points using the original
    create_minimap.ipynb drawing parameters.
    """

    for point in points:

        frame, x, y, attr = point

        color = get_color(
            attr
        )

        x_int = int(x)
        y_int = int(y)

        cv2.circle(
            image,
            (x_int, y_int),
            radius=30,
            color=color,
            thickness=-1
        )

        cv2.putText(
            image,
            attr,
            (
                x_int + 10,
                y_int - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            color,
            thickness=8
        )

    return image


def draw_points_with_mapping(
    image,
    points,
    attr_color_mapping
):
    """
    Draw prediction points using the GT
    attribute-color mapping.
    """

    for point in points:

        frame, x, y, attr = point

        if attr in attr_color_mapping:

            color = (
                attr_color_mapping[
                    attr
                ]
            )

        else:

            color = get_color(
                attr
            )

        x_int = int(x)
        y_int = int(y)

        cv2.circle(
            image,
            (x_int, y_int),
            radius=30,
            color=color,
            thickness=-1
        )

        cv2.putText(
            image,
            attr,
            (
                x_int + 10,
                y_int - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            color,
            thickness=8
        )

    return image