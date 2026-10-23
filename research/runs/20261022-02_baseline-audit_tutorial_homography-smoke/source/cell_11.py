# ==============================
# 5. Outdoor court homography
# ==============================

with open(keypoint_file, "r") as f:
    data = ndjson.load(f)

# Retrieve the Labelbox project
project = next(
    iter(
        data[0]["projects"].values()
    )
)

annotations = (
    project["labels"][0]
    ["annotations"]["objects"]
)

# Outdoor court coordinates in centimeters
keypoint_mapping = {
    "rku_0":  [0, 0],
    "rku_1":  [0, 1105],
    "rku_10": [1505, 0],
    "rku_11": [1505, 1105],
}

image_points = []
court_points = []

for obj in annotations:

    name = obj["value"]

    if name not in keypoint_mapping:
        continue

    x = obj["point"]["x"]
    y = obj["point"]["y"]

    image_points.append(
        [x, y]
    )

    court_points.append(
        keypoint_mapping[name]
    )


image_points = np.asarray(
    image_points,
    dtype=np.float32
)

court_points = np.asarray(
    court_points,
    dtype=np.float32
)

if len(image_points) != 4:

    raise ValueError(
        "Expected 4 Outdoor court keypoints, "
        f"but found {len(image_points)}."
    )


print("Image coordinates:")
print(image_points)

print("\nCourt coordinates [cm]:")
print(court_points)


# Compute homography
H, _ = cv2.findHomography(
    image_points,
    court_points
)

if H is None:

    raise RuntimeError(
        "Failed to calculate the homography."
    )

print("\nHomography:")
print(H)


def homographic_transformation(x, y, H):
    """
    Transform one image point into Outdoor court
    coordinates in centimeters.
    """

    point = np.array(
        [x, y, 1.0],
        dtype=np.float64
    ).reshape(3, 1)

    transformed = H @ point

    transformed /= transformed[
        2, 0
    ]

    return (
        float(transformed[0, 0]),
        float(transformed[1, 0])
    )


# Outdoor court boundaries [cm]
min_x = 0.0
max_x = 1505.0

min_y = 0.0
max_y = 1105.0

print(
    "\nCourt range [cm]: "
    f"x=[{min_x}, {max_x}], "
    f"y=[{min_y}, {max_y}]"
)