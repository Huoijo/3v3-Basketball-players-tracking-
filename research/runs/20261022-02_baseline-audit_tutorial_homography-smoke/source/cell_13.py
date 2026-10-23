# ==============================
# Check the homography
# ==============================

from google.colab.patches import cv2_imshow

cap = cv2.VideoCapture(
    video_path
)

ret, first_frame = cap.read()

cap.release()

if not ret:

    raise IOError(
        "Could not read the first video frame."
    )


# 1 cm corresponds to 1 output pixel
court_width_pixels = 1505
court_height_pixels = 1105

warped_court_image = cv2.warpPerspective(
    first_frame,
    H,
    (
        court_width_pixels,
        court_height_pixels
    )
)

# Resize only for display
display_width = 1000

display_height = int(
    display_width
    * court_height_pixels
    / court_width_pixels
)

display_image = cv2.resize(
    warped_court_image,
    (
        display_width,
        display_height
    )
)

cv2_imshow(
    display_image
)

output_image_path = (
    "/content/"
    "transformed_court_image.jpg"
)

cv2.imwrite(
    output_image_path,
    warped_court_image
)

print(
    "Rectified court image:",
    output_image_path
)