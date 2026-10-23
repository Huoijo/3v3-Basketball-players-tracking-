# ==============================
# 2. Download tutorial files
# ==============================

folder_url = (
    "https://drive.google.com/drive/folders/"
    "1_yDMy8YVpxyqcKg2_eXN1xz_vg2FeCGu"
)

download_dir = "/content/trackid3x3_tutorial_data"
os.makedirs(download_dir, exist_ok=True)

print("Downloading tutorial files from Google Drive...")

!gdown "{folder_url}" --folder -O "{download_dir}"


def find_file(filename):
    """Find a downloaded file recursively."""

    matches = glob.glob(
        os.path.join(download_dir, "**", filename),
        recursive=True
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"{filename} was not found in {download_dir}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files named {filename} were found:\n{matches}"
        )

    return matches[0]


video_path = find_file("IMG_0104_1.MOV")
gt_file = find_file("IMG_0104_1_tutorial_challenge.csv")
keypoint_file = find_file("Outdoor.ndjson")
court_image_path = find_file("Outdoor.png")

print("\nDownloaded files:")
print("Input video :", video_path)
print("Ground truth:", gt_file)
print("Keypoints   :", keypoint_file)
print("Court image :", court_image_path)