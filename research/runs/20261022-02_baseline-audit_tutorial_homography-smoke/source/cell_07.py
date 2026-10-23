# ==============================
# 3. Player detection and tracking
# ==============================

# Load YOLO11x
model = YOLO("yolo11x.pt")

# Run ByteTrack in streaming mode.
# stream=True prevents Results objects for all video frames
# from accumulating in RAM.
results = model.track(
    source=video_path,
    tracker="bytetrack.yaml",
    device="0",
    classes=[0],       # COCO class 0: person
    save=True,
    save_txt=True,
    save_conf=True,
    stream=True,
    verbose=False
)

# Consume the generator frame by frame.
# Only the current Result object is kept in memory.
out_dir = None

for result in results:
    if out_dir is None:
        out_dir = str(result.save_dir)

if out_dir is None:
    raise RuntimeError(
        "Tracking produced no results."
    )

print("Output directory:", out_dir)