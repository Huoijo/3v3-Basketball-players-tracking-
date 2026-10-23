# ==============================
# 9. Run TI-HOTA evaluation
# ==============================

print(
    "Ground truth:",
    gt_file
)

print(
    "Prediction  :",
    predictions_file
)

frames = load_video(
    gt_file,
    predictions_df
)

# Challenge coordinates are expressed in centimeters
tau = 100.0

metrics, alphas = compute_ti_hota(
    frames,
    tau
)

print(
    "\n=============================="
)

print(
    "TI-HOTA evaluation"
)

print(
    "=============================="
)

print(
    f"TI-HOTA: "
    f"{metrics['TI-HOTA']:.4f}"
)

print(
    f"TI-DetA: "
    f"{metrics['TI-DetA']:.4f}"
)

print(
    f"TI-AssA: "
    f"{metrics['TI-AssA']:.4f}"
)

print(
    f"TI-TP: "
    f"{metrics['TI-TP']:.2f}"
)

print(
    f"TI-FP: "
    f"{metrics['TI-FP']:.2f}"
)

print(
    f"TI-FN: "
    f"{metrics['TI-FN']:.2f}"
)