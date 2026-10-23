# ==============================
# 9.1 Location-only diagnostics
# ==============================

def compute_location_diagnostics(
    gt_file,
    predictions_df,
    max_distance_cm=100.0
):
    """
    Match GT and predictions using court position only.

    Diagnostic metrics are computed only for matched pairs
    within max_distance_cm.
    """

    gt_df = pd.read_csv(gt_file)

    gt_by_frame = {
        int(frame_id): group.reset_index(drop=True)
        for frame_id, group in gt_df.groupby("frame_id")
    }

    pred_by_frame = {
        int(frame_id): group.reset_index(drop=True)
        for frame_id, group in predictions_df.groupby("frame_id")
    }

    position_errors = []
    team_correct = []
    jersey_correct = []

    all_frame_ids = sorted(
        set(gt_by_frame.keys())
        | set(pred_by_frame.keys())
    )

    for frame_id in all_frame_ids:

        gt = gt_by_frame.get(frame_id)
        pred = pred_by_frame.get(frame_id)

        if gt is None or pred is None:
            continue

        gt_xy = gt[
            ["court_x", "court_y"]
        ].to_numpy(dtype=float)

        pred_xy = pred[
            ["court_x", "court_y"]
        ].to_numpy(dtype=float)

        distances = np.linalg.norm(
            gt_xy[:, None, :]
            - pred_xy[None, :, :],
            axis=2
        )

        # Location-only Hungarian matching
        cost = distances.copy()
        cost[distances > max_distance_cm] = 1e6

        gt_indices, pred_indices = (
            linear_sum_assignment(cost)
        )

        for i, j in zip(
            gt_indices,
            pred_indices
        ):
            distance = distances[i, j]

            if distance > max_distance_cm:
                continue

            position_errors.append(distance)

            team_correct.append(
                str(gt.iloc[i]["team_id"])
                == str(pred.iloc[j]["team_id"])
            )

            jersey_correct.append(
                int(gt.iloc[i]["jersey_number"])
                == int(pred.iloc[j]["jersey_number"])
            )

    if not position_errors:
        return {
            "matched_pairs": 0,
            "median_position_error_cm": np.nan,
            "team_accuracy": np.nan,
            "jersey_accuracy": np.nan,
        }

    return {
        "matched_pairs": len(position_errors),
        "median_position_error_cm":
            float(np.median(position_errors)),
        "team_accuracy":
            float(np.mean(team_correct)),
        "jersey_accuracy":
            float(np.mean(jersey_correct)),
    }


diagnostics = compute_location_diagnostics(
    gt_file,
    predictions_df,
    max_distance_cm=100.0
)

print("\n==============================")
print("Location-only diagnostics")
print("==============================")

print(
    "Matched pairs:",
    diagnostics["matched_pairs"]
)

print(
    "Median court-position error:",
    f"{diagnostics['median_position_error_cm']:.2f} cm"
)

print(
    "Team-identification accuracy:",
    f"{diagnostics['team_accuracy']:.4f}"
)

print(
    "Jersey-number accuracy:",
    f"{diagnostics['jersey_accuracy']:.4f}"
)