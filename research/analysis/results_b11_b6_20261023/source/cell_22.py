# ==============================
# 8. TI-HOTA evaluation functions
# ==============================


def loc_sim(pred_position, gt_position, tau):
    """
    Compute location similarity.

    Coordinates and tau must use the same unit.
    In this tutorial they are expressed in centimeters.
    """

    pred_position = np.asarray(
        pred_position,
        dtype=float
    )

    gt_position = np.asarray(
        gt_position,
        dtype=float
    )

    distance_squared = np.sum(
        (
            pred_position
            - gt_position
        ) ** 2
    )

    return math.exp(
        math.log(0.05)
        *
        distance_squared
        /
        (tau ** 2)
    )


def id_sim(pred_attribute, gt_attribute):
    """
    Identity similarity is 1 only when both
    team ID and jersey number are correct.
    """

    return int(
        pred_attribute
        == gt_attribute
    )


def sim_ti_hota(prediction, ground_truth, tau):
    """Combined TI-HOTA similarity."""

    return (
        loc_sim(
            prediction["position"],
            ground_truth["position"],
            tau
        )
        *
        id_sim(
            prediction["attributes"],
            ground_truth["attributes"]
        )
    )


def compute_ti_hota_alpha(
    frames,
    tau,
    alpha
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    association_counts = {}
    false_positive_association_counts = {}


    for frame in frames:

        gt_objects = frame.get(
            "ground_truths",
            []
        )

        pred_objects = frame.get(
            "predictions",
            []
        )

        n_gt = len(
            gt_objects
        )

        n_pred = len(
            pred_objects
        )


        # Similarity matrix
        combined = np.zeros(
            (
                n_gt,
                n_pred
            )
        )

        for i, gt in enumerate(
            gt_objects
        ):

            for j, pred in enumerate(
                pred_objects
            ):

                combined[
                    i, j
                ] = (
                    loc_sim(
                        pred["position"],
                        gt["position"],
                        tau
                    )
                    *
                    id_sim(
                        pred["attributes"],
                        gt["attributes"]
                    )
                )


        current_gt_matches = {}


        if (
            n_gt > 0
            and
            n_pred > 0
        ):

            cost = (
                1.0
                - combined
            )

            cost[
                combined < alpha
            ] = 1e6

            row_indices, col_indices = (
                linear_sum_assignment(
                    cost
                )
            )

            for i, j in zip(
                row_indices,
                col_indices
            ):

                if (
                    combined[i, j]
                    >= alpha
                ):
                    current_gt_matches[
                        i
                    ] = j


        # Detection counts
        tp = len(
            current_gt_matches
        )

        fp = (
            n_pred
            - tp
        )

        fn = (
            n_gt
            - tp
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn


        # --------------------------------------------------
        # Association counts
        # --------------------------------------------------

        matched_gt = set(
            current_gt_matches.keys()
        )


        for i, gt in enumerate(
            gt_objects
        ):

            identity = gt[
                "attributes"
            ]

            association_counts.setdefault(
                identity,
                {
                    "TPA": 0,
                    "FNA": 0
                }
            )

            if i not in matched_gt:

                association_counts[
                    identity
                ][
                    "FNA"
                ] += 1


        for i, j in (
            current_gt_matches.items()
        ):

            identity = gt_objects[
                i
            ][
                "attributes"
            ]

            association_counts[
                identity
            ][
                "TPA"
            ] += 1


        matched_predictions = set(
            current_gt_matches.values()
        )

        for j, pred in enumerate(
            pred_objects
        ):

            if j not in matched_predictions:

                identity = pred[
                    "attributes"
                ]

                false_positive_association_counts[
                    identity
                ] = (
                    false_positive_association_counts.get(
                        identity,
                        0
                    )
                    + 1
                )


    # Detection accuracy
    detection_denominator = (
        total_tp
        + total_fp
        + total_fn
    )

    det_a = (
        total_tp
        / detection_denominator
        if detection_denominator > 0
        else 0.0
    )


    # Association accuracy
    association_sum = 0.0
    n_identities = 0

    for identity, counts in (
        association_counts.items()
    ):

        denominator = (
            counts["TPA"]
            +
            counts["FNA"]
            +
            false_positive_association_counts.get(
                identity,
                0
            )
        )

        if denominator > 0:

            association_sum += (
                counts["TPA"]
                / denominator
            )

            n_identities += 1


    ass_a = (
        association_sum
        / n_identities
        if n_identities > 0
        else 0.0
    )


    ti_hota_alpha = math.sqrt(
        det_a * ass_a
    )


    return (
        ti_hota_alpha,
        det_a,
        ass_a,
        total_tp,
        total_fp,
        total_fn
    )


def compute_ti_hota(
    frames,
    tau
):

    alphas = np.arange(
        0.05,
        1.0,
        0.05
    )

    ti_hota_values = []
    det_a_values = []
    ass_a_values = []

    tp_values = []
    fp_values = []
    fn_values = []


    for alpha in alphas:

        (
            ti_hota_alpha,
            det_a,
            ass_a,
            tp,
            fp,
            fn
        ) = compute_ti_hota_alpha(
            frames,
            tau,
            alpha
        )

        ti_hota_values.append(
            ti_hota_alpha
        )

        det_a_values.append(
            det_a
        )

        ass_a_values.append(
            ass_a
        )

        tp_values.append(
            tp
        )

        fp_values.append(
            fp
        )

        fn_values.append(
            fn
        )


    metrics = {
        "TI-HOTA":
            float(
                np.mean(
                    ti_hota_values
                )
            ),

        "TI-HOTA_alphas":
            ti_hota_values,

        "TI-DetA":
            float(
                np.mean(
                    det_a_values
                )
            ),

        "TI-AssA":
            float(
                np.mean(
                    ass_a_values
                )
            ),

        "TI-TP":
            float(
                np.mean(
                    tp_values
                )
            ),

        "TI-FP":
            float(
                np.mean(
                    fp_values
                )
            ),

        "TI-FN":
            float(
                np.mean(
                    fn_values
                )
            ),
    }

    return (
        metrics,
        alphas
    )


# --------------------------------------------------
# Read Challenge-format files
# --------------------------------------------------

def load_gt_frames(file_path):
    """
    Read Challenge-format ground truth:

    frame_id, court_x, court_y, team_id, jersey_number

    team_id + jersey_number is the persistent
    semantic player identity.
    """

    gt_df = pd.read_csv(
        file_path
    )

    required_columns = [
        "frame_id",
        "court_x",
        "court_y",
        "team_id",
        "jersey_number",
    ]

    if list(gt_df.columns) != required_columns:
        raise ValueError(
            "GT columns must be:\n"
            + ", ".join(required_columns)
        )

    frames_dict = defaultdict(list)

    for row in gt_df.itertuples(index=False):

        semantic_identity = (
            str(row.team_id)
            + str(int(row.jersey_number))
        )

        frames_dict[
            int(row.frame_id)
        ].append({
            "position": [
                float(row.court_x),
                float(row.court_y)
            ],
            "attributes": semantic_identity
        })

    return frames_dict


def load_prediction_frames(predictions_df):
    """
    Convert the canonical final prediction table into
    the structure used by TI-HOTA.

    team_id + jersey_number is the persistent
    semantic player identity.
    """

    frames_dict = defaultdict(list)

    for row in predictions_df.itertuples(index=False):

        semantic_identity = (
            str(row.team_id)
            + str(int(row.jersey_number))
        )

        frames_dict[
            int(row.frame_id)
        ].append({
            "position": [
                float(row.court_x),
                float(row.court_y)
            ],
            "attributes": semantic_identity
        })

    return frames_dict


def load_video(
    gt_file,
    predictions_df
):

    gt_frames = load_gt_frames(
        gt_file
    )

    pred_frames = load_prediction_frames(
        predictions_df
    )

    all_frame_ids = sorted(
        set(gt_frames.keys())
        |
        set(pred_frames.keys())
    )

    frames = []

    for frame_id in all_frame_ids:

        frames.append({
            "ground_truths":
                gt_frames.get(
                    frame_id,
                    []
                ),

            "predictions":
                pred_frames.get(
                    frame_id,
                    []
                )
        })

    return frames