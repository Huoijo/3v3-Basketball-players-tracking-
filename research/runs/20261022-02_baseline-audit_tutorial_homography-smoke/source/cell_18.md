The final `predictions_df` above is used as the single source of truth in all subsequent steps.

- TI-HOTA uses its court position and identity columns.
- The prediction minimap uses the same court position and identity columns.
- The original-video overlay uses the bbox columns from the same rows.

No separate visualization prediction file is created.
