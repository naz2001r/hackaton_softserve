from typing import List

import numpy as np
import pandas as pd


def prep_data4eval(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> List:
    merged_df = pd.merge(
        gt_df, pred_df, on=["query", "top_n"], how="left", suffixes=["_gt", "_pred"]
    )
    merged_df.drop(
        index=merged_df[merged_df["text_page_pred"].isna()].index, inplace=True
    )

    merged_df.sort_values(
        by=["query", "top_n"], ascending=True, na_position="first", inplace=True
    )
    merged_df["labels_gt"] = (
        merged_df["doc_path_gt"]
        + "_"
        + merged_df["text_page_gt"].astype(int).astype(str)
    )
    merged_df["labels_pred"] = (
        merged_df["doc_path_pred"]
        + "_"
        + merged_df["text_page_pred"].astype(int).astype(str)
    )
    queries = merged_df["query"].unique().tolist()
    gt_labels = [
        merged_df[merged_df["query"] == query]["labels_gt"].tolist()
        for query in queries
    ]
    pred_labels = [
        merged_df[merged_df["query"] == query]["labels_pred"].tolist()
        for query in queries
    ]
    return gt_labels, pred_labels


def preproc_gt4eval(query: str) -> str:
    return query.lower().strip()


def average_precision(groundtruth: List, predictions: List, k: int = 5) -> np.float:
    if len(groundtruth) != k:
        groundtruth = groundtruth[:k]
    if len(predictions) != k:
        predictions = predictions[:k]

    score, num_rel = 0.0, 0.0
    for i, p in enumerate(predictions):
        if p in groundtruth and p not in predictions[:i]:
            num_rel += 1
            score += num_rel / (i + 1)
    return score / k


def mean_average_precision(
    groundtruth: pd.DataFrame, predictions: pd.DataFrame, k: int = 5
) -> np.float:
    groundtruth, predictions = prep_data4eval(groundtruth, predictions)
    return np.mean(
        [
            average_precision(gt, p, k if len(gt) == k else len(gt))
            for gt, p in zip(groundtruth, predictions)
        ]
    )
