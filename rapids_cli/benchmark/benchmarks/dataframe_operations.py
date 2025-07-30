# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DataFrame operations benchmark comparing pandas vs cuDF."""

import time

import numpy as np


def _prepare_join_data(n_left=5_000_000, n_right=2_000_000):
    """Prepare test data for join operations."""
    np.random.seed(42)

    # Left dataset (customer transactions)
    left_data = {
        "customer_id": np.random.randint(0, n_right // 2, n_left),
        "transaction_id": np.arange(n_left),
        "amount": np.random.uniform(10.0, 1000.0, n_left),
        "category": np.random.choice(
            ["grocery", "entertainment", "gas", "retail"], n_left
        ),
        "region": np.random.choice(["north", "south", "east", "west"], n_left),
        "timestamp": np.random.randint(1000000, 2000000, n_left),
    }

    # Right dataset (customer information)
    right_data = {
        "customer_id": np.arange(n_right),
        "age": np.random.randint(18, 80, n_right),
        "income": np.random.uniform(30000, 150000, n_right),
        "credit_score": np.random.randint(300, 850, n_right),
        "segment": np.random.choice(["premium", "standard", "basic", "new"], n_right),
        "loyalty_years": np.random.randint(0, 20, n_right),
    }

    return left_data, right_data


def _dataframe_join_operations_cpu(left_data, right_data, verbose=False):
    """CPU implementation using pandas. Returns execution time."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for this benchmark") from e

    if verbose:
        print("Running pandas (CPU) implementation...")

    # Start timing
    start_time = time.time()

    # Create DataFrames
    left_df = pd.DataFrame(left_data)
    right_df = pd.DataFrame(right_data)

    # Perform joins
    inner_join = left_df.merge(right_df, on="customer_id", how="inner")
    left_join = left_df.merge(right_df, on="customer_id", how="left")

    # Aggregations on inner join
    inner_agg = (
        inner_join.groupby(["category", "segment", "region"])
        .agg(
            {
                "amount": ["sum", "mean", "count"],
                "age": "mean",
                "income": ["median", "std"],
                "credit_score": ["mean", "min", "max"],
                "loyalty_years": "mean",
            }
        )
        .reset_index()
    )

    # Process left join
    left_join["has_profile"] = left_join["age"].notna()
    left_join["amount_bucket"] = pd.cut(
        left_join["amount"],
        bins=5,
        labels=["very_low", "low", "medium", "high", "very_high"],
    )

    # Aggregations on left join
    left_agg = (
        left_join.groupby(["category", "region", "has_profile"])
        .agg(
            {
                "amount": ["sum", "mean", "count"],
                "credit_score": "mean",
                "transaction_id": "count",
            }
        )
        .reset_index()
    )

    # Additional operations
    inner_agg.columns = ["_".join(col).strip("_") for col in inner_agg.columns]
    left_agg.columns = ["_".join(col).strip("_") for col in left_agg.columns]

    # Derived metrics
    inner_agg["avg_amount_per_loyalty_year"] = inner_agg["amount_mean"] / (
        inner_agg["loyalty_years_mean"] + 1
    )
    inner_agg["credit_score_range"] = (
        inner_agg["credit_score_max"] - inner_agg["credit_score_min"]
    )

    # Final processing
    final_inner = inner_agg.sort_values(
        ["amount_sum", "credit_score_mean"], ascending=[False, False]
    ).head(1000)
    final_left = left_agg.sort_values(
        ["amount_sum", "transaction_id_count"], ascending=[False, False]
    ).head(1000)

    # Stop timing
    cpu_time = time.time() - start_time

    if verbose:
        print(
            f"Pandas completed in {cpu_time:.3f}s, results: {final_inner.shape}, {final_left.shape}"
        )

    return cpu_time


def _dataframe_join_operations_gpu(left_data, right_data, verbose=False):
    """GPU implementation using cuDF. Returns execution time (excludes data transfer)."""
    try:
        import cudf
    except ImportError as e:
        raise ImportError("cudf is required for GPU benchmarking") from e

    if verbose:
        print("Running cuDF (GPU) implementation...")
        print("Transferring data to GPU (not timed)...")

    # Data transfer to GPU (NOT TIMED)
    left_df_gpu = cudf.DataFrame(left_data)
    right_df_gpu = cudf.DataFrame(right_data)

    # Warmup run (NOT TIMED - ensures GPU is ready)
    if verbose:
        print("GPU warmup...")
    _ = left_df_gpu.head(100).merge(
        right_df_gpu.head(100), on="customer_id", how="inner"
    )

    # Start timing (computation only)
    if verbose:
        print("Starting GPU computation timing...")
    start_time = time.time()

    # Perform joins
    inner_join_gpu = left_df_gpu.merge(right_df_gpu, on="customer_id", how="inner")
    left_join_gpu = left_df_gpu.merge(right_df_gpu, on="customer_id", how="left")

    # Aggregations on inner join
    inner_agg_gpu = (
        inner_join_gpu.groupby(["category", "segment", "region"])
        .agg(
            {
                "amount": ["sum", "mean", "count"],
                "age": "mean",
                "income": ["median", "std"],
                "credit_score": ["mean", "min", "max"],
                "loyalty_years": "mean",
            }
        )
        .reset_index()
    )

    # Process left join
    left_join_gpu["has_profile"] = left_join_gpu["age"].notna()
    left_join_gpu["amount_bucket"] = (left_join_gpu["amount"] // 200).astype(int)

    # Aggregations on left join
    left_agg_gpu = (
        left_join_gpu.groupby(["category", "region", "has_profile"])
        .agg(
            {
                "amount": ["sum", "mean", "count"],
                "credit_score": "mean",
                "transaction_id": "count",
            }
        )
        .reset_index()
    )

    # Additional operations
    inner_agg_gpu.columns = ["_".join(col).strip("_") for col in inner_agg_gpu.columns]
    left_agg_gpu.columns = ["_".join(col).strip("_") for col in left_agg_gpu.columns]

    # Derived metrics
    inner_agg_gpu["avg_amount_per_loyalty_year"] = inner_agg_gpu["amount_mean"] / (
        inner_agg_gpu["loyalty_years_mean"] + 1
    )
    inner_agg_gpu["credit_score_range"] = (
        inner_agg_gpu["credit_score_max"] - inner_agg_gpu["credit_score_min"]
    )

    # Final processing
    final_inner_gpu = inner_agg_gpu.sort_values(
        ["amount_sum", "credit_score_mean"], ascending=[False, False]
    ).head(1000)
    final_left_gpu = left_agg_gpu.sort_values(
        ["amount_sum", "transaction_id_count"], ascending=[False, False]
    ).head(1000)

    # Stop timing
    gpu_time = time.time() - start_time

    if verbose:
        print(
            f"cuDF computation completed in {gpu_time:.3f}s, results: {final_inner_gpu.shape}, {final_left_gpu.shape}"
        )

    return gpu_time


def dataframe_join_operations(verbose: bool = False) -> tuple[float, float]:
    """Benchmark DataFrame join and merge operations with aggregations."""
    # Prepare data once
    left_data, right_data = _prepare_join_data()

    if verbose:
        print(
            f"Created join test data: {len(left_data['customer_id']):,} x {len(right_data['customer_id']):,} rows"
        )

    # Run CPU implementation
    cpu_time = _dataframe_join_operations_cpu(left_data, right_data, verbose)

    # Run GPU implementation
    gpu_time = _dataframe_join_operations_gpu(left_data, right_data, verbose)

    if verbose:
        print(
            f"Final comparison - CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, Speedup: {cpu_time/gpu_time:.1f}x"
        )

    return cpu_time, gpu_time
