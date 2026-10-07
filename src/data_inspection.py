"""Dataset inspection script for 151673.h5ad.

Reports shape, obs/var columns, obsm/uns keys, spatial coordinates,
image availability, raw availability, and preprocessing state.
Saves report to outputs/data_inspection/report.json.
"""

import os
import json
import numpy as np
import scanpy as sc


def inspect_dataset(h5ad_path: str, output_path: str = "outputs/data_inspection/report.json") -> dict:
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"File not found: {h5ad_path}")

    adata = sc.read_h5ad(h5ad_path)

    report = {
        "dataset_path": os.path.abspath(h5ad_path),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns": list(adata.obs.columns),
        "var_columns": list(adata.var.columns),
        "obsm_keys": list(adata.obsm.keys()),
        "layers": list(adata.layers.keys()),
        "uns_keys": list(adata.uns.keys()),
        "raw_available": adata.raw is not None,
    }

    # Spatial coordinates
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"]
        report["spatial_coordinates"] = {
            "key": "obsm['spatial']",
            "shape": list(coords.shape),
            "dtype": str(coords.dtype),
            "x_min": float(np.min(coords[:, 0])),
            "x_max": float(np.max(coords[:, 0])),
            "y_min": float(np.min(coords[:, 1])),
            "y_max": float(np.max(coords[:, 1])),
            "sample_coords": coords[:5].tolist()
        }
    else:
        report["spatial_coordinates"] = None

    # Images
    if "spatial" in adata.uns:
        img_info = {}
        for sample_id, s_data in adata.uns["spatial"].items():
            img_info[sample_id] = {
                "available_images": list(s_data.get("images", {}).keys()),
                "scalefactors": list(s_data.get("scalefactors", {}).keys()),
            }
        report["image_availability"] = {
            "has_images": True,
            "details": img_info
        }
    else:
        report["image_availability"] = {
            "has_images": False,
            "details": None
        }

    # Preprocessing inspection
    sample_sub = adata.X[:100].toarray() if hasattr(adata.X, "toarray") else adata.X[:100]
    min_val = float(np.min(sample_sub))
    max_val = float(np.max(sample_sub))
    is_integer = bool(np.all(np.equal(np.mod(sample_sub, 1), 0)))
    has_log1p_uns = "log1p" in adata.uns
    has_negative = bool(np.any(sample_sub < 0))

    if is_integer and not has_log1p_uns:
        preprocessing_state = "raw counts"
    elif has_log1p_uns or (not is_integer and min_val >= 0):
        preprocessing_state = "normalized and log1p-transformed"
    elif has_negative:
        preprocessing_state = "scaled / z-scored"
    else:
        preprocessing_state = "continuous / unknown"

    report["preprocessing_state"] = {
        "inferred_state": preprocessing_state,
        "is_integer_sample": is_integer,
        "min_value": min_val,
        "max_value": max_val,
        "has_negative_values": has_negative,
        "uns_has_log1p": has_log1p_uns,
        "uns_has_hvg": "hvg" in adata.uns,
        "matrix_format": str(type(adata.X))
    }

    # Obs summary
    report["obs_summary"] = {
        col: {
            "dtype": str(adata.obs[col].dtype),
            "num_unique": int(adata.obs[col].nunique()),
            "sample_values": [str(x) for x in adata.obs[col].dropna().unique()[:5]]
        }
        for col in adata.obs.columns
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Data inspection report successfully saved to {output_path}")
    return report


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "151673.h5ad"
    if not os.path.exists(path) and os.path.exists(os.path.expanduser("~/151673.h5ad")):
        path = os.path.expanduser("~/151673.h5ad")
    inspect_dataset(path)
