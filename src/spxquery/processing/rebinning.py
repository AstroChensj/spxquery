"""
Derived photometry rebinning for SPHEREx time-domain analysis.
"""

import logging
from datetime import datetime
from math import floor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.config import PhotometryConfig, PhotometryResult, Source
from ..processing.lightcurve import save_lightcurve_csv
from ..processing.magnitudes import calculate_ab_magnitude_from_jy
from ..utils.helpers import check_flag_bits, create_flag_mask

logger = logging.getLogger(__name__)


def _format_float_for_filename(value: float) -> str:
    """Format numeric config values compactly for filenames."""
    return f"{value:g}"


def get_binned_output_suffix(photometry_config: PhotometryConfig) -> str:
    """Build a stable filename suffix for rebinned products."""
    if photometry_config.time_bin_days is None:
        raise ValueError("time_bin_days must be set to construct rebinned output filenames")

    time_str = _format_float_for_filename(photometry_config.time_bin_days)
    wavelength_str = _format_float_for_filename(photometry_config.wavelength_bin_scale)
    return f"binned_t{time_str}d_w{wavelength_str}"


def get_binned_output_paths(results_dir: Path, photometry_config: PhotometryConfig) -> Tuple[Path, Path]:
    """Return CSV and PNG paths for rebinned products for the given configuration."""
    suffix = get_binned_output_suffix(photometry_config)
    return results_dir / f"lightcurve_{suffix}.csv", results_dir / f"combined_plot_{suffix}.png"


def _valid_weight(result: PhotometryResult) -> bool:
    """Check whether a photometry point can contribute to weighted combination."""
    return np.isfinite(result.flux_error) and result.flux_error > 0 and np.isfinite(result.flux)


def _median_bandwidth_by_band(results: List[PhotometryResult], scale: float) -> Dict[str, float]:
    """Compute default wavelength-bin width per band from native bandwidths."""
    bandwidths_by_band: Dict[str, List[float]] = {}

    for result in results:
        if np.isfinite(result.bandwidth) and result.bandwidth > 0:
            bandwidths_by_band.setdefault(result.band, []).append(result.bandwidth)

    widths = {}
    for band, bandwidths in bandwidths_by_band.items():
        widths[band] = float(np.median(bandwidths)) * scale

    return widths


def _group_binned_results(
    photometry_results: List[PhotometryResult],
    photometry_config: PhotometryConfig,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Group raw photometry into time and wavelength bins.

    Returns
    -------
    Tuple[List[dict], Dict[str, float]]
        Grouped raw result payloads and the wavelength-bin width used per band.
    """
    bad_flags_mask = create_flag_mask(photometry_config.bad_flags)
    accepted = [r for r in photometry_results if not check_flag_bits(r.flag, bad_flags_mask)]

    if not accepted:
        logger.warning("No photometry measurements remain after flag filtering for rebinning")
        return [], {}

    wavelength_widths = _median_bandwidth_by_band(accepted, photometry_config.wavelength_bin_scale)
    if not wavelength_widths:
        logger.warning("Could not determine wavelength bins for rebinned photometry")
        return [], {}

    global_mjd_origin = min(r.mjd for r in accepted)
    wavelength_origins = {
        band: min(r.wavelength for r in accepted if r.band == band and band in wavelength_widths)
        for band in wavelength_widths
    }

    grouped: Dict[Tuple[str, int, int], Dict[str, object]] = {}

    for result in photometry_results:
        band = result.band
        if band not in wavelength_widths:
            continue

        time_bin_index = int(floor((result.mjd - global_mjd_origin) / photometry_config.time_bin_days))
        wavelength_bin_width = wavelength_widths[band]
        wavelength_bin_index = int(floor((result.wavelength - wavelength_origins[band]) / wavelength_bin_width))
        key = (band, time_bin_index, wavelength_bin_index)

        time_bin_start = global_mjd_origin + time_bin_index * photometry_config.time_bin_days
        time_bin_end = time_bin_start + photometry_config.time_bin_days
        wavelength_bin_start = wavelength_origins[band] + wavelength_bin_index * wavelength_bin_width
        wavelength_bin_end = wavelength_bin_start + wavelength_bin_width

        payload = grouped.setdefault(
            key,
            {
                "all_results": [],
                "accepted_results": [],
                "band": band,
                "time_bin_start": time_bin_start,
                "time_bin_end": time_bin_end,
                "wavelength_bin_start": wavelength_bin_start,
                "wavelength_bin_end": wavelength_bin_end,
                "bandwidth": wavelength_bin_width,
            },
        )
        payload["all_results"].append(result)

        if not check_flag_bits(result.flag, bad_flags_mask) and _valid_weight(result):
            payload["accepted_results"].append(result)

    rows = []
    for payload in grouped.values():
        accepted_results = payload["accepted_results"]
        all_results = payload["all_results"]

        if not accepted_results:
            continue

        weights = np.array([1.0 / (r.flux_error**2) for r in accepted_results], dtype=float)
        fluxes = np.array([r.flux for r in accepted_results], dtype=float)
        mjds = np.array([r.mjd for r in accepted_results], dtype=float)
        wavelengths = np.array([r.wavelength for r in accepted_results], dtype=float)

        weight_sum = float(np.sum(weights))
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            continue

        flux = float(np.sum(weights * fluxes) / weight_sum)
        flux_error = float(np.sqrt(1.0 / weight_sum))
        mjd = float(np.sum(weights * mjds) / weight_sum)
        wavelength = float(np.sum(weights * wavelengths) / weight_sum)

        combined_flag = 0
        for result in accepted_results:
            combined_flag |= int(result.flag)

        mag_ab, mag_ab_error = calculate_ab_magnitude_from_jy(flux / 1e6, flux_error / 1e6, wavelength)

        rows.append(
            {
                "obs_id": (
                    f"bin_{payload['band']}_t{payload['time_bin_start']:.6f}"
                    f"_w{payload['wavelength_bin_start']:.6f}"
                ),
                "mjd": mjd,
                "flux": flux,
                "flux_error": flux_error,
                "mag_ab": mag_ab,
                "mag_ab_error": mag_ab_error,
                "wavelength": wavelength,
                "bandwidth": float(payload["bandwidth"]),
                "band": payload["band"],
                "flag": combined_flag,
                "pix_x": float(np.mean([r.pix_x for r in accepted_results])),
                "pix_y": float(np.mean([r.pix_y for r in accepted_results])),
                "n_total_in_bin": len(all_results),
                "n_used_in_bin": len(accepted_results),
                "n_flag_rejected_in_bin": sum(
                    1 for r in all_results if check_flag_bits(r.flag, bad_flags_mask)
                ),
                "time_bin_start": payload["time_bin_start"],
                "time_bin_end": payload["time_bin_end"],
                "wavelength_bin_start": payload["wavelength_bin_start"],
                "wavelength_bin_end": payload["wavelength_bin_end"],
            }
        )

    rows.sort(key=lambda row: (row["mjd"], row["band"], row["wavelength"]))
    logger.info("Generated %d rebinned photometry measurements", len(rows))
    return rows, wavelength_widths


def generate_binned_lightcurve_dataframe(
    photometry_results: List[PhotometryResult],
    source: Source,
    photometry_config: PhotometryConfig,
) -> pd.DataFrame:
    """Generate rebinned lightcurve DataFrame from single-epoch photometry."""
    if not photometry_results:
        logger.warning("No photometry results available for rebinning")
        return pd.DataFrame()

    rows, wavelength_widths = _group_binned_results(photometry_results, photometry_config)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["is_upper_limit"] = df["flux_error"] > df["flux"]
    df["snr"] = np.where(df["flux_error"] > 0, df["flux"] / df["flux_error"], 0.0)

    df.attrs["source_ra"] = source.ra
    df.attrs["source_dec"] = source.dec
    df.attrs["source_name"] = source.name or f"RA{source.ra:.4f}_Dec{source.dec:.4f}"
    df.attrs["generated_at"] = datetime.now().isoformat()
    df.attrs["time_bin_days"] = photometry_config.time_bin_days
    df.attrs["wavelength_bin_scale"] = photometry_config.wavelength_bin_scale
    df.attrs["wavelength_bin_widths"] = wavelength_widths

    return df


def dataframe_to_photometry_results(df: pd.DataFrame) -> List[PhotometryResult]:
    """Convert a rebinned DataFrame into PhotometryResult objects for plotting."""
    photometry_results = []
    if df.empty:
        return photometry_results

    for _, row in df.iterrows():
        mag_ab = row.get("mag_ab")
        mag_ab = None if pd.isna(mag_ab) else float(mag_ab)

        mag_ab_error = row.get("mag_ab_error")
        mag_ab_error = None if pd.isna(mag_ab_error) else float(mag_ab_error)

        photometry_results.append(
            PhotometryResult(
                obs_id=str(row["obs_id"]),
                mjd=float(row["mjd"]),
                flux=float(row["flux"]),
                flux_error=float(row["flux_error"]),
                wavelength=float(row["wavelength"]),
                bandwidth=float(row["bandwidth"]),
                flag=int(row["flag"]),
                pix_x=float(row["pix_x"]),
                pix_y=float(row["pix_y"]),
                band=str(row["band"]),
                mag_ab=mag_ab,
                mag_ab_error=mag_ab_error,
            )
        )

    return photometry_results


def save_binned_lightcurve_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save rebinned lightcurve DataFrame using the standard metadata header."""
    save_lightcurve_csv(df, output_path, include_metadata=True)


def load_or_generate_binned_photometry(
    source: Source,
    photometry_config: PhotometryConfig,
    raw_photometry_results: Optional[List[PhotometryResult]] = None,
    raw_csv_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[PhotometryResult]]:
    """
    Load existing rebinned photometry from CSV or generate it from raw photometry.
    """
    from .lightcurve import load_lightcurve_from_csv

    if output_csv_path is not None and output_csv_path.exists():
        binned_results = load_lightcurve_from_csv(output_csv_path)
        if binned_results:
            df = pd.read_csv(output_csv_path, comment="#")
            return df, binned_results

    if raw_photometry_results is None:
        if raw_csv_path is None or not raw_csv_path.exists():
            logger.warning("Raw lightcurve CSV not found; cannot generate rebinned photometry")
            return pd.DataFrame(), []
        raw_photometry_results = load_lightcurve_from_csv(raw_csv_path)

    if not raw_photometry_results:
        logger.warning("No raw photometry available for rebinned photometry generation")
        return pd.DataFrame(), []

    df = generate_binned_lightcurve_dataframe(raw_photometry_results, source, photometry_config)
    if df.empty:
        return df, []

    if output_csv_path is not None:
        save_binned_lightcurve_csv(df, output_csv_path)

    return df, dataframe_to_photometry_results(df)
