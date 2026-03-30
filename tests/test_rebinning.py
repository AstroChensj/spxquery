from pathlib import Path

import pandas as pd

from spxquery.core.config import PhotometryConfig, PhotometryResult, Source
from spxquery.processing.rebinning import (
    generate_binned_lightcurve_dataframe,
    get_binned_output_paths,
    get_binned_output_suffix,
    load_or_generate_binned_photometry,
)


def _result(
    obs_id: str,
    mjd: float,
    flux: float,
    flux_error: float,
    wavelength: float,
    bandwidth: float,
    band: str,
    flag: int = 0,
) -> PhotometryResult:
    return PhotometryResult(
        obs_id=obs_id,
        mjd=mjd,
        flux=flux,
        flux_error=flux_error,
        wavelength=wavelength,
        bandwidth=bandwidth,
        flag=flag,
        pix_x=10.0,
        pix_y=20.0,
        band=band,
    )


def test_generate_binned_lightcurve_dataframe_combines_by_band_time_and_wavelength():
    source = Source(ra=1.0, dec=2.0, name="test")
    config = PhotometryConfig(enable_binned_photometry=True, time_bin_days=1.0)
    results = [
        _result("a", 100.1, 10.0, 1.0, 1.00, 0.10, "D1"),
        _result("b", 100.2, 20.0, 2.0, 1.04, 0.10, "D1"),
        _result("c", 100.3, 30.0, 1.0, 1.18, 0.10, "D1"),
        _result("d", 100.2, 99.0, 1.0, 1.02, 0.10, "D2"),
    ]

    df = generate_binned_lightcurve_dataframe(results, source, config)

    assert len(df) == 3

    combined = df[(df["band"] == "D1") & (df["n_used_in_bin"] == 2)].iloc[0]
    assert combined["flux"] == 12.0
    assert round(combined["flux_error"], 6) == round((1.0 / 1.25) ** 0.5, 6)
    assert combined["n_total_in_bin"] == 2
    assert combined["n_flag_rejected_in_bin"] == 0


def test_generate_binned_lightcurve_dataframe_drops_only_bad_flags():
    source = Source(ra=1.0, dec=2.0, name="test")
    config = PhotometryConfig(enable_binned_photometry=True, time_bin_days=1.0, bad_flags=[0])
    results = [
        _result("good", 100.1, 10.0, 10.0, 1.00, 0.10, "D1", flag=0),
        _result("upper", 100.2, 5.0, 10.0, 1.02, 0.10, "D1", flag=0),
        _result("flagged", 100.3, 100.0, 1.0, 1.04, 0.10, "D1", flag=1),
    ]

    df = generate_binned_lightcurve_dataframe(results, source, config)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["n_total_in_bin"] == 3
    assert row["n_used_in_bin"] == 2
    assert row["n_flag_rejected_in_bin"] == 1
    assert bool(row["is_upper_limit"]) is True


def test_load_or_generate_binned_photometry_reuses_saved_csv(tmp_path):
    source = Source(ra=1.0, dec=2.0, name="test")
    config = PhotometryConfig(enable_binned_photometry=True, time_bin_days=1.0)
    results = [
        _result("a", 100.1, 10.0, 1.0, 1.00, 0.10, "D1"),
        _result("b", 100.2, 20.0, 2.0, 1.04, 0.10, "D1"),
    ]
    output_csv = tmp_path / "lightcurve_binned_t1d_w1.csv"

    df1, photometry1 = load_or_generate_binned_photometry(
        source=source,
        photometry_config=config,
        raw_photometry_results=results,
        output_csv_path=output_csv,
    )

    assert output_csv.exists()
    assert len(df1) == 1
    assert len(photometry1) == 1

    df2, photometry2 = load_or_generate_binned_photometry(
        source=source,
        photometry_config=config,
        output_csv_path=output_csv,
    )

    assert len(df2) == 1
    assert len(photometry2) == 1
    saved = pd.read_csv(output_csv, comment="#")
    assert saved.iloc[0]["n_used_in_bin"] == 2


def test_get_binned_output_suffix_and_paths():
    config = PhotometryConfig(enable_binned_photometry=True, time_bin_days=50.0, wavelength_bin_scale=1.5)

    assert get_binned_output_suffix(config) == "binned_t50d_w1.5"

    csv_path, png_path = get_binned_output_paths(Path("/tmp"), config)
    assert str(csv_path).endswith("lightcurve_binned_t50d_w1.5.csv")
    assert str(png_path).endswith("combined_plot_binned_t50d_w1.5.png")


def test_different_binning_configs_use_different_output_filenames(tmp_path):
    config_a = PhotometryConfig(enable_binned_photometry=True, time_bin_days=7.0, wavelength_bin_scale=1.0)
    config_b = PhotometryConfig(enable_binned_photometry=True, time_bin_days=100.0, wavelength_bin_scale=1.0)

    csv_a, png_a = get_binned_output_paths(tmp_path, config_a)
    csv_b, png_b = get_binned_output_paths(tmp_path, config_b)

    assert csv_a != csv_b
    assert png_a != png_b
