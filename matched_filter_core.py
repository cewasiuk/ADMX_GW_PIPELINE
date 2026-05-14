# """
# matched_filter_core.py
# ----------------------
# Matched filtering and statistical analysis for the ADMX GW search pipeline.

# Pipeline position
# -----------------
#     hr_raw_prep.py          →  *_prepared.h5   (V_flat per scan)
#     psd_computation.py      →  *_psd.h5        (stacked noise PSD)
#     create_waveform_template.py → V_template   (voltage template)
#     matched_filter_core.py  →  SNR time series, candidate list, strain limits

# Pretty sure a lot of these functions are legacy and dont get called, but the pipeline works so if its not broken dont fix it
# """

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy import stats
from pycbc.filter import matched_filter_core as _pycbc_mf_core
from pycbc.psd import interpolate as _pycbc_interp
from pycbc.types import FrequencySeries, TimeSeries

from admx_db_datatypes import ScanParameters
from hr_raw_prep import load_prepared_scan
from psd_computation import load_psd_h5, ensure_psd_on_grid
from create_waveform_template import (
    build_template_for_scan,
    build_template_bank,
    snr_threshold_to_strain,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default analysis parameters
# ---------------------------------------------------------------------------
DEFAULT_PSD_FLOOR    = 1e-60   # prevents divide-by-zero in whitening
DEFAULT_SNR_THRESH   = 5.0     # initial candidate threshold (Rayleigh units)
DEFAULT_SIGMA_GAUSS  = 3.0     # Gaussian-equivalent sigma for strain limit


# ---------------------------------------------------------------------------
# Template normalization
# ---------------------------------------------------------------------------

def normalize_template(
    template_fd: FrequencySeries,
    psd_fd: FrequencySeries,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
    psd_floor: float = DEFAULT_PSD_FLOOR,
) -> tuple[FrequencySeries, float]:
    df = float(template_fd.delta_f)
    freqs = np.arange(len(template_fd), dtype=np.float64) * df

    psd_arr = np.asarray(psd_fd.numpy(), dtype=np.float64)
    psd_arr = np.where(np.isfinite(psd_arr), psd_arr, psd_floor)
    psd_arr = np.maximum(psd_arr, psd_floor)

    h = np.asarray(template_fd.numpy(), dtype=np.complex128)

    band = np.ones(len(freqs), dtype=bool)
    if f_low is not None:
        band &= freqs >= float(f_low)
    if f_high is not None:
        band &= freqs <= float(f_high)

    norm2 = 4.0 * df * np.sum((np.abs(h[band]) ** 2) / psd_arr[band])

    if not np.isfinite(norm2) or norm2 <= 0:
        raise ValueError(
            f"Template has zero or non-finite inner product (norm²={norm2:.3e}). "
            "Check that the template has power in the band [f_low, f_high] and "
            "that the PSD is finite and positive."
        )

    sigma = float(np.sqrt(norm2))
    h_normed = h / sigma
    return FrequencySeries(h_normed, delta_f=df), sigma


# ---------------------------------------------------------------------------
# Core single-template matched filter
# ---------------------------------------------------------------------------

def _safe_to_timeseries(snr_obj) -> TimeSeries:
    """Extract TimeSeries from pyCBC matched_filter_core output robustly."""
    # pyCBC >= 2.x returns (snr_ts, corr, norm); older versions return snr_ts directly
    if isinstance(snr_obj, tuple):
        snr_obj = snr_obj[0]
    out = snr_obj.to_timeseries() if hasattr(snr_obj, "to_timeseries") else snr_obj
    if isinstance(out, tuple):
        out = out[0]
    return out

def matched_filter_single(
    data_fd, template_fd, psd_fd,
    f_low=None, f_high=None,
    psd_floor=DEFAULT_PSD_FLOOR,
    normalize=True,
):
    df = float(data_fd.delta_f)
    N  = len(data_fd)

    psd_on = ensure_psd_on_grid(psd_fd, delta_f_hz=df, n_freq=N, psd_floor=psd_floor)

    sigma = 0.0
    tmpl  = template_fd
    if normalize:
        tmpl, sigma = normalize_template(
            template_fd, psd_on, f_low=f_low, f_high=f_high, psd_floor=psd_floor
        )

    raw = _pycbc_mf_core(
        tmpl,
        data_fd,
        psd=psd_on,
        low_frequency_cutoff=f_low,
        high_frequency_cutoff=f_high,
    )

    # pyCBC matched_filter_core returns (snr_series, corr, norm)
    # snr_series is the RAW convolution, must divide by norm to get SNR
    if isinstance(raw, tuple):
        snr_raw, corr, pycbc_norm = raw
        snr_ts = snr_raw.to_timeseries()
        # divide by pyCBC's internal norm to get true SNR units
        snr_arr = np.asarray(snr_ts.numpy(), dtype=np.complex128) * float(pycbc_norm)
        snr_ts  = TimeSeries(snr_arr, delta_t=float(snr_ts.delta_t))
    else:
        snr_ts = raw.to_timeseries() if hasattr(raw, 'to_timeseries') else raw

    snr_abs = np.abs(np.asarray(snr_ts.numpy()))
    i_peak  = int(np.argmax(snr_abs))

    peak = {
        "snr_peak":         float(snr_abs[i_peak]),
        "snr_peak_complex": complex(np.asarray(snr_ts.numpy())[i_peak]),
        "snr_re_at_peak":   float(np.real(np.asarray(snr_ts.numpy())[i_peak])),
        "snr_im_at_peak":   float(np.imag(np.asarray(snr_ts.numpy())[i_peak])),
        "t_peak_sec":       float(i_peak * float(snr_ts.delta_t)),
        "idx_peak":         i_peak,
        "delta_t":          float(snr_ts.delta_t),
        "delta_f":          df,
        "sigma":            sigma,
    }

    return snr_ts, sigma, peak


# ---------------------------------------------------------------------------
# Template bank search over a single scan
# ---------------------------------------------------------------------------

@dataclass
class BankResult:
    """Result of running a template bank against one scan."""
    scan_file:       str
    best_snr:        float
    best_mass1:      float
    best_mass2:      float
    best_t_peak_sec: float
    best_snr_complex: complex
    all_peaks:       list[dict] = field(default_factory=list)
    scan_params:     Optional[ScanParameters] = None


def matched_filter_bank(
    data_fd: FrequencySeries,
    template_bank: list[dict],
    psd_fd: FrequencySeries,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
    psd_floor: float = DEFAULT_PSD_FLOOR,
    scan_label: str = "",
    scan_params: Optional[ScanParameters] = None,
) -> BankResult:
    peaks = []
    best_snr   = 0.0
    best_idx   = 0

    for i, tmpl_dict in enumerate(template_bank):
        m1, m2 = tmpl_dict.get("mass_pair", (0.0, 0.0))
        pycbc_tmpl = tmpl_dict["pycbc_template"]

        try:
            _, sigma, pk = matched_filter_single(
                data_fd=data_fd,
                template_fd=pycbc_tmpl,
                psd_fd=psd_fd,
                f_low=f_low,
                f_high=f_high,
                psd_floor=psd_floor,
                normalize=True,
            )
            pk["mass1"] = m1
            pk["mass2"] = m2
            peaks.append(pk)

            if pk["snr_peak"] > best_snr:
                best_snr = pk["snr_peak"]
                best_idx = i

        except Exception as exc:
            log.warning(
                "Bank filter failed for (%.2f, %.2f) Msun: %s", m1, m2, exc
            )

    if not peaks:
        raise RuntimeError(f"All templates failed for scan '{scan_label}'")

    bp = peaks[best_idx]
    return BankResult(
        scan_file=scan_label,
        best_snr=bp["snr_peak"],
        best_mass1=bp["mass1"],
        best_mass2=bp["mass2"],
        best_t_peak_sec=bp["t_peak_sec"],
        best_snr_complex=bp["snr_peak_complex"],
        all_peaks=peaks,
        scan_params=scan_params,
    )


# ---------------------------------------------------------------------------
# SNR statistics and candidate identification
# ---------------------------------------------------------------------------

@dataclass
class SNRDistribution:
    """Population of peak SNR values from many scans."""
    snr_values:   np.ndarray     # one value per scan (best across bank)
    scan_labels:  list[str]
    n_samples:    int            # total independent time samples searched
    f_low:        Optional[float]
    f_high:       Optional[float]

    # fitted Rayleigh parameter (= 1.0 for unit-normalized Gaussian noise)
    rayleigh_sigma: float = 1.0
    is_gaussian:    bool  = True
    ks_pvalue:      float = 1.0


def compute_snr_threshold(
    n_independent_samples: int,
    target_false_alarms: float = 1.0,
) -> float:
    if n_independent_samples <= 0:
        raise ValueError("n_independent_samples must be positive")
    p_fa = float(target_false_alarms) / float(n_independent_samples)
    p_fa = np.clip(p_fa, 1e-300, 1.0)
    return float(np.sqrt(-2.0 * np.log(p_fa)))


def analyze_snr_distribution(
    snr_values: np.ndarray,
    scan_labels: Optional[list[str]] = None,
    n_time_samples_per_scan: int = 1,
    snr_threshold: Optional[float] = None,
    target_false_alarms: float = 1.0,
    plot: bool = False,
    save_plot: Optional[Path] = None,
) -> dict:
    snr_values = np.asarray(snr_values, dtype=np.float64)
    n_scans = len(snr_values)
    n_total = n_scans * n_time_samples_per_scan

    if snr_threshold is None:
        snr_threshold = compute_snr_threshold(n_total, target_false_alarms)

    # fit Rayleigh distribution (σ_fit should be ≈ 1 for unit-normalized noise)
    # Rayleigh σ = E[|z|] / sqrt(π/2); MLE: σ² = mean(|z|²) / 2
    rayleigh_sigma = float(np.sqrt(np.mean(snr_values ** 2) / 2.0))

    # KS test against Rayleigh(σ=1) — the expected distribution
    rayleigh_rvs = snr_values / rayleigh_sigma    # normalize to σ=1
    # Rayleigh with σ=1 has CDF 1 - exp(-x²/2)
    ks_stat, ks_pval = stats.kstest(
        rayleigh_rvs,
        lambda x: 1.0 - np.exp(-x**2 / 2.0),
    )
    is_gaussian = bool(ks_pval > 0.05)

    if not is_gaussian:
        log.warning(
            "SNR distribution fails KS test (p=%.4f < 0.05). "
            "Non-Gaussian noise suspected. Candidates may be unreliable.",
            ks_pval,
        )
    else:
        log.info("SNR distribution consistent with Gaussian (KS p=%.3f)", ks_pval)

    if not np.isclose(rayleigh_sigma, 1.0, rtol=0.1):
        log.warning(
            "Fitted Rayleigh σ = %.3f (expected 1.0 ± 10%%). "
            "Check template normalization or PSD calibration.",
            rayleigh_sigma,
        )

    # candidates
    cand_idx  = [i for i, ρ in enumerate(snr_values) if ρ > snr_threshold]
    cand_snrs = [float(snr_values[i]) for i in cand_idx]
    cand_lbls = [scan_labels[i] if scan_labels else str(i) for i in cand_idx]

    log.info(
        "Candidates above ρ>%.2f: %d / %d scans",
        snr_threshold, len(cand_idx), n_scans,
    )

    if plot or save_plot:
        _plot_snr_distribution(
            snr_values, snr_threshold, rayleigh_sigma,
            ks_pval, cand_idx, plot, save_plot,
        )

    return {
        "snr_threshold":      snr_threshold,
        "n_candidates":       len(cand_idx),
        "candidate_indices":  cand_idx,
        "candidate_snrs":     cand_snrs,
        "candidate_labels":   cand_lbls,
        "rayleigh_sigma_fit": rayleigh_sigma,
        "ks_statistic":       float(ks_stat),
        "ks_pvalue":          float(ks_pval),
        "is_gaussian":        is_gaussian,
        "n_scans":            n_scans,
        "n_samples_total":    n_total,
    }


def _plot_snr_distribution(
    snr_values, threshold, rayleigh_sigma, ks_pval,
    cand_idx, show, save_path,
):
    import matplotlib.pyplot as plt
    from scipy.special import erfc

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"SNR distribution  (KS p={ks_pval:.3f}, "
        f"Rayleigh σ_fit={rayleigh_sigma:.3f}, "
        f"threshold={threshold:.2f})",
        fontsize=11,
    )

    rho_max = max(float(np.max(snr_values)) * 1.1, threshold * 1.2)
    rho_grid = np.linspace(0, rho_max, 300)

    # histogram
    ax = axes[0]
    ax.hist(snr_values, bins=40, density=True, alpha=0.65, label="Data")
    # expected Rayleigh(σ=1) pdf
    rayleigh_pdf = rho_grid * np.exp(-rho_grid**2 / 2.0)
    ax.plot(rho_grid, rayleigh_pdf, "C1--", lw=1.5, label="Rayleigh(σ=1) expected")
    ax.axvline(threshold, color="C3", lw=1.5, ls=":", label=f"Threshold ρ={threshold:.2f}")
    if cand_idx:
        ax.scatter(
            [snr_values[i] for i in cand_idx],
            np.zeros(len(cand_idx)),
            color="red", zorder=5, s=60, label=f"{len(cand_idx)} candidate(s)",
        )
    ax.set_xlabel("|z| (SNR)")
    ax.set_ylabel("Probability density")
    ax.set_title("SNR histogram")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # complementary CDF (log scale — easier to see tails)
    ax = axes[1]
    sorted_snr = np.sort(snr_values)[::-1]
    ecdf_x = np.concatenate([[rho_max], sorted_snr, [0]])
    ecdf_y = np.concatenate([[0], np.arange(1, len(sorted_snr)+1)/len(sorted_snr), [1]])
    ax.semilogy(ecdf_x, ecdf_y, label="Empirical 1-CDF")
    # Rayleigh(σ=1) survival function
    ax.semilogy(rho_grid, np.exp(-rho_grid**2 / 2.0), "C1--", lw=1.5,
                label="Rayleigh(σ=1) expected")
    ax.axvline(threshold, color="C3", lw=1.5, ls=":", label=f"Threshold")
    ax.set_xlabel("|z| (SNR)")
    ax.set_ylabel("P(|z| > ρ)   [log]")
    ax.set_title("Survival function (log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved SNR distribution plot → %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Strain exclusion limits
# ---------------------------------------------------------------------------

def compute_strain_limits(
    scan_results: list[BankResult],
    psd_h5: Path,
    snr_threshold: float,
) -> list[dict]:
    from create_waveform_template import berlin_voltage_scale

    limits = []
    for result in scan_results:
        sp = result.scan_params
        is_cand = result.best_snr > snr_threshold

        h0_limit = None
        if sp is not None and not is_cand:
            try:
                h0_limit = snr_threshold_to_strain(snr_threshold, sp, sp.f0_hz)
            except Exception as exc:
                log.warning(
                    "strain limit failed for %s: %s", result.scan_file, exc
                )

        limits.append({
            "scan_file":       result.scan_file,
            "h0_limit_at_f0":  h0_limit,
            "f0_hz":           sp.f0_hz if sp else None,
            "snr_peak":        result.best_snr,
            "is_candidate":    is_cand,
        })

    return limits


# ---------------------------------------------------------------------------
# Synthetic injection and recovery test
# ---------------------------------------------------------------------------
def make_synthetic_injection(
    template_fd: FrequencySeries,
    psd_fd: FrequencySeries,
    target_snr: float = 10.0,
    t_inject_sec: float = -20.0,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
    psd_floor: float = DEFAULT_PSD_FLOOR,
    seed: Optional[int] = None,
) -> tuple[FrequencySeries, FrequencySeries, FrequencySeries, FrequencySeries, dict]:
    df = float(template_fd.delta_f)
    N  = len(template_fd)

    # ── PSD on grid ───────────────────────────────────────────────────────
    psd_on  = ensure_psd_on_grid(psd_fd, delta_f_hz=df, n_freq=N, psd_floor=psd_floor)
    psd_arr = np.asarray(psd_on.numpy(), dtype=np.float64)

    # ── Normalize template ────────────────────────────────────────────────
    # tmpl_norm satisfies (tmpl_norm | tmpl_norm) = 1
    # sigma is sqrt((template | template)) in the noise-weighted inner product
    tmpl_norm, sigma = normalize_template(
        template_fd, psd_on, f_low=f_low, f_high=f_high, psd_floor=psd_floor
    )
    tmpl_arr = np.asarray(tmpl_norm.numpy(), dtype=np.complex128)

    # ── Signal: target_snr * tmpl_norm * phase_shift ──────────────────────
    # No extra division by sigma — tmpl_norm is already unit-normalized.
    # The matched filter against tmpl_norm returns exactly target_snr at t_inject.
    freqs      = np.arange(N, dtype=np.float64) * df
    phase_shift = np.exp(-2j * np.pi * freqs * float(t_inject_sec))
    signal_arr  = float(target_snr) * tmpl_arr * phase_shift
    signal_fd   = FrequencySeries(signal_arr, delta_f=df)

    # ── Noise: CN(0, S_n(f) / (4 Δf)) ───────────────────────────────────

    rng         = np.random.default_rng(seed)
    std_per_bin = np.sqrt(psd_arr / (4.0 * df))   # correct convention
    noise_re    = rng.standard_normal(N) * std_per_bin
    noise_im    = rng.standard_normal(N) * std_per_bin
    noise_arr   = (noise_re + 1j * noise_im).astype(np.complex128)
    noise_arr[0]  = noise_arr[0].real   # DC must be real
    noise_arr[-1] = noise_arr[-1].real  # Nyquist must be real
    noise_fd    = FrequencySeries(noise_arr, delta_f=df)

    # ── Data ──────────────────────────────────────────────────────────────
    data_arr = noise_arr + signal_arr
    data_fd  = FrequencySeries(data_arr, delta_f=df)

    info = {
        "sigma":        sigma,
        "t_inject_sec": float(t_inject_sec),
        "target_snr":   target_snr,
        "seed":         seed,
        "N":            N,
        "df":           df,
    }

    return data_fd, noise_fd, signal_fd, psd_on, info
# ---------------------------------------------------------------------------
# Full pipeline runner for a nibble
# ---------------------------------------------------------------------------

def run_matched_filter_nibble(
    prepared_dir: str | Path,
    psd_h5: str | Path,
    run_definition_path: str = "run1b_definitions.yaml",
    approximant: str = "TaylorF2",
    mass_grid: Optional[list[tuple[float, float]]] = None,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
    psd_floor: float = DEFAULT_PSD_FLOOR,
    snr_threshold: Optional[float] = None,
    target_false_alarms: float = 1.0,
    output_h5: Optional[str | Path] = None,
    plot_distribution: bool = False,
    save_plot: Optional[str] = None,
) -> dict:
    prepared_dir = Path(prepared_dir)
    psd_h5       = Path(psd_h5)

    if mass_grid is None:
        mass_grid = [(1.4, 1.4)]

    # load PSD
    psd_fd, f_bb, f_abs, psd_meta = load_psd_h5(psd_h5)
    log.info(
        "Loaded PSD: %d bins, df=%.4f Hz, n_scans_used=%d",
        len(psd_fd), float(psd_fd.delta_f), psd_meta["n_scans_used"],
    )

    prepared_files = sorted(prepared_dir.glob("*_prepared.h5"))
    if not prepared_files:
        raise FileNotFoundError(f"No *_prepared.h5 in {prepared_dir}")

    scan_results: list[BankResult] = []
    peak_snrs:    list[float]      = []
    scan_labels:  list[str]        = []
    n_time_per_scan = 1

    for prepared_path in prepared_files:
        scan_data = load_prepared_scan(prepared_path)
        if scan_data is None:
            continue    # cut scan

        sp = scan_data["scan_params"]
        v_flat  = scan_data["V_flat"]
        delta_f = scan_data["delta_f_hz"]

        data_fd = FrequencySeries(
            v_flat.astype(np.complex128), delta_f=delta_f
        )

        # build template bank for this scan's frequency grid and scan params
        try:
            bank = build_template_bank(
                scan_params=sp,
                f_baseband_hz=scan_data["f_baseband"],
                f_abs_hz=scan_data["f_abs"],
                delta_f_hz=delta_f,
                mass_grid=mass_grid,
                approximant=approximant,
                apply_cavity=True,
                complex_cavity=True,
            )
        except Exception as exc:
            log.warning("Template bank failed for %s: %s", prepared_path.name, exc)
            continue

        try:
            result = matched_filter_bank(
                data_fd=data_fd,
                template_bank=bank,
                psd_fd=psd_fd,
                f_low=f_low,
                f_high=f_high,
                psd_floor=psd_floor,
                scan_label=prepared_path.name,
                scan_params=sp,
            )
        except Exception as exc:
            log.warning("Matched filter failed for %s: %s", prepared_path.name, exc)
            continue

        scan_results.append(result)
        peak_snrs.append(result.best_snr)
        scan_labels.append(prepared_path.name)

        # estimate independent samples: N_freq bins for a single scan
        n_time_per_scan = max(n_time_per_scan, len(v_flat))

        log.info(
            "%s  peak_SNR=%.3f  (m1=%.2f, m2=%.2f)",
            prepared_path.name, result.best_snr,
            result.best_mass1, result.best_mass2,
        )

    if not peak_snrs:
        raise RuntimeError("No scans produced a matched-filter result.")

    # --- statistical analysis ---
    snr_stats = analyze_snr_distribution(
        snr_values=np.array(peak_snrs),
        scan_labels=scan_labels,
        n_time_samples_per_scan=n_time_per_scan,
        snr_threshold=snr_threshold,
        target_false_alarms=target_false_alarms,
        plot=plot_distribution,
        save_plot=Path(save_plot) if save_plot else None,
    )

    threshold = snr_stats["snr_threshold"]

    # --- strain limits ---
    strain_limits = compute_strain_limits(scan_results, psd_h5, threshold)

    # --- save results ---
    if output_h5:
        _save_results_h5(
            Path(output_h5), scan_results, peak_snrs,
            scan_labels, snr_stats, strain_limits,
        )

    return {
        "scan_results":  scan_results,
        "snr_stats":     snr_stats,
        "strain_limits": strain_limits,
        "n_scans":       len(scan_results),
        "n_candidates":  snr_stats["n_candidates"],
    }


def _save_results_h5(
    out_path: Path,
    scan_results: list[BankResult],
    peak_snrs: list[float],
    scan_labels: list[str],
    snr_stats: dict,
    strain_limits: list[dict],
) -> None:
    """Save matched-filter results to HDF5."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("peak_snrs",   data=np.array(peak_snrs, dtype=np.float64))
        f.create_dataset("scan_labels", data=np.array(scan_labels, dtype=h5py.string_dtype()))

        # best mass pairs
        f.create_dataset("best_mass1", data=np.array([r.best_mass1 for r in scan_results]))
        f.create_dataset("best_mass2", data=np.array([r.best_mass2 for r in scan_results]))
        f.create_dataset("best_t_peak", data=np.array([r.best_t_peak_sec for r in scan_results]))

        # strain limits
        h0_limits = np.array([
            d["h0_limit_at_f0"] if d["h0_limit_at_f0"] is not None else np.nan
            for d in strain_limits
        ])
        f.create_dataset("h0_limits_at_f0", data=h0_limits)
        f.create_dataset("f0_hz", data=np.array([
            d["f0_hz"] if d["f0_hz"] is not None else np.nan
            for d in strain_limits
        ]))
        f.create_dataset("is_candidate", data=np.array([
            int(d["is_candidate"]) for d in strain_limits
        ]))

        # stats
        g = f.require_group("snr_stats")
        for k, v in snr_stats.items():
            if isinstance(v, (int, float, bool)):
                g.attrs[k] = v
            elif isinstance(v, list) and v:
                if isinstance(v[0], (int, float)):
                    g.create_dataset(k, data=np.array(v))

        f.attrs["pipeline_stage"] = "matched_filter_core"
        f.attrs["n_scans"]        = len(scan_results)
        f.attrs["n_candidates"]   = int(snr_stats["n_candidates"])
        f.attrs["snr_threshold"]  = float(snr_stats["snr_threshold"])

    log.info("Saved matched-filter results → %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ADMX matched filter pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prepared_dir", required=True,
                   help="Directory of *_prepared.h5 files from hr_raw_prep.")
    p.add_argument("--psd_h5",       required=True,
                   help="Path to nibble PSD H5 from psd_computation.")
    p.add_argument("-r", "--run_definition", default="run1b_definitions.yaml")
    p.add_argument("--approximant",  default="TaylorF2")
    p.add_argument("--mass_pairs",   default="1.4,1.4",
                   help="Comma-separated m1,m2 pairs.  "
                        "Multiple pairs: '1.0,1.0 1.2,1.4 1.4,1.4'.")
    p.add_argument("--f_low",  type=float, default=None)
    p.add_argument("--f_high", type=float, default=None)
    p.add_argument("--snr_threshold", type=float, default=None)
    p.add_argument("--target_false_alarms", type=float, default=1.0)
    p.add_argument("--output_h5",  default=None)
    p.add_argument("--plot",       action="store_true")
    p.add_argument("--save_plot",  default=None)
    p.add_argument("--log_level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _parse_mass_grid(mass_pairs_str: str) -> list[tuple[float, float]]:
    grid = []
    for pair in mass_pairs_str.split():
        m1, m2 = pair.split(",")
        grid.append((float(m1), float(m2)))
    return grid


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mass_grid = _parse_mass_grid(args.mass_pairs)

    results = run_matched_filter_nibble(
        prepared_dir=args.prepared_dir,
        psd_h5=args.psd_h5,
        run_definition_path=args.run_definition,
        approximant=args.approximant,
        mass_grid=mass_grid,
        f_low=args.f_low,
        f_high=args.f_high,
        snr_threshold=args.snr_threshold,
        target_false_alarms=args.target_false_alarms,
        output_h5=args.output_h5,
        plot_distribution=args.plot,
        save_plot=args.save_plot,
    )

    stats = results["snr_stats"]
    print(f"\nResults: {results['n_scans']} scans, {results['n_candidates']} candidates")
    print(f"  SNR threshold:  {stats['snr_threshold']:.3f}")
    print(f"  Rayleigh σ_fit: {stats['rayleigh_sigma_fit']:.3f}  (expected 1.0)")
    print(f"  KS p-value:     {stats['ks_pvalue']:.4f}  ({'PASS' if stats['is_gaussian'] else 'FAIL'})")
    if results["n_candidates"]:
        print(f"\nCandidates:")
        for lbl, ρ in zip(stats["candidate_labels"], stats["candidate_snrs"]):
            print(f"  {lbl}  ρ={ρ:.3f}")


if __name__ == "__main__":
    main()
