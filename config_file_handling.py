# """
# config_file_handling.py
# -----------------------
# Utilities for loading and querying the ADMX run-definition YAML and
# matching raw data files to their per-scan parameters.
# """

from __future__ import annotations

import os
import re
import glob
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz
import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename pattern
# ---------------------------------------------------------------------------
# Matches: admx_data_2018_05_19_23_24_49_channel_1.dat
#          admx_data_2018_05_19_23_24_49_channel_1_binned.h5   (processed)
_FILENAME_RE = re.compile(
    r"admx_data_"
    r"(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_"
    r"(?P<hour>\d{2})_(?P<min>\d{2})_(?P<sec>\d{2})"
    r"_channel_(?P<channel>\d+)"
)

# UTC offset assumed for ADMX Run 1B filenames (US/Pacific)
_PACIFIC_TZ = pytz.timezone("US/Pacific")


# ---------------------------------------------------------------------------
# parse_filename_timestamp
# ---------------------------------------------------------------------------
def parse_filename_timestamp(filename: str) -> Optional[datetime]:

    stem = Path(filename).stem
    m = _FILENAME_RE.search(stem)
    if not m:
        log.warning("Could not parse timestamp from filename: %s", filename)
        return None

    naive_dt = datetime(
        int(m.group("year")), int(m.group("month")), int(m.group("day")),
        int(m.group("hour")), int(m.group("min")), int(m.group("sec")),
    )
    # Localise to Pacific then convert to UTC
    local_dt = _PACIFIC_TZ.localize(naive_dt)
    return local_dt.astimezone(timezone.utc)


def parse_filename_channel(filename: str) -> Optional[int]:
    """Return the channel number embedded in an ADMX filename, or None."""
    m = _FILENAME_RE.search(Path(filename).stem)
    return int(m.group("channel")) if m else None


# ---------------------------------------------------------------------------
# find_parameter_row  (fixes str.contains substring-match bug)
# ---------------------------------------------------------------------------
def find_parameter_row(df: pd.DataFrame, filename: str) -> Optional[pd.Series]:

    def _normalise(s: str) -> str:
        """Strip path, known extensions, and pipeline suffixes."""
        s = Path(s).name                     # basename only
        for ext in (".dat", ".h5"):
            if s.endswith(ext):
                s = s[: -len(ext)]
        for suffix in ("_binned", "_prepared"):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
        return s

    query_tag = _normalise(filename)

    # Normalise the DataFrame column too (handles tags stored with .dat)
    df_tags_norm = df["Filename_Tag"].astype(str).apply(_normalise)

    matches = df[df_tags_norm == query_tag]
    if len(matches) == 0:
        log.debug("No parameter row for tag '%s' (normalised from '%s')",
                  query_tag, filename)
        return None
    if len(matches) > 1:
        log.warning(
            "Multiple parameter rows matched tag '%s'; using first.", query_tag
        )
    return matches.iloc[0]


# ---------------------------------------------------------------------------
# Cut functions
# ---------------------------------------------------------------------------
def _ensure_aware(dt: datetime, tz=_PACIFIC_TZ) -> datetime:
    """Make a datetime timezone-aware if it is not already."""
    if dt.tzinfo is None:
        return tz.localize(dt)
    return dt


def is_timestamp_cut(
    timestamp_cut_yaml: list, target_time: datetime
) -> tuple[bool, str]:
    target = _ensure_aware(target_time)
    for cut in timestamp_cut_yaml:
        start = _ensure_aware(cut["start_time"])
        stop = _ensure_aware(cut["stop_time"])
        if start < target < stop:
            return True, f"timestamp_cut: {cut['why']}"
    return False, ""


def is_frequency_cut(
    frequency_cut_yaml,
    target_time: datetime,
    target_frequency_hz: float,
) -> tuple[bool, str]:
    # YAML frequency_cuts may be an empty dict when no cuts defined
    if not frequency_cut_yaml or not isinstance(frequency_cut_yaml, list):
        return False, ""

    target = _ensure_aware(target_time)
    for cut in frequency_cut_yaml:
        start = _ensure_aware(cut["start_time"])
        stop = _ensure_aware(cut["stop_time"])
        f_start = float(cut["start_frequency"]) * 1e6  # MHz → Hz
        f_stop = float(cut["stop_frequency"]) * 1e6
        if start < target < stop and f_start < target_frequency_hz < f_stop:
            return True, f"frequency_cut: {cut['why']}"
    return False, ""


# ---------------------------------------------------------------------------
# get_output_path  (replaces get_intermediate_data_file_name)
# ---------------------------------------------------------------------------
def get_output_path(nibble_config: dict, label: str) -> Path:
    data_dir = Path(os.path.expandvars(nibble_config["data_directory"]))
    prefix = nibble_config["file_prefix"]
    return data_dir / f"{prefix}_{label}"


# ---------------------------------------------------------------------------
# glob_hr_dat_files  (new – enables full automation)
# ---------------------------------------------------------------------------
def glob_hr_dat_files(
    hr_root: str | Path,
    nibble_config: Optional[dict] = None,
    channel: int = 1,
) -> list[Path]:
    pattern = str(Path(hr_root) / "**" / f"admx_data_*_channel_{channel}.dat")
    all_files = sorted(Path(p) for p in glob.glob(pattern, recursive=True))

    if nibble_config is None:
        return all_files

    t_start = _ensure_aware(nibble_config["start_time"])
    t_stop = _ensure_aware(nibble_config["stop_time"])

    filtered = []
    for f in all_files:
        ts = parse_filename_timestamp(f.name)
        if ts is None:
            continue
        if t_start <= ts <= t_stop:
            filtered.append(f)

    log.info(
        "glob_hr_dat_files: %d / %d files in nibble window [%s, %s]",
        len(filtered), len(all_files), t_start.isoformat(), t_stop.isoformat(),
    )
    return filtered


# ---------------------------------------------------------------------------
# RunConfig  –  parsed, validated YAML wrapper
# ---------------------------------------------------------------------------

#: Columns expected in every ADMX parameter text file.
PARAMETER_COLUMNS = [
    "Start_Frequency",          # MHz
    "Stop_Frequency",           # MHz
    "Digitizer_Log_ID",
    "Integration_Time",         # seconds
    "Filename_Tag",
    "Quality_Factor",
    "Cavity_Resonant_Frequency", # MHz
    "JPA_SNR",                  # dB
    "Thfet",                    # K
    "Attenuation",              # dB
    "Reflection",               # β (coupling)
    "Transmission",
]


@dataclass
class RunConfig:

    yaml_path: Path
    _raw: dict = field(repr=False)

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "RunConfig":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Run definition not found: {yaml_path}")
        with open(yaml_path, "r") as fh:
            raw = yaml.load(fh, Loader=yaml.Loader)
        return cls(yaml_path=yaml_path, _raw=raw)

    # ---- nibble access --------------------------------------------------
    @property
    def nibble_names(self) -> list[str]:
        return sorted(self._raw.get("nibbles", {}).keys())

    def nibble(self, name: str) -> dict:
        nibbles = self._raw.get("nibbles", {})
        if name not in nibbles:
            raise KeyError(
                f"Nibble '{name}' not found. Available: {list(nibbles.keys())}"
            )
        return nibbles[name]

    # ---- global cuts ----------------------------------------------------
    @property
    def timestamp_cuts(self) -> list:
        return self._raw.get("timestamp_cuts", [])

    @property
    def frequency_cuts(self):
        return self._raw.get("frequency_cuts", {})

    @property
    def parameter_cuts(self) -> dict:
        return self._raw.get("parameter_cuts", {})

    # ---- helpers --------------------------------------------------------
    def b_field(self, nibble_name: str) -> float:
        """Magnetic field [T] for the given nibble."""
        return float(self.nibble(nibble_name)["Bfield"])

    def dat_files(
        self,
        nibble_name: str,
        hr_root: str | Path,
        channel: int = 1,
    ) -> list[Path]:
        return glob_hr_dat_files(
            hr_root,
            nibble_config=self.nibble(nibble_name),
            channel=channel,
        )

    def load_parameter_df(
        self,
        nibble_name: str,
        param_date: str,
    ) -> pd.DataFrame:
        nibble_cfg = self.nibble(nibble_name)
        param_path = get_output_path(nibble_cfg, param_date + ".txt")

        if not param_path.exists():
            raise FileNotFoundError(
                f"Parameter file not found: {param_path}"
            )

        df = pd.read_csv(
            param_path,
            delimiter="\t",
            names=PARAMETER_COLUMNS,
            header=None,
        )
        # The parameter file stores tags like:
        #   _data_2018_05_19_00_02_05_channel_1.dat
        # Prepend 'admx' and strip .dat so tags are bare stems matching
        # on-disk filename stems exactly:
        #   admx_data_2018_05_19_00_02_05_channel_1
        raw_tags = "admx" + df["Filename_Tag"].astype(str)
        df["Filename_Tag"] = raw_tags.str.replace(r"\.dat$", "", regex=True)
        return df

    def is_scan_cut(
        self,
        nibble_name: str,
        timestamp: datetime,
        frequency_hz: Optional[float] = None,
    ) -> tuple[bool, str]:
        cut, reason = is_timestamp_cut(self.timestamp_cuts, timestamp)
        if cut:
            return cut, reason
        if frequency_hz is not None:
            cut, reason = is_frequency_cut(
                self.frequency_cuts, timestamp, frequency_hz
            )
        return cut, reason

    def output_path(self, nibble_name: str, label: str) -> Path:
        """Build an output path for the given nibble and label."""
        return get_output_path(self.nibble(nibble_name), label)

    def __repr__(self) -> str:
        return (
            f"RunConfig(yaml='{self.yaml_path.name}', "
            f"nibbles={self.nibble_names})"
        )