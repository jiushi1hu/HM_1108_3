"""Read/validate source data without changing hydrology or applying policies."""
import csv
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

IDS = ("WDD", "BHT", "XLD", "XJB", "THR")
HEAD_MODELS = ("fixed", "storage_sensitivity")
HEAD_SENSITIVITY_FILE = "reservoir_head_sensitivity.csv"
SERVICE_REQUIREMENTS_FILE = "reservoir_service_requirements.csv"
DAILY_LEVEL_LIMITS_FILE = "five_reservoir_happo_input_bounds_1965_2024.csv"
FLOW_COLUMNS = ("WDD_external_inflow_m3s",) + tuple(
    name + "_interval_inflow_m3s" for name in IDS[1:]
)
INFLOW_FORECAST_FILE = "five_reservoir_next7d_mean_inflow_forecast_1965_2024.csv"
FORECAST_COLUMNS = tuple(column.replace("_inflow_m3s", "_inflow_forecast_next7d_mean_m3s")
                         for column in FLOW_COLUMNS)
DEFAULT_EVALUATION_YEARS = (1998, 1999, 1970, 1997, 2006)
DEFAULT_EVALUATION_LABELS = ("extremely_wet", "wet", "normal", "dry", "extremely_dry")


def validate_service_scale(value, name):
    """Validate a positive sensitivity multiplier without changing source data."""
    message = name + ' must be finite and positive'
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        scale = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(scale) or scale <= 0.:
        raise ValueError(message)
    return scale


def resolve_year_split(experiment):
    """Partition all 60 source years into 55 training and five evaluation years.

    The training years are the complete complement of the evaluation years.
    Optional explicit training years must describe that same full complement;
    partial training subsets cannot silently change the production experiment.
    Evaluation order is retained for the representative five-year sequence.
    """
    available = set(range(1965, 2025))

    def years(key, default):
        values = experiment.get(key, default)
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(key + ' must be a nonempty list of integer years')
        if any(isinstance(y, (bool, np.bool_)) or not isinstance(y, (int, np.integer)) for y in values):
            raise ValueError(key + ' must contain integer years')
        values = tuple(int(y) for y in values)
        if len(set(values)) != len(values):
            raise ValueError(key + ' must contain unique years')
        if not set(values) <= available:
            raise ValueError(key + ' must lie within 1965--2024')
        return values

    evaluation = years('evaluation_years', DEFAULT_EVALUATION_YEARS)
    if len(evaluation) != 5:
        raise ValueError('evaluation_years must contain exactly five years')
    training = tuple(sorted(available - set(evaluation)))
    if experiment.get('training_years') is not None and set(years('training_years', ())) != set(training):
        raise ValueError('training_years must contain all 55 years outside evaluation_years')
    labels = experiment.get('evaluation_labels')
    if labels is None:
        labels = (DEFAULT_EVALUATION_LABELS if evaluation == DEFAULT_EVALUATION_YEARS else
                  tuple('evaluation_'+str(y) for y in evaluation))
    if (not isinstance(labels, (list, tuple)) or len(labels) != len(evaluation) or
            any(not isinstance(label, str) or not label.strip() for label in labels) or
            len(set(labels)) != len(labels)):
        raise ValueError('Require one unique nonempty evaluation label per evaluation year')
    return dict(training_years=training, evaluation_years=evaluation, evaluation_labels=tuple(labels))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty input: " + str(path))
    return rows


def load_inflow_forecast(path, dates):
    """Read the supplied forecast issued for each action date, in IDS order.

    Column names preserve the actual forcing definitions: WDD external inflow,
    followed by downstream incremental inflows. No date shift, rolling average
    of realized inflows, interpolation, or missing-value fallback is performed.
    The CSV must contain exactly one complete row for every forcing date.
    """
    expected = ('date',) + FORECAST_COLUMNS
    indices = {day: i for i, day in enumerate(dates)}
    values = np.empty((len(dates), len(IDS)), dtype=np.float64)
    seen = np.zeros(len(dates), dtype=bool)
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or set(headers) != set(expected):
            raise ValueError('Missing, duplicate or unknown inflow forecast columns')
        for row in reader:
            if None in row or any(row.get(key) is None or not row[key].strip() for key in expected):
                raise ValueError('Missing inflow forecast value at row '+str(reader.line_num))
            try:
                day = datetime.strptime(row['date'].strip(), '%Y-%m-%d').date()
                forecast = np.asarray([float(row[column]) for column in FORECAST_COLUMNS])
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError('Invalid inflow forecast at row '+str(reader.line_num)) from exc
            if day not in indices:
                raise ValueError('Inflow forecast date outside forcing calendar: '+str(day))
            i = indices[day]
            if seen[i]:
                raise ValueError('Duplicate inflow forecast date: '+str(day))
            if not np.isfinite(forecast).all() or (forecast < 0.).any():
                raise ValueError('Inflow forecasts must be finite and nonnegative')
            values[i] = forecast
            seen[i] = True
    if not seen.all():
        raise ValueError('Inflow forecasts must cover every forcing date')
    values.setflags(write=False)
    return values


class CascadeData:
    def __init__(self, root, config, experiment=None):
        self.root = Path(root).resolve()
        self.service_requirement_scales = {
            term: validate_service_scale(config.get(term + '_scale', 1.), term + '_scale')
            for term in ('guaranteed_output', 'ecological_flow')}
        self.head_model = config.get("head_model", "fixed")
        if self.head_model not in HEAD_MODELS:
            raise ValueError("Unknown head model: " + str(self.head_model))
        experiment = experiment or {}
        for name, values in resolve_year_split(experiment).items():
            setattr(self, name, values)
        if tuple(config["reservoir_ids"]) != IDS:
            raise ValueError("Reservoir order must be WDD/BHT/XLD/XJB/THR")
        rels = ["daily_incremental_inflows_1965_2024.csv", "initial_storage_1965_2024.csv",
                "reservoir_stage_storage.csv", "reservoir_parameters.csv"]
        self.hashes = {r: hashlib.sha256((self.root / r).read_bytes()).hexdigest() for r in rels}
        flows = read_csv(self.root / rels[0])
        self.dates = tuple(datetime.strptime(r["date"].strip(), "%Y-%m-%d").date() for r in flows)
        if any(b - a != timedelta(days=1) for a, b in zip(self.dates, self.dates[1:])):
            raise ValueError("Inflow dates must be unique, increasing and consecutive")
        self.external = np.array([[float(r[c]) for c in FLOW_COLUMNS] for r in flows], dtype=np.float64)
        if not np.isfinite(self.external).all() or (self.external < 0).any():
            raise ValueError("Inflow must be finite and nonnegative")
        expected = (datetime(1965, 1, 1).date(), datetime(2024, 12, 31).date())
        if (self.dates[0], self.dates[-1]) != expected or len(self.dates) != (expected[1] - expected[0]).days + 1:
            raise ValueError("Require complete daily inflows for 1965 through 2024")
        self.start_year, self.end_year = 1965, 2024
        self.years = np.array([d.year for d in self.dates])
        all_years = set(range(self.start_year, self.end_year + 1))
        self.train_mask = np.isin(self.years, self.training_years)
        train_flow = self.external[self.train_mask]
        self.flow_mean = train_flow.mean(axis=0)
        self.flow_std = np.maximum(train_flow.std(axis=0), 1.0)
        self.inflow_forecast = load_inflow_forecast(self.root / INFLOW_FORECAST_FILE, self.dates)
        self.hashes[INFLOW_FORECAST_FILE] = hashlib.sha256(
            (self.root / INFLOW_FORECAST_FILE).read_bytes()).hexdigest()
        initial = read_csv(self.root / rels[1])
        physics = read_csv(self.root / rels[3])
        if len(physics) != 5 or {r["reservoir_id"] for r in physics} != set(IDS):
            raise ValueError("Physics must contain exactly one row per reservoir")
        self.initial_storage_by_year = {}
        keys = set()
        for row in initial:
            date = datetime.strptime(row["date"], "%Y-%m-%d").date()
            key = (date.year, row["reservoir_id"])
            if date.month != 1 or date.day != 1 or key in keys or row["reservoir_id"] not in IDS:
                raise ValueError("Invalid or duplicate calendar-year-start storage record")
            keys.add(key)
            self.initial_storage_by_year.setdefault(date.year, {})[row["reservoir_id"]] = float(row["initial_storage_m3"])
        if set(self.initial_storage_by_year) != all_years or any(set(v) != set(IDS) for v in self.initial_storage_by_year.values()):
            raise ValueError("Require five initial storages for every year 1965--2024")
        self.initial_storage_by_year = {year: np.array([v[id] for id in IDS])
                                        for year, v in self.initial_storage_by_year.items()}
        records = {r["reservoir_id"]: r for r in physics}
        if [int(records[i]["order"]) for i in IDS] != [1, 2, 3, 4, 5]:
            raise ValueError("Invalid hydraulic order")
        self.power_cap = np.array([float(config.get("installed_capacity_overrides_mw", {}).get(
            i, records[i]["installed_capacity_mw"])) for i in IDS])
        self.turbine_cap = np.array([float(records[i]["max_turbine_flow_m3s"]) for i in IDS])
        self.release_cap = np.array([float(records[i]["max_total_release_m3s"]) for i in IDS])
        self.head = np.array([float(records[i]["typical_net_head_m"]) for i in IDS])
        self.anchors = {}
        anchor_rows = read_csv(self.root / rels[2])
        if {r["reservoir_id"] for r in anchor_rows} != set(IDS):
            raise ValueError("Unexpected anchor reservoir IDs")
        for i in IDS:
            pairs = sorted((float(r["level_m"]), float(r["storage_m3"]))
                           for r in anchor_rows if r["reservoir_id"] == i)
            points = np.asarray(pairs, dtype=np.float64)
            if len(points) < 2 or not np.isfinite(points).all() or (np.diff(points, axis=0) <= 0).any():
                raise ValueError("Non-monotone or invalid anchors: " + i)
            self.anchors[i] = points
        lower, upper = [], []
        for i in IDS:
            z = np.asarray(config["level_bounds_m"][i], dtype=float)
            points = self.anchors[i]
            if (z.shape != (2,) or not np.isfinite(z).all() or
                    z[0] >= z[1] or z[0] < points[0, 0] or z[1] > points[-1, 0]):
                raise ValueError("Operating bounds outside curve: " + i)
            v = np.interp(z, points[:, 0], points[:, 1])
            lower.append(v[0]); upper.append(v[1])
        self.vmin, self.vmax = np.array(lower), np.array(upper)
        for name in ["power_cap", "turbine_cap", "release_cap", "head"]:
            a = getattr(self, name)
            if not np.isfinite(a).all() or (a <= 0).any():
                raise ValueError("Invalid positive physics input: " + name)
        for year, storage in self.initial_storage_by_year.items():
            if not np.isfinite(storage).all() or ((storage < self.vmin) | (storage > self.vmax)).any():
                raise ValueError("Initial storage outside operating bounds in year " + str(year))
        self.head_sensitivity = None
        if self.head_model == "storage_sensitivity":
            self._load_head_sensitivity(config)
        self.service_requirements = None
        load_services = config.get("load_service_requirements", False)
        if not isinstance(load_services, bool):
            raise ValueError("load_service_requirements must be a boolean")
        if load_services:
            self.load_service_requirements()
        self.daily_level_limits = None
        use_daily_limits = config.get("use_daily_level_limits", False)
        if not isinstance(use_daily_limits, bool):
            raise ValueError("use_daily_level_limits must be a boolean")
        if use_daily_limits:
            self.load_daily_level_limits()

    def load_service_requirements(self):
        """Load constant service thresholds without applying penalties or bounds."""
        if self.service_requirements is None:
            requirements = ServiceRequirementsData(self)
            self.hashes[SERVICE_REQUIREMENTS_FILE] = requirements.sha256
            self.service_requirements = requirements
        return self.service_requirements

    def load_daily_level_limits(self):
        """Load the bounds on the state reached after each dated action.

        An action_date row already describes that action's next state. It is
        indexed without shifting the date and never changes global vmin/vmax.
        """
        if self.daily_level_limits is None:
            limits = DailyLevelLimitsData(self)
            self.hashes[DAILY_LEVEL_LIMITS_FILE] = limits.sha256
            self.daily_level_limits = limits
        return self.daily_level_limits

    def _load_head_sensitivity(self, config):
        """Validate the explicitly assumed scenario against its source inputs."""
        path = self.root / HEAD_SENSITIVITY_FILE
        if not path.is_file():
            raise ValueError("storage_sensitivity requires " + str(path))
        rows = read_csv(path)
        if len(rows) != 5 or {r.get("reservoir_id") for r in rows} != set(IDS):
            raise ValueError("Head sensitivity must contain exactly one row per reservoir")
        records = {r["reservoir_id"]: r for r in rows}
        if any(records[i].get("data_status") != "assumed_sensitivity_only" for i in IDS):
            raise ValueError("Head sensitivity must be labelled assumed_sensitivity_only")
        columns = ("operating_level_min_m", "operating_level_max_m",
                   "assumed_reference_forebay_level_m", "reference_net_head_m",
                   "assumed_effective_tailwater_plus_loss_m", "head_slope_m_per_m",
                   "scenario_net_head_min_m", "scenario_net_head_max_m", "efficiency")
        try:
            values = {key: np.array([float(records[i][key]) for i in IDS]) for key in columns}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Missing or nonnumeric head sensitivity field") from exc
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError("Head sensitivity fields must be finite")
        low, high = np.array([config["level_bounds_m"][i] for i in IDS], dtype=float).T
        reference_level = (low + high) / 2.
        expected = {
            "operating_level_min_m": low,
            "operating_level_max_m": high,
            "assumed_reference_forebay_level_m": reference_level,
            "reference_net_head_m": self.head,
            "assumed_effective_tailwater_plus_loss_m": reference_level - self.head,
            "scenario_net_head_min_m": self.head + low - reference_level,
            "scenario_net_head_max_m": self.head + high - reference_level,
        }
        for key, value in expected.items():
            if not np.allclose(values[key], value, rtol=0., atol=1e-6):
                raise ValueError("Head sensitivity inconsistent with source inputs: " + key)
        if not np.allclose(values["head_slope_m_per_m"], 1., rtol=0., atol=1e-12):
            raise ValueError("This head scenario requires a constant offset (head slope 1)")
        if ((values["scenario_net_head_min_m"] <= 0).any() or
                (values["scenario_net_head_max_m"] <= 0).any()):
            raise ValueError("Scenario net head must remain positive throughout operating bounds")
        eta = float(config["efficiency"])
        if not 0. < eta <= 1. or not np.allclose(values["efficiency"], eta, rtol=0., atol=1e-12):
            raise ValueError("Head sensitivity efficiency must match the physical configuration")
        self.head_sensitivity = values
        self.hashes[HEAD_SENSITIVITY_FILE] = hashlib.sha256(path.read_bytes()).hexdigest()

    def initial_storage(self, year):
        """Return a copy of the selected year's supplied initial storage."""
        if int(year) not in self.initial_storage_by_year:
            raise ValueError("Initial storage year outside the dataset")
        return self.initial_storage_by_year[int(year)].copy()

    def indices(self, start_year, end_year):
        if not self.start_year <= start_year <= end_year <= self.end_year:
            raise ValueError("Requested years lie outside the dataset")
        selected = np.flatnonzero((self.years >= start_year) & (self.years <= end_year))
        if not len(selected):
            raise ValueError("Empty year selection")
        return int(selected[0]), int(selected[-1]) + 1

    def levels(self, storage):
        result = []
        for i, value in zip(IDS, storage):
            p = self.anchors[i]
            if not p[0, 1] - 1.0 <= value <= p[-1, 1] + 1.0:
                raise ValueError("Storage outside interpolation domain")
            result.append(np.interp(value, p[:, 1], p[:, 0]))
        return np.asarray(result)


class ServiceRequirementsData:
    """Fixed daily service thresholds with source values and sensitivity scales.

    Zero explicitly disables one requirement. Loading never activates a reward
    or adds a release/power lower bound. Thresholds do not depend on year splits.
    Effective values are always formed once from the unchanged source CSV.
    """

    def __init__(self, data):
        path = data.root / SERVICE_REQUIREMENTS_FILE
        if not path.is_file():
            raise ValueError("Service requirements require " + str(path))
        required = ("reservoir_id", "paper_reservoir_id", "guaranteed_output_mw",
                    "ecological_flow_m3s")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)) or not set(required) <= set(headers):
                raise ValueError("Missing or duplicate service requirement columns")
            records = {}
            for row in reader:
                if None in row or any(row.get(k) is None or not row[k].strip() for k in required):
                    raise ValueError("Missing service requirement value at row " + str(reader.line_num))
                row = {k: row[k].strip() for k in required}
                name = row["reservoir_id"]
                if name not in IDS or name in records:
                    raise ValueError("Invalid or duplicate service requirement reservoir: " + name)
                paper_name = "TGR" if name == "THR" else name
                if row["paper_reservoir_id"] != paper_name:
                    raise ValueError("Inconsistent paper reservoir mapping: " + name)
                records[name] = row
        if set(records) != set(IDS):
            raise ValueError("Service requirements must contain exactly one row per reservoir")
        self.threshold_scales = {
            term: validate_service_scale(getattr(data, 'service_requirement_scales', {}).get(term, 1.),
                                         term + '_scale')
            for term in ('guaranteed_output', 'ecological_flow')}
        for term, field, cap in (("guaranteed_output", "guaranteed_output_mw", data.power_cap),
                                ("ecological_flow", "ecological_flow_m3s", data.release_cap)):
            try:
                base = np.array([float(records[name][field]) for name in IDS], dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Invalid service requirement: " + field) from exc
            if not np.isfinite(base).all() or (base < 0).any():
                raise ValueError("Service requirements must be finite and nonnegative: " + field)
            with np.errstate(over='ignore', under='ignore'):
                values = base * self.threshold_scales[term]
            if not np.isfinite(values).all() or ((base > 0.) & (values == 0.)).any():
                raise ValueError("Scaled service requirements must remain finite and preserve active requirements: " + field)
            if (values > cap).any():
                raise ValueError("Service requirement exceeds configured capacity: " + field)
            base.setflags(write=False)
            values.setflags(write=False)
            setattr(self, 'base_' + field, base)
            setattr(self, field, values)
        self.paper_reservoir_ids = tuple(records[name]["paper_reservoir_id"] for name in IDS)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def metadata(self):
        """Return JSON-compatible values and the input hash without exposing arrays."""
        return dict(file=SERVICE_REQUIREMENTS_FILE, sha256=self.sha256,
                    reservoir_ids=list(IDS), paper_reservoir_ids=list(self.paper_reservoir_ids),
                    guaranteed_output_mw=self.guaranteed_output_mw.tolist(),
                    ecological_flow_m3s=self.ecological_flow_m3s.tolist(),
                    base_guaranteed_output_mw=self.base_guaranteed_output_mw.tolist(),
                    base_ecological_flow_m3s=self.base_ecological_flow_m3s.tolist(),
                    threshold_scales=dict(self.threshold_scales),
                    time_basis="constant_daily_requirement", source_value_status="paper_reported",
                    value_status=("paper_reported" if all(scale == 1. for scale in self.threshold_scales.values())
                                  else "sensitivity_adjusted"))


class DailyLevelLimitsData:
    """Authoritative permitted next-state levels and storage by action date.

    Values are supplied envelopes, not water-level targets to be attained.
    The row for action_date t applies directly to the state after action t.
    Loading performs only data checks and never solves a future trajectory.
    """

    def __init__(self, data):
        path = data.root / DAILY_LEVEL_LIMITS_FILE
        if not path.is_file():
            raise ValueError("Daily level limits require " + str(path))
        fields = ("action_date", "reservoir_id", "next_state_min_level_m", "next_state_max_level_m",
                  "next_state_min_storage_m3", "next_state_max_storage_m3")
        shape = (len(data.dates), len(IDS))
        self.level_lower, self.level_upper, self.storage_lower, self.storage_upper = [
            np.empty(shape, dtype=np.float64) for _ in range(4)]
        arrays = (self.level_lower, self.level_upper, self.storage_lower, self.storage_upper)
        seen = np.zeros(shape, dtype=bool)
        day_index = {day.isoformat(): index for index, day in enumerate(data.dates)}
        reservoir_index = {name: index for index, name in enumerate(IDS)}
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)) or set(headers) != set(fields):
                raise ValueError("Missing, duplicate or unknown daily level-limit columns")
            for row in reader:
                if None in row or any(row.get(key) is None or not row[key].strip() for key in fields):
                    raise ValueError("Missing daily level-limit value at row " + str(reader.line_num))
                stamp, name = row["action_date"].strip(), row["reservoir_id"].strip()
                if stamp not in day_index or name not in reservoir_index:
                    raise ValueError("Unknown daily level-limit date/reservoir: " + stamp + " " + name)
                t, j = day_index[stamp], reservoir_index[name]
                if seen[t, j]:
                    raise ValueError("Duplicate daily level-limit date/reservoir: " + stamp + " " + name)
                try:
                    values = np.array([float(row[key]) for key in fields[2:]], dtype=np.float64)
                except (ValueError, TypeError, OverflowError) as exc:
                    raise ValueError("Invalid daily level-limit number at row " + str(reader.line_num)) from exc
                if not np.isfinite(values).all():
                    raise ValueError("Daily level limits must be finite")
                for array, value in zip(arrays, values):
                    array[t, j] = value
                seen[t, j] = True
        if not seen.all():
            raise ValueError("Daily level limits must cover every action date and all five reservoirs")
        if (self.level_lower > self.level_upper).any() or (self.storage_lower > self.storage_upper).any():
            raise ValueError("Daily lower limits cannot exceed upper limits")
        operating_lower, operating_upper = data.levels(data.vmin), data.levels(data.vmax)
        if ((self.level_lower < operating_lower-1e-6).any() or
                (self.level_upper > operating_upper+1e-6).any() or
                (self.storage_lower < data.vmin-1.).any() or (self.storage_upper > data.vmax+1.).any()):
            raise ValueError("Daily limits must lie within the configured operating range")
        for j, name in enumerate(IDS):
            anchors = data.anchors[name]
            for levels, storage in ((self.level_lower, self.storage_lower),
                                    (self.level_upper, self.storage_upper)):
                interpolated = np.interp(storage[:, j], anchors[:, 1], anchors[:, 0])
                # Source levels are rounded to six decimal places and storage
                # to whole m3; compare in metres to respect both precisions.
                if not np.allclose(levels[:, j], interpolated, rtol=0., atol=1e-6):
                    raise ValueError("Daily level/storage limits disagree with the stage-storage curve: " + name)
        for array in arrays:
            array.setflags(write=False)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def _check_range(self, start, stop):
        if (any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in (start, stop)) or
                not 0 <= start < stop <= self.storage_lower.shape[0]):
            raise ValueError("Invalid daily level-limit indices")

    def storage_bounds(self, start, stop):
        """Copies of after-action bounds for a half-open action-date range."""
        self._check_range(start, stop)
        return self.storage_lower[start:stop].copy(), self.storage_upper[start:stop].copy()

    def level_bounds(self, start, stop):
        """Copies of the supplied next-state water-level bounds in metres."""
        self._check_range(start, stop)
        return self.level_lower[start:stop].copy(), self.level_upper[start:stop].copy()
