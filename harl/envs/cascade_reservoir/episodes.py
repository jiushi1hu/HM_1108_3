"""Fixed five-year episodes with explicit calendar views and split isolation."""
import copy
import numpy as np

DEFAULT_EPISODE_DAYS = 1825
DEFAULT_ACTION_BINS = 100


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name+' must be a positive integer')
    return int(value)


def training_windows(data, days=DEFAULT_EPISODE_DAYS):
    """Jan-1 starts whose entire consecutive daily window belongs to training."""
    days = positive_integer(days, 'episode_days')
    allowed = set(data.training_years)
    windows = {}
    for year in sorted(allowed):
        start, _ = data.indices(year, year)
        stop = start+days
        if stop > len(data.dates):
            continue
        years = list(dict.fromkeys(d.year for d in data.dates[start:stop]))
        if set(years).issubset(allowed):
            windows[int(year)] = years
    if not windows:
        raise ValueError('No complete training-only windows for episode_days={}'.format(days))
    return windows


class EpisodeData:
    """Read-only daily view; static physics and initial-storage inputs are shared.

    Ordered years can be nonconsecutive for the representative evaluation
    sequence. Their source dates and operating rules are retained explicitly.
    No hydrology is interpolated and no leap days are deleted.
    """

    def __init__(self, source, years, days=DEFAULT_EPISODE_DAYS):
        days = positive_integer(days, 'episode_days')
        if not years or len(set(years)) != len(years):
            raise ValueError('Episode source years must be nonempty and unique')
        indices = np.concatenate([np.arange(*source.indices(year, year)) for year in years])
        if len(indices) < days:
            raise ValueError('Episode source years do not contain the requested daily horizon')
        self.source = source
        # A row contains this action's next-state limits, so the daily input
        # view needs exactly the requested transitions and no future rows.
        self.source_indices = indices[:days]
        self.dates = tuple(source.dates[i] for i in self.source_indices)
        self.years = np.array([d.year for d in self.dates])
        self.source_years = list(dict.fromkeys(self.years.tolist()))
        self.start_year, self.end_year = min(self.source_years), max(self.source_years)
        self.external = source.external[self.source_indices].copy()
        self.external.setflags(write=False)
        # Use the forecast issued on each original source date, including
        # nonconsecutive held-out year joins; never average joined forcing rows.
        self.inflow_forecast = source.inflow_forecast[self.source_indices].copy()
        self.inflow_forecast.setflags(write=False)
        self.daily_level_limits = None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'source'), name)

    def indices(self, start_year, end_year):
        selected = np.flatnonzero((self.years >= start_year) & (self.years <= end_year))
        if not len(selected):
            raise ValueError('Year selection is outside this episode')
        if not np.array_equal(selected, np.arange(selected[0], selected[-1]+1)):
            raise ValueError('Year selection is not contiguous in this episode sequence')
        return int(selected[0]), int(selected[-1])+1

    def load_daily_level_limits(self):
        """Slice action-date rows without shifting their next-state limits."""
        if self.daily_level_limits is None:
            original = self.source.load_daily_level_limits()
            view = copy.copy(original)
            view.data = self
            for name in ('storage_lower', 'storage_upper', 'level_lower', 'level_upper'):
                value = getattr(original, name)[self.source_indices].copy()
                value.setflags(write=False)
                setattr(view, name, value)
            self.daily_level_limits = view
        return self.daily_level_limits


def episode_environment(data, physical, years, days=DEFAULT_EPISODE_DAYS,
                        action_bins=DEFAULT_ACTION_BINS):
    from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
    view = EpisodeData(data, years, days)
    return CascadeReservoirEnv(view, physical, start_year=years[0],
                               action_bins=action_bins, episode_days=days)
