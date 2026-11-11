"""Step 1: independent water accounting on 30 actual historical cascade days.

Run directly or with unittest discovery. No training, output files or RNG draws.
The environment's reported residual/system_residual are deliberately unused.
"""
import csv
from datetime import date, timedelta
from decimal import Decimal, localcontext
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade


RESERVOIRS = ("WDD", "BHT", "XLD", "XJB", "THR")
START = date(1965, 1, 1)
DAYS = 30
ACTION_BINS = 200
SECONDS = Decimal(86400)
EPSILON = float(np.finfo(np.float64).eps)
RELATIVE_TOLERANCE = 64.0 * EPSILON


def exact(value):
    """Retain every bit of an observed float64 for independent accounting."""
    return Decimal.from_float(float(value))


def historical_inflows():
    """Read the actual CSV independently of the environment's data loader."""
    columns = ("WDD_external_inflow_m3s", "BHT_interval_inflow_m3s",
               "XLD_interval_inflow_m3s", "XJB_interval_inflow_m3s",
               "THR_interval_inflow_m3s")
    wanted = {START + timedelta(days=day) for day in range(DAYS)}
    result = {}
    path = ROOT / "data/cascade_reservoir_new/daily_incremental_inflows_1965_2024.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            day = date.fromisoformat(row["date"])
            if day in wanted:
                if day in result:
                    raise AssertionError("Duplicate historical date: " + day.isoformat())
                result[day] = np.asarray([float(row[column]) for column in columns],
                                         dtype=np.float64)
    if set(result) != wanted:
        raise AssertionError("The selected window lacks 30 consecutive historical inflow rows")
    return result


def statistics():
    return dict(max_abs=0.0, max_relative=0.0, max_tolerance=0.0)


def accumulate(summary, residual, scale):
    absolute = float(abs(residual))
    relative = absolute / scale
    tolerance = RELATIVE_TOLERANCE * scale
    summary["max_abs"] = max(summary["max_abs"], absolute)
    summary["max_relative"] = max(summary["max_relative"], relative)
    summary["max_tolerance"] = max(summary["max_tolerance"], tolerance)
    return absolute <= tolerance


class CascadeMassBalanceTest(unittest.TestCase):
    def test_five_year_terminal_storage_without_annual_resets(self):
        from harl.envs.cascade_reservoir.episodes import episode_environment, training_windows
        physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        self.assertTrue(physical['terminal_storage_equal_initial'])
        data = CascadeData(ROOT/'data/cascade_reservoir_new', physical)
        windows = training_windows(data)
        rng = np.random.RandomState(723)
        for years in (windows[1965], windows[1971], list(data.evaluation_years)):
            env = episode_environment(data, physical, years)
            initial = env.storage.copy()
            intermediate_year_end_errors = []
            for step in range(env.episode_days):
                command = rng.randint(env.action_bins, size=5)
                if step % 31 == 0:
                    command.fill(0 if step % 62 == 0 else env.action_bins-1)
                env.step(command)
                row = env.last_transition
                np.testing.assert_allclose(row['storage_end']-row['storage_start'],
                    (row['inflow']-row['release'])*86400., rtol=0., atol=1e-4)
                np.testing.assert_allclose(row['inflow'][1:],
                    row['external'][1:]+row['release'][:-1], rtol=0., atol=1e-10)
                if step < env.episode_days-1:
                    self.assertTrue(np.any(row['storage_upper_m3'] > row['storage_lower_m3']))
                    if env.data.dates[step].year != env.data.dates[step+1].year:
                        intermediate_year_end_errors.append(float(np.abs(env.storage-initial).max()))
            np.testing.assert_allclose(env.storage, initial, rtol=0., atol=1e-3)
            np.testing.assert_array_equal(env.last_transition['controls'], np.zeros(5))
            self.assertEqual(len(intermediate_year_end_errors), 4)
            self.assertTrue(all(error > 1. for error in intermediate_year_end_errors))
            env.reset()
            np.testing.assert_array_equal(env.storage, initial)

    def test_terminal_requirement_also_applies_without_daily_limits(self):
        from harl.envs.cascade_reservoir.episodes import episode_environment
        physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        physical['use_daily_level_limits'] = False
        data = CascadeData(ROOT/'data/cascade_reservoir_new', physical)
        env = episode_environment(data, physical, [1965], days=5)
        for _ in range(5):
            env.step(np.full(5, 99))
        np.testing.assert_allclose(env.storage, env.initial_storage, rtol=0., atol=1e-3)
        np.testing.assert_array_equal(env.last_transition['storage_lower_m3'], env.initial_storage)
        np.testing.assert_array_equal(env.last_transition['storage_upper_m3'], env.initial_storage)

    def check_day(self, condition, reason, context):
        if not condition:
            self.fail(json.dumps(dict(context, status="FAIL", cause=reason), sort_keys=True))

    def test_30_day_historical_mass_balance_and_propagation(self):
        with (ROOT / "harl/configs/envs_cfgs/cascade_reservoir.yaml").open() as stream:
            physical = yaml.safe_load(stream)
        with (ROOT / "harl/configs/algos_cfgs/happo_cascade.yaml").open() as stream:
            experiment = yaml.safe_load(stream)["experiment"]
        data = CascadeData(ROOT / "data/cascade_reservoir_new", physical, experiment)
        forcing = historical_inflows()
        self.assertEqual(tuple(data.dates[:DAYS]),
                         tuple(START + timedelta(days=i) for i in range(DAYS)))
        self.assertEqual(tuple(physical["reservoir_ids"]), RESERVOIRS)
        self.assertEqual(physical["timestep_seconds"], 86400.0)
        context = dict(date=START.isoformat(), validated_days=0, completed_days=0,
                       storage_start_m3=data.initial_storage(START.year).tolist())
        try:
            env = CascadeReservoirEnv(data, physical, start_year=START.year,
                                      action_bins=ACTION_BINS, episode_days=DAYS)
        except InfeasibleCascade as exc:
            self.fail(json.dumps(dict(context, status="FAIL", physical_failures=1,
                cause=str(exc), error_type=type(exc).__name__), sort_keys=True))

        mass = [statistics() for _ in RESERVOIRS]
        propagation = [statistics() for _ in RESERVOIRS]
        partition = [statistics() for _ in RESERVOIRS]
        system = statistics()
        initial = env.storage.copy()
        total_external = Decimal(0)
        total_outlet = Decimal(0)
        validated = 0
        spill_days = 0

        # 80-digit arithmetic keeps rounding in the independent checker far
        # below float64 rounding in the actual observed states and flows.
        with localcontext() as decimal_context:
            decimal_context.prec = 80
            for day in range(DAYS):
                current_date = START + timedelta(days=day)
                start = env.storage.copy()
                external = forcing[current_date]
                # A fixed, heterogeneous sequence spans the existing action
                # grid without training, random sampling or feasibility retries.
                actions = (17 + 37 * np.arange(5) + 13 * day) % ACTION_BINS
                context = dict(date=current_date.isoformat(), validated_days=validated,
                    completed_days=env.index-env.start, storage_start_m3=start.tolist(),
                    historical_external_inflow_m3s=external.tolist(),
                    discrete_actions=actions.tolist(), action_bins=ACTION_BINS,
                    release_capacity_m3s=data.release_cap.tolist(),
                    global_storage_lower_m3=data.vmin.tolist(),
                    global_storage_upper_m3=data.vmax.tolist())
                if env.level_limits is not None:
                    context.update(
                        next_storage_lower_m3=env.level_limits.storage_lower[env.index].tolist(),
                        next_storage_upper_m3=env.level_limits.storage_upper[env.index].tolist())
                self.check_day(data.dates[env.index] == current_date,
                               "Environment date is not the next consecutive historical day", context)
                self.check_day(np.array_equal(data.external[env.index], external),
                               "Loaded inflows disagree with independently read historical CSV", context)
                try:
                    _, _, _, dones, _, _ = env.step(actions)
                except InfeasibleCascade as exc:
                    self.fail(json.dumps(dict(context, status="FAIL", physical_failures=1,
                        cause=str(exc), error_type=type(exc).__name__), sort_keys=True))
                end = env.storage.copy()
                row = env.last_transition
                context.update(completed_days=env.index-env.start,
                    storage_end_m3=end.tolist(), inflow_m3s=row["inflow"].tolist(),
                    release_m3s=row["release"].tolist())
                for key in ("inflow", "release", "turbine", "spill"):
                    self.check_day(row[key].shape == (5,) and np.isfinite(row[key]).all(),
                                   "Invalid observed " + key + " vector", context)
                self.check_day(np.isfinite(start).all() and np.isfinite(end).all(),
                               "Non-finite observed reservoir storage", context)
                self.check_day(row["date"] == current_date.isoformat(),
                               "Transition date mismatch", context)
                self.check_day(np.array_equal(row["storage_start"], start) and
                               np.array_equal(row["storage_end"], end),
                               "Reported storage differs from the actual environment state", context)
                self.check_day(np.array_equal(row["external"], external),
                               "Transition external inflows differ from historical data", context)
                self.check_day(bool(np.asarray(dones).all()) == (day == DAYS-1),
                               "Episode completion does not match the requested 30 days", context)

                inflow, release = row["inflow"], row["release"]
                for i, reservoir in enumerate(RESERVOIRS):
                    residual = exact(end[i])-exact(start[i])-SECONDS*(exact(inflow[i])-exact(release[i]))
                    scale = max(abs(start[i]), abs(end[i]),
                                86400.0*abs(inflow[i]), 86400.0*abs(release[i]), 1.0)
                    self.check_day(accumulate(mass[i], residual, scale),
                        reservoir + " independent daily mass balance exceeded tolerance: " + str(residual), context)

                    upstream = 0.0 if i == 0 else release[i-1]
                    flow_error = exact(inflow[i])-exact(external[i])-exact(upstream)
                    flow_scale = max(abs(inflow[i]), abs(external[i]), abs(upstream), 1.0)
                    self.check_day(accumulate(propagation[i], flow_error, flow_scale),
                        reservoir + " same-day upstream-release propagation failed: " + str(flow_error), context)

                    split_error = exact(release[i])-exact(row["turbine"][i])-exact(row["spill"][i])
                    split_scale = max(abs(release[i]), abs(row["turbine"][i]),
                                      abs(row["spill"][i]), 1.0)
                    self.check_day(accumulate(partition[i], split_error, split_scale),
                        reservoir + " total release does not equal turbine flow plus spill: " + str(split_error), context)

                external_sum = sum((exact(q) for q in external), Decimal(0))
                start_sum = sum((exact(v) for v in start), Decimal(0))
                end_sum = sum((exact(v) for v in end), Decimal(0))
                # Internal transfers cancel. THR total release is the only
                # modeled outlet; interval inflows enter the system once each.
                system_residual = end_sum-start_sum-SECONDS*(external_sum-exact(release[-1]))
                system_scale = max(float(abs(start_sum)), float(abs(end_sum)),
                    float(SECONDS*external_sum), 86400.0*abs(release[-1]), 1.0)
                self.check_day(accumulate(system, system_residual, system_scale),
                    "Independent system-wide daily water balance failed: " + str(system_residual), context)
                total_external += SECONDS*external_sum
                total_outlet += SECONDS*exact(release[-1])
                spill_days += int(np.any(row["spill"] > 0.0))
                validated += 1

            initial_sum = sum((exact(v) for v in initial), Decimal(0))
            final_sum = sum((exact(v) for v in env.storage), Decimal(0))
            cumulative = final_sum-initial_sum-total_external+total_outlet
            cumulative_scale = max(float(abs(initial_sum)), float(abs(final_sum)),
                                   float(total_external), float(total_outlet), 1.0)
            cumulative_stats = statistics()
            context.update(validated_days=validated)
            self.check_day(accumulate(cumulative_stats, cumulative, cumulative_scale),
                "Independent full-window system water balance failed: " + str(cumulative), context)

        self.assertEqual(validated, DAYS)
        print("WINDOW {} through {}; validated_days={}; reservoir_days={}; physical_failures=0".format(
            START.isoformat(), (START+timedelta(days=DAYS-1)).isoformat(), validated, 5*validated))
        print("ACTIONS bins=200; a[i,day]=(17+37*i+13*day)%200; zero-based i/day; spill_days={}".format(spill_days))
        print("TOLERANCE float64_epsilon={:.17e}; relative_limit=64*epsilon={:.17e}".format(
            EPSILON, RELATIVE_TOLERANCE))
        print("VOLUME_SCALE=max(|V_current|,|V_next|,86400*|Inflow|,86400*|Release|,1 m3)")
        print("FLOW_SCALE=max(|Inflow|,|external|,|upstream Release|,1 m3/s)")
        print("SYSTEM_SCALE=max(total start storage,total end storage,external volume,outlet volume,1 m3)")
        print("All checks use |residual|<=64*epsilon*scale; relative residual=|residual|/scale.")
        for reservoir, summary in zip(RESERVOIRS, mass):
            print("MASS {} max_abs_m3={:.17e} max_relative={:.17e} max_abs_tolerance_m3={:.17e} PASS".format(
                reservoir, summary["max_abs"], summary["max_relative"], summary["max_tolerance"]))
        for i, (reservoir, summary) in enumerate(zip(RESERVOIRS, propagation)):
            label = "external->WDD" if i == 0 else RESERVOIRS[i-1]+"->"+reservoir
            print("FLOW {} max_abs_m3s={:.17e} max_relative={:.17e} PASS".format(
                label, summary["max_abs"], summary["max_relative"]))
        print("RELEASE_PARTITION max_abs_m3s={:.17e} max_relative={:.17e} PASS".format(
            max(item["max_abs"] for item in partition), max(item["max_relative"] for item in partition)))
        for label, summary in (("SYSTEM_DAILY", system), ("SYSTEM_30_DAY", cumulative_stats)):
            print("{} max_abs_m3={:.17e} max_relative={:.17e} max_abs_tolerance_m3={:.17e} PASS".format(
                label, summary["max_abs"], summary["max_relative"], summary["max_tolerance"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
