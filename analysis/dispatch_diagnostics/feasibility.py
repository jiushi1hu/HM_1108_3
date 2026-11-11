"""Observational, same-state dispatch attainability for every reservoir.

Only one reservoir's command varies in each counterfactual. Starting storage,
current external inflows, action-date bounds, active terminal reachability
constraints and the other actors' originally sampled commands stay fixed. A
same-day witness does not establish a better
multi-day policy; an unattainable requirement in the saved state does not prove
that the state, or the observed long-run violation rate, was unavoidable.

This module deliberately lives outside ``harl``/``examples`` so investigation
does not change the implementation fingerprint of saved policies.
"""
import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS


CLASSIFICATION_DEFINITIONS = {
    'requirement_inactive': 'The effective service requirement is zero.',
    'no_violation': 'The executed service meets its requirement within reward tolerance.',
    'same_day_policy_alternative': 'Another available discrete command meets this service from the same saved state; future water value is unassessed.',
    'discrete_grid_gap': 'A continuous command meets the service but no configured discrete command does.',
    'same_state_capacity_limit': 'The service exceeds its optimistic outlet or turbine/installed capacity bound.',
    'same_state_water_limit': 'Releasable water above the current next-state storage floor cannot meet the service even at optimistic maximum head for power.',
    'same_state_dispatch_limit': 'The current feasible release ceiling cannot meet the service even at optimistic maximum head for power.',
    'same_state_head_limit': 'All continuous current-state power choices fail despite optimistic water, dispatch and hardware bounds.',
}
SCOPE = ('one command changes; same starting storage, current forcing, next-state bounds, '
         'terminal reachability constraints and other originally sampled commands; '
         'no additional forcing or trajectory changes')
LIMITATION = ('Same-state restrictions do not prove exogenous or globally unavoidable failures. '
              'A meeting alternative may change other services and future storage; no long-term '
              'optimality or improvement claim is made.')


class DispatchDiagnosticError(RuntimeError):
    """Counterfactual verification failed; the actual trajectory did not fail."""


def _vector(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (5,) or not np.isfinite(value).all():
        raise ValueError(name+' must be a finite five-vector')
    return value


class _TransitionAnalysis:
    def __init__(self, env, row):
        self.env, self.row, self.data, self.physics = env, row, env.data, env.physics
        bins = env.action_bins
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 2:
            raise ValueError('action_bins must be an integer >= 2')
        self.bins = int(bins)
        for name in ('storage_start', 'storage_end', 'external', 'inflow', 'release',
                     'power', 'net_head_m', 'turbine', 'low', 'high', 'controls'):
            _vector(row[name], name)
        self.lower = _vector(row.get('storage_lower_m3', self.data.vmin), 'storage lower')
        self.upper = _vector(row.get('storage_upper_m3', self.data.vmax), 'storage upper')
        requirements = getattr(getattr(env, 'operation_reward', None), 'requirements', None)
        if requirements is None:
            requirements = getattr(self.data, 'service_requirements', None)
        if requirements is None:
            raise ValueError('Dispatch diagnostics require loaded service requirements')
        self.requirements = {
            'guaranteed_output': _vector(requirements.guaranteed_output_mw, 'guaranteed output'),
            'ecological_flow': _vector(requirements.ecological_flow_m3s, 'ecological flow'),
        }
        if any((v < 0).any() for v in self.requirements.values()):
            raise ValueError('Requirements must be nonnegative')
        sampled = row.get('sampled_actions')
        if sampled is None:
            if not np.all(np.asarray(row['controls']) == 1.):
                raise ValueError('Forced counterfactuals require original sampled_actions')
            self.actions = _vector(row['normalized_actions'], 'normalized_actions').copy()
            indices = self.actions*(self.bins-1)
            if not np.allclose(indices, np.round(indices), rtol=0., atol=1e-8):
                raise ValueError('Counterfactual commands must identify the discrete grid')
            sampled = np.round(indices)
        else:
            sampled = _vector(sampled, 'sampled_actions')
            self.actions = sampled/(self.bins-1)
        if ((sampled != np.floor(sampled)) | (sampled < 0) | (sampled >= self.bins)).any():
            raise ValueError('Invalid sampled discrete command')
        self.sampled = np.asarray(sampled, dtype=int)
        if ((self.actions < 0) | (self.actions > 1)).any():
            raise ValueError('Normalized commands must lie within [0,1]')
        self.grid = np.arange(self.bins, dtype=np.float64)/(self.bins-1)
        # Shared once across all five analyses; local grid/head arrays and full
        # cascade witnesses are also cached once per reservoir/command.
        _, self.downstream_upper = self.physics.downstream_intervals(
            row['storage_start'], row['external'], self.lower, self.upper)
        self.start_levels = self.physics._forebay_levels(row['storage_start'])
        self.end_levels = self.physics._forebay_levels(row['storage_end'])
        self.witnesses, self.local = {}, {}
        if self.physics.head_model == 'fixed':
            self.reference_heads = np.asarray(self.data.head)
            self.maximum_heads = self.reference_heads
        elif self.physics.head_model == 'storage_sensitivity':
            scenario = self.data.head_sensitivity
            self.reference_heads = np.asarray(scenario['reference_net_head_m'])
            low, high = self.data.levels(self.data.vmin), self.data.levels(self.data.vmax)
            self.maximum_heads = np.maximum(*[
                self.reference_heads+scenario['head_slope_m_per_m']*(level-scenario['assumed_reference_forebay_level_m'])
                for level in (low, high)])
        else:
            raise ValueError('Unsupported head model')

    def generation(self, i, releases):
        """Exact local equivalent of shared daily mean-head generation."""
        releases = np.asarray(releases, dtype=np.float64)
        if self.physics.head_model == 'fixed':
            heads = np.full(releases.shape, self.reference_heads[i])
        else:
            points = self.data.anchors[IDS[i]]
            end = self.row['storage_start'][i]+self.physics.dt*(self.row['inflow'][i]-releases)
            z = np.interp(np.clip(end, self.data.vmin[i], self.data.vmax[i]), points[:, 1], points[:, 0])
            scenario = self.data.head_sensitivity
            heads = self.reference_heads[i]+scenario['head_slope_m_per_m'][i]*(
                (self.start_levels[i]+z)/2.-scenario['assumed_reference_forebay_level_m'][i])
        if not np.isfinite(heads).all() or (heads <= 0).any():
            raise DispatchDiagnosticError('Invalid counterfactual head')
        coefficient = self.physics.rho_g_eta*heads/1e6
        cap = np.minimum(self.data.turbine_cap[i], self.data.power_cap[i]/coefficient)
        return np.minimum(releases, cap)*coefficient, heads

    def continuous_candidates(self, i, low, high, forced):
        if forced:
            return np.array([float(self.row['release'][i])])
        points = [low, high]
        turbine = float(self.data.turbine_cap[i])
        if low < turbine < high:
            points.append(turbine)
        if self.physics.head_model == 'storage_sensitivity':
            storage = np.r_[self.data.anchors[IDS[i]][:, 1], self.data.vmin[i], self.data.vmax[i]]
            breaks = self.row['inflow'][i]+(self.row['storage_start'][i]-storage)/self.physics.dt
            points.extend(float(q) for q in breaks if low < q < high)
        points = np.unique(points)
        _, heads = self.generation(i, points)
        coefficients = self.physics.rho_g_eta*heads/1e6
        stationary = []
        for left, right, first, last in zip(points[:-1], points[1:], coefficients[:-1], coefficients[1:]):
            if right <= left or left >= turbine:
                continue
            slope = (last-first)/(right-left)
            intercept = first-slope*left
            if slope < 0:
                vertex = -intercept/(2.*slope)
                if left < vertex < right:
                    stationary.append(vertex)
        return np.unique(np.r_[points, stationary])

    def candidates(self, i):
        if i not in self.local:
            low, high = float(self.row['low'][i]), float(self.row['high'][i])
            if low > high+1e-8 or low < -1e-8:
                raise ValueError('Invalid feasible release interval')
            forced = high-low <= 1e-8
            releases = (np.full(self.bins, float(self.row['release'][i])) if forced else
                        low+self.grid**2*(high-low))
            powers, heads = self.generation(i, releases)
            continuous = self.continuous_candidates(i, low, high, forced)
            continuous_power, continuous_head = self.generation(i, continuous)
            self.local[i] = dict(low=low, high=high, forced=forced, releases=releases,
                powers=powers, heads=heads, continuous=continuous,
                continuous_power=continuous_power, continuous_head=continuous_head)
        return self.local[i]

    def replay(self, i, normalized, release, power):
        key = (i, float(normalized))
        if key not in self.witnesses:
            actions = self.actions.copy()
            actions[i] = normalized
            constraints = self.row.get('storage_constraints')
            options = {} if constraints is None else dict(storage_constraints=constraints)
            replay = self.physics.transition(self.row['storage_start'], self.row['external'],
                                              actions, self.lower, self.upper, **options)
            self.witnesses[key] = replay
        replay = self.witnesses[key]
        if (not np.isclose(replay['release'][i], release, rtol=1e-10, atol=1e-7) or
                not np.isclose(replay['power'][i], power, rtol=1e-10, atol=1e-7)):
            raise DispatchDiagnosticError('Local candidate disagrees with full cascade physics')
        return replay

    def at_release(self, i, release):
        local = self.candidates(i)
        normalized = (.5 if local['forced'] else
                      float(np.sqrt(np.clip((release-local['low'])/(local['high']-local['low']), 0., 1.))))
        power, _ = self.generation(i, np.array([release]))
        return self.replay(i, normalized, release, float(power[0]))

    def other_counts(self, i, row):
        result = {}
        for term, actual in (('guaranteed', row['power']), ('ecological', row['release'])):
            requirement = self.requirements['guaranteed_output' if term == 'guaranteed' else 'ecological_flow']
            active = requirement > 0
            active = active.copy()
            active[i] = False
            result[term] = int(((requirement-np.asarray(actual) > 1e-10*np.maximum(requirement, 1.)) & active).sum())
        return result

    def witness_fields(self, i, prefix, witness):
        counts = self.other_counts(i, witness)
        return {
            prefix+'_other_guaranteed_violations': counts['guaranteed'],
            prefix+'_other_ecological_violations': counts['ecological'],
            prefix+'_end_storage_delta_m3': float(witness['storage_end'][i]-self.row['storage_end'][i]),
            prefix+'_other_end_storage_delta_m3': float(np.sum(witness['storage_end']-self.row['storage_end'])-
                                                        witness['storage_end'][i]+self.row['storage_end'][i]),
        }

    def service(self, i, term, water_upper):
        is_power = term == 'guaranteed_output'
        threshold = float(self.requirements[term][i])
        tolerance = 1e-10*max(threshold, 1.)
        actual = float(self.row['power' if is_power else 'release'][i])
        raw = max(threshold-actual, 0.)
        shortfall = raw if raw > tolerance else 0.
        record = dict(active=bool(threshold > 0), violating=bool(shortfall), threshold=threshold,
            actual=actual, shortfall=shortfall, relative_shortfall=shortfall/threshold if threshold else 0.,
            shortfall_integral=shortfall*(self.physics.dt/3600. if is_power else self.physics.dt),
            classification='requirement_inactive' if threshold == 0 else 'no_violation',
            water_limit_proven=False, dispatch_limit_proven=False, capacity_limit_proven=False)
        optional = ('best_grid_action_index', 'best_grid_actual', 'best_grid_release_m3s', 'best_grid_head_m',
            'meeting_grid_action_count', 'minimum_meeting_action_index', 'minimum_meeting_release_m3s',
            'meeting_release_delta_m3s', 'meeting_end_storage_delta_m3', 'continuous_max_actual',
            'continuous_best_release_m3s', 'continuous_best_head_m', 'continuous_minimum_meeting_release_m3s',
            'continuous_minimum_meeting_normalized_action', 'continuous_meeting_release_delta_m3s',
            'minimum_achievable_shortfall', 'minimum_grid_shortfall')
        for name in optional:
            record[name] = None
        for prefix in ('best_grid', 'minimum_meeting', 'continuous_best', 'continuous_minimum_meeting'):
            for suffix in ('other_guaranteed_violations', 'other_ecological_violations',
                           'end_storage_delta_m3', 'other_end_storage_delta_m3'):
                record[prefix+'_'+suffix] = None
        if not shortfall:
            return record
        local = self.candidates(i)
        values = local['powers'] if is_power else local['releases']
        best = int(np.argmax(values))
        witness = self.replay(i, float(self.grid[best]), float(local['releases'][best]), float(local['powers'][best]))
        record.update(best_grid_action_index=best, best_grid_actual=float(witness['power' if is_power else 'release'][i]),
            best_grid_release_m3s=float(witness['release'][i]), best_grid_head_m=float(witness['net_head_m'][i]),
            minimum_grid_shortfall=max(0., threshold-float(values[best])))
        record.update(self.witness_fields(i, 'best_grid', witness))
        meeting = np.flatnonzero(threshold-values <= tolerance)
        record['meeting_grid_action_count'] = int(len(meeting))
        if len(meeting):
            first = int(meeting[0])
            witness = self.replay(i, float(self.grid[first]), float(local['releases'][first]), float(local['powers'][first]))
            if threshold-float(witness['power' if is_power else 'release'][i]) > tolerance:
                raise DispatchDiagnosticError('A minimum-meeting grid witness fails its requirement')
            record.update(minimum_meeting_action_index=first,
                minimum_meeting_release_m3s=float(witness['release'][i]),
                meeting_release_delta_m3s=float(witness['release'][i]-self.row['release'][i]),
                meeting_end_storage_delta_m3=float(witness['storage_end'][i]-self.row['storage_end'][i]))
            record.update(self.witness_fields(i, 'minimum_meeting', witness))
        continuous_values = local['continuous_power'] if is_power else local['continuous']
        best_continuous = int(np.argmax(continuous_values))
        continuous_release = float(local['continuous'][best_continuous])
        witness = self.at_release(i, continuous_release)
        continuous_actual = float(witness['power' if is_power else 'release'][i])
        record.update(continuous_max_actual=continuous_actual,
            continuous_best_release_m3s=float(witness['release'][i]),
            continuous_best_head_m=float(witness['net_head_m'][i]),
            minimum_achievable_shortfall=max(0., threshold-continuous_actual))
        record.update(self.witness_fields(i, 'continuous_best', witness))
        # The candidate list includes every interpolation kink and stationary
        # point. Between consecutive points power is monotone (possibly flat
        # at installed capacity), so the first threshold crossing is bracketed.
        continuous_meeting = np.flatnonzero(threshold-continuous_values <= tolerance)
        if len(continuous_meeting):
            first = int(continuous_meeting[0])
            release = float(local['continuous'][first])
            if first and not local['forced']:
                left, right = float(local['continuous'][first-1]), release
                # Bracket against the reward's same tolerance. Bisection retains
                # the meeting side, avoiding a false infeasibility at a root.
                for _ in range(45):
                    midpoint = .5*(left+right)
                    value = float(self.generation(i, np.array([midpoint]))[0][0]) if is_power else midpoint
                    if threshold-value <= tolerance:
                        right = midpoint
                    else:
                        left = midpoint
                release = right
            witness = self.at_release(i, release)
            # An extra ulp on the meeting side absorbs the inverse quadratic
            # action mapping's roundoff at an exact threshold boundary.
            if threshold-float(witness['power' if is_power else 'release'][i]) > tolerance+1e-7:
                raise DispatchDiagnosticError('A continuous meeting witness fails its requirement')
            normalized = (.5 if local['forced'] else float(np.sqrt(np.clip(
                (release-local['low'])/(local['high']-local['low']), 0., 1.))))
            record.update(continuous_minimum_meeting_release_m3s=float(witness['release'][i]),
                continuous_minimum_meeting_normalized_action=normalized,
                continuous_meeting_release_delta_m3s=float(witness['release'][i]-self.row['release'][i]))
            record.update(self.witness_fields(i, 'continuous_minimum_meeting', witness))
        if is_power:
            maximum_coefficient = self.physics.rho_g_eta*self.maximum_heads[i]/1e6
            hardware = min(float(self.data.power_cap[i]), maximum_coefficient*float(self.data.turbine_cap[i]))
            water = max(water_upper, 0.)*maximum_coefficient
            dispatch = min(float(self.data.power_cap[i]), maximum_coefficient*min(max(local['high'], 0.), float(self.data.turbine_cap[i])))
        else:
            hardware, water, dispatch = float(self.data.release_cap[i]), max(water_upper, 0.), max(local['high'], 0.)
        record.update(capacity_limit_proven=bool(threshold-hardware > tolerance),
            water_limit_proven=bool(threshold-water > tolerance),
            dispatch_limit_proven=bool(threshold-dispatch > tolerance))
        if len(meeting):
            classification = 'same_day_policy_alternative'
        elif threshold-continuous_actual <= tolerance:
            classification = 'discrete_grid_gap'
        elif record['capacity_limit_proven']:
            classification = 'same_state_capacity_limit'
        elif record['water_limit_proven']:
            classification = 'same_state_water_limit'
        elif record['dispatch_limit_proven']:
            classification = 'same_state_dispatch_limit'
        elif is_power:
            classification = 'same_state_head_limit'
        else:
            raise DispatchDiagnosticError('Ecological maximum must be the feasible release ceiling')
        record['classification'] = classification
        return record

    def records(self):
        records = []
        for i, name in enumerate(IDS):
            row = self.row
            water = float(row['inflow'][i]+(row['storage_start'][i]-self.lower[i])/self.physics.dt)
            others = self.other_counts(i, row)
            reference_power = min(float(self.data.power_cap[i]), self.physics.rho_g_eta*self.reference_heads[i]*row['turbine'][i]/1e6)
            threshold = float(self.requirements['guaranteed_output'][i])
            tolerance = 1e-10*max(threshold, 1.)
            record = dict(date=str(row.get('date', 'unknown')), reservoir_id=name, reservoir_index=i,
                action_bins=self.bins, sampled_action_index=int(self.sampled[i]),
                forced_action=bool(row['controls'][i] <= 0),
                storage_start_m3=float(row['storage_start'][i]), storage_end_m3=float(row['storage_end'][i]),
                level_start_m=float(self.start_levels[i]), level_end_m=float(self.end_levels[i]),
                storage_lower_m3=float(self.lower[i]), storage_upper_m3=float(self.upper[i]),
                inflow_m3s=float(row['inflow'][i]), external_inflow_m3s=float(row['external'][i]),
                release_m3s=float(row['release'][i]), turbine_m3s=float(row['turbine'][i]),
                spill_m3s=float(row['spill'][i]), power_mw=float(row['power'][i]),
                net_head_m=float(row['net_head_m'][i]), reference_head_m=float(self.reference_heads[i]),
                maximum_modeled_head_m=float(self.maximum_heads[i]),
                power_at_reference_head_same_turbine_mw=float(reference_power),
                head_contribution_at_executed_release=bool(threshold-row['power'][i] > tolerance and threshold-reference_power <= tolerance),
                feasible_release_low_m3s=float(row['low'][i]), feasible_release_high_m3s=float(row['high'][i]),
                water_budget_upper_m3s=water, downstream_upper_m3s=float(self.downstream_upper[i]),
                outlet_capacity_m3s=float(self.data.release_cap[i]), turbine_capacity_m3s=float(self.data.turbine_cap[i]),
                installed_capacity_mw=float(self.data.power_cap[i]),
                actual_other_guaranteed_violations=others['guaranteed'],
                actual_other_ecological_violations=others['ecological'])
            for term in self.requirements:
                service = self.service(i, term, water)
                record.update({term+'_'+key:value for key,value in service.items()})
            record['guaranteed_output_shortfall_mw'] = record['guaranteed_output_shortfall']
            record['ecological_flow_shortfall_m3s'] = record['ecological_flow_shortfall']
            records.append(record)
        return records


def analyze_transition(env, transition):
    """Return five flat diagnostic rows without modifying env, inputs or RNGs.

    Only violated services trigger alternative searches. The discrete grid is
    exact for the quadratic command mapping; power extrema additionally include
    stage/storage breakpoints, turbine saturation and quadratic stationary
    points. Best, minimum-meeting and continuous witnesses use full cascade
    physics with other original commands fixed. Errors abort the diagnostic,
    never label an actual physical trajectory as infeasible.
    """
    try:
        return _TransitionAnalysis(env, transition).records()
    except DispatchDiagnosticError:
        raise
    except Exception as exc:
        raise DispatchDiagnosticError('Same-state dispatch diagnostic failed at {}: {}'.format(
            transition.get('date', 'unknown'), exc)) from exc
