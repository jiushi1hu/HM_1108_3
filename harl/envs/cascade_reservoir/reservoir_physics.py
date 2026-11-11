"""Water balance, chain feasibility, and optional hard terminal reachability."""
import numpy as np
from harl.envs.cascade_reservoir.data_loader import HEAD_MODELS


class InfeasibleCascade(RuntimeError):
    pass


def close_storage_constraints(constraints):
    """Close difference bounds on [0, V0, V0+V1, ..., sum(V)].

    Entry (i,j) bounds prefix[i]-prefix[j]. Leading batch axes are allowed.
    A closed matrix describes exact projections, including coupled storage
    restrictions which cannot be represented by five independent intervals.
    """
    bounds = np.asarray(constraints, dtype=np.float64).copy()
    if bounds.shape[-2:] != (6, 6) or np.isnan(bounds).any():
        raise ValueError('Storage constraints require six by six difference bounds')
    for k in range(6):
        candidate = bounds[..., :, k, None]+bounds[..., None, k, :]
        # Do not amplify cancellation at large cumulative storages into
        # artificial negative cycles when fixing successive coordinates.
        np.copyto(bounds, candidate, where=candidate < bounds-1e-4)
        if (np.diagonal(bounds, axis1=-2, axis2=-1) < -1e-3).any():
            raise InfeasibleCascade('No trajectory can satisfy the terminal storage requirement')
        bounds[..., np.arange(6), np.arange(6)] = 0.
    return bounds


def _storage_box_constraints(lower, upper):
    lower, upper = np.asarray(lower), np.asarray(upper)
    bounds = np.full(lower.shape[:-1]+(6, 6), np.inf)
    bounds[..., np.arange(6), np.arange(6)] = 0.
    for i in range(5):
        bounds[..., i+1, i], bounds[..., i, i+1] = upper[..., i], -lower[..., i]
    return close_storage_constraints(bounds)


def terminal_storage_constraints(data, external, storage_lower, storage_upper, target, dt):
    """Exact backward reachability to the episode's initial storage.

    The known episode hydrology is used by the physical feasibility filter,
    not added to policy observations. No annual targets or storage resets are
    introduced. In cumulative storage, each outlet controls one coordinate:
    X'[i] = X[i] + dt * (sum(external[:i]) - release[i-1]).
    Backward projection and intersection with each day's envelope remain
    difference-bound sets, so every online command has a feasible completion.
    """
    target = np.asarray(target, dtype=np.float64)
    lower, upper = np.asarray(storage_lower), np.asarray(storage_upper)
    flows = np.asarray(external, dtype=np.float64)
    if (target.shape != (5,) or flows.ndim != 2 or flows.shape[1] != 5 or
            lower.shape != flows.shape or upper.shape != flows.shape or
            not all(np.isfinite(x).all() for x in (target, flows, lower, upper))):
        raise ValueError('Invalid terminal-storage planning inputs')
    if (target < lower[-1]).any() or (target > upper[-1]).any():
        raise InfeasibleCascade('Initial storage is outside the final-day permitted envelope')
    planned = np.empty((len(flows), 6, 6), dtype=np.float64)
    zone = _storage_box_constraints(target, target)
    for t in reversed(range(len(flows))):
        zone = close_storage_constraints(np.minimum(zone, _storage_box_constraints(lower[t], upper[t])))
        planned[t] = zone
        incoming = np.r_[0., np.cumsum(flows[t])*dt]
        minimum_change = incoming-np.r_[0., data.release_cap*dt]
        zone = zone-minimum_change[:, None]+incoming[None, :]
        np.fill_diagonal(zone, 0.)
    prefix = np.r_[0., np.cumsum(target)]
    if (prefix[:, None]-prefix[None, :] > zone+1e-3).any():
        raise InfeasibleCascade('Initial storage cannot reach the required terminal storage')
    planned.setflags(write=False)
    return planned


def constrained_releases(data, dt, storage, external, actions, constraints):
    """Choose releases from an exact reachable zone; supports batch axes."""
    shape = storage.shape
    bounds = np.asarray(constraints, dtype=np.float64)
    if bounds.shape not in ((6, 6), shape[:-1]+(6, 6)) or not np.isfinite(bounds).all():
        raise ValueError('Storage constraints must be finite matching six by six matrices')
    bounds = np.broadcast_to(bounds, shape[:-1]+(6, 6)).copy()
    prefix = np.concatenate((np.zeros(shape[:-1]+(1,)), np.cumsum(storage, axis=-1)), axis=-1)
    incoming = np.concatenate((np.zeros(shape[:-1]+(1,)), np.cumsum(external, axis=-1)*dt), axis=-1)
    maximum = prefix+incoming
    minimum = maximum-np.r_[0., data.release_cap*dt]
    bounds[..., :, 0] = np.minimum(bounds[..., :, 0], maximum)
    bounds[..., 0, :] = np.minimum(bounds[..., 0, :], -minimum)
    bounds = close_storage_constraints(bounds)
    inflow, release, low, high = [np.empty(shape, dtype=np.float64) for _ in range(4)]
    controls, executed = np.ones(shape, dtype=np.float32), actions.copy()
    for i in range(5):
        low[..., i] = np.maximum(0., (maximum[..., i+1]-bounds[..., i+1, 0])/dt)
        high[..., i] = np.minimum(data.release_cap[i], (maximum[..., i+1]+bounds[..., 0, i+1])/dt)
        if (low[..., i] > high[..., i]+1e-8).any():
            raise InfeasibleCascade('Empty terminal-compatible release interval')
        forced = high[..., i]-low[..., i] <= 1e-8
        controls[..., i] = np.where(forced, 0., 1.)
        executed[..., i] = np.where(forced, .5, executed[..., i])
        midpoint = np.minimum(data.release_cap[i], np.maximum(0., .5*(low[..., i]+high[..., i])))
        release[..., i] = np.where(forced, midpoint,
            low[..., i]+executed[..., i]**2*(high[..., i]-low[..., i]))
        inflow[..., i] = external[..., i]+(release[..., i-1] if i else 0.)
        selected = maximum[..., i+1]-release[..., i]*dt
        bounds[..., i+1, 0] = np.minimum(bounds[..., i+1, 0], selected)
        bounds[..., 0, i+1] = np.minimum(bounds[..., 0, i+1], -selected)
        bounds = close_storage_constraints(bounds)
    return inflow, release, low, high, controls, executed


class ReservoirPhysics:
    def __init__(self, data, config):
        self.data = data
        self.config = dict(config)
        if config.get("release_mapping", "quadratic") != "quadratic":
            raise ValueError("This experiment uses quadratic normalized-action mapping")
        self.dt = float(config["timestep_seconds"])
        if self.dt != 86400.0:
            raise ValueError("This model is daily")
        eta = float(config["efficiency"])
        if not 0 < eta <= 1:
            raise ValueError("Efficiency must lie in (0, 1]")
        density = float(config["water_density_kg_m3"])
        gravity = float(config["gravity_m_s2"])
        if not np.isfinite([density, gravity]).all() or density <= 0 or gravity <= 0:
            raise ValueError("Water density and gravity must be finite and positive")
        self.head_model = config.get("head_model", "fixed")
        if self.head_model not in HEAD_MODELS:
            raise ValueError("Unknown head model: " + str(self.head_model))
        self.rho_g_eta = density * gravity * eta
        self.head_data_status = "supplied_typical_head"
        if self.head_model == "fixed":
            self.coefficient = self.rho_g_eta * data.head / 1e6
            self.effective_turbine_cap = np.minimum(data.turbine_cap, data.power_cap / self.coefficient)
        else:
            if getattr(data, "head_sensitivity", None) is None:
                raise ValueError("Load the sensitivity data before selecting storage_sensitivity")
            if not np.allclose(data.head_sensitivity["efficiency"], eta, rtol=0., atol=1e-12):
                raise ValueError("Head sensitivity efficiency must match the physical configuration")
            self.head_data_status = "assumed_sensitivity_only"
        self.reference_energy = float(24 * data.power_cap.sum())

    def _forebay_levels(self, storage):
        storage = np.asarray(storage, dtype=np.float64)
        if storage.shape != (5,) or not np.isfinite(storage).all():
            raise ValueError("Generation requires a finite five-vector of storage")
        tolerance = np.maximum(1., 1e-10 * np.abs(storage))
        if ((storage < self.data.vmin - tolerance) | (storage > self.data.vmax + tolerance)).any():
            raise InfeasibleCascade("Generation storage violates operating bounds")
        # Absorb tolerated roundoff only in the elevation lookup. Water balance
        # and the physical storage vectors are never clipped or changed.
        return self.data.levels(np.clip(storage, self.data.vmin, self.data.vmax))

    def generation(self, release, storage_start=None, storage_end=None):
        """Daily mean-head generation shared by transitions and the reference.

        The sensitivity scenario has constant efficiency and a constant combined
        tailwater/loss offset, so no flow/head root solve is required. Installed
        capacity is enforced at the modeled daily mean head.
        """
        release = np.asarray(release, dtype=np.float64)
        if (release.ndim < 1 or release.shape[-1] != 5 or
                not np.isfinite(release).all() or (release < 0).any()):
            raise ValueError("Generation releases must be finite nonnegative five-vectors")
        result = {}
        if self.head_model == "fixed":
            head = self.data.head.copy()
            coefficient, cap = self.coefficient, self.effective_turbine_cap
        else:
            start_level = self._forebay_levels(storage_start)
            end_level = self._forebay_levels(storage_end)
            mean_level = (start_level + end_level) / 2.
            scenario = self.data.head_sensitivity
            head = (scenario["reference_net_head_m"] + scenario["head_slope_m_per_m"] *
                    (mean_level - scenario["assumed_reference_forebay_level_m"]))
            if not np.isfinite(head).all() or (head <= 0).any():
                raise ValueError("Calculated net head must be finite and positive")
            coefficient = self.rho_g_eta * head / 1e6
            cap = np.minimum(self.data.turbine_cap, self.data.power_cap / coefficient)
            result["forebay_level_mean_m"] = mean_level
        turbine = np.minimum(release, cap)
        power = turbine * coefficient
        result.update(net_head_m=head, effective_turbine_cap_m3s=cap.copy(),
                      turbine=turbine, spill=release-turbine, power=power, energy=24.*power)
        return result

    def _storage_bounds(self, storage_lower, storage_upper):
        """Validate END-of-day bounds without changing the global curve domain."""
        if storage_lower is None and storage_upper is None:
            return self.data.vmin, self.data.vmax
        if storage_lower is None or storage_upper is None:
            raise ValueError("Supply both daily storage_lower and storage_upper")
        lower, upper = [np.asarray(x, dtype=np.float64) for x in (storage_lower, storage_upper)]
        if (lower.shape != (5,) or upper.shape != (5,) or
                not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper).any()):
            raise ValueError("Daily storage bounds must be finite ordered five-vectors")
        if (lower < self.data.vmin).any() or (upper > self.data.vmax).any():
            raise ValueError("Daily storage bounds must lie within global operating bounds")
        return lower, upper

    def downstream_intervals(self, storage, external, storage_lower=None, storage_upper=None):
        storage_min, storage_max = self._storage_bounds(storage_lower, storage_upper)
        d = self.data
        lo, hi = np.zeros(5), d.release_cap.copy()
        for i in range(3, -1, -1):
            lo[i] = max(0., lo[i + 1] - external[i + 1] + (storage_min[i + 1] - storage[i + 1]) / self.dt)
            hi[i] = min(d.release_cap[i], hi[i + 1] - external[i + 1] + (storage_max[i + 1] - storage[i + 1]) / self.dt)
            if lo[i] > hi[i] + 1e-8:
                raise InfeasibleCascade("Empty downstream-compatible release interval")
        return lo, hi

    def transition(self, storage, external, actions, storage_lower=None, storage_upper=None,
                   storage_constraints=None):
        """Apply one action using only current inflow and its next-state bounds.

        Bounds constrain storage after this action. Optional cumulative-storage
        constraints also preserve reachability of the episode terminal target.
        Starting storage can exceed
        the next-state ceiling; it is checked against the preceding action's
        bounds by the environment. Future hydrology is read only when the
        environment constructs its optional terminal reachability constraints.
        """
        storage, external, actions = [np.asarray(x, dtype=np.float64) for x in (storage, external, actions)]
        if any(x.shape != (5,) or not np.isfinite(x).all() for x in (storage, external, actions)):
            raise ValueError("Expected three finite five-vectors")
        if ((actions < 0) | (actions > 1)).any() or (external < 0).any():
            raise ValueError("Invalid normalized action or external flow")
        d = self.data
        storage_min, storage_max = self._storage_bounds(storage_lower, storage_upper)
        if ((storage < d.vmin - 1) | (storage > d.vmax + 1)).any():
            raise InfeasibleCascade("Starting storage violates operating bounds")
        lower, upper = self.downstream_intervals(storage, external, storage_min, storage_max)
        inflow, release, low, high = [np.empty(5) for _ in range(4)]
        controls, executed_actions = np.ones(5, dtype=np.float32), actions.copy()
        for i in range(5) if storage_constraints is None else ():
            inflow[i] = external[i] + (release[i - 1] if i else 0.)
            low[i] = max(lower[i], inflow[i] + (storage[i] - storage_max[i]) / self.dt)
            high[i] = min(upper[i], inflow[i] + (storage[i] - storage_min[i]) / self.dt)
            if low[i] > high[i] + 1e-8:
                raise InfeasibleCascade("Empty feasible release interval at reservoir " + str(i))
            if high[i] - low[i] <= 1e-8:
                controls[i] = 0
                executed_actions[i] = 0.5
                # A tolerance-sized interval inversion can put the midpoint
                # just outside [0, outlet capacity]. Bound the actual flow;
                # storage is still computed from it and checked below.
                release[i] = min(d.release_cap[i], max(0., 0.5 * (low[i] + high[i])))
            else:
                release[i] = low[i] + executed_actions[i] ** 2 * (high[i] - low[i])
        if storage_constraints is not None:
            combined = np.minimum(storage_constraints, _storage_box_constraints(storage_min, storage_max))
            inflow, release, low, high, controls, executed_actions = constrained_releases(
                d, self.dt, storage, external, actions, combined)
        next_storage = storage + (inflow - release) * self.dt
        residual = next_storage - storage - (inflow - release) * self.dt
        tolerance = np.maximum(1., 1e-10 * np.abs(storage))
        if ((next_storage < storage_min - tolerance) | (next_storage > storage_max + tolerance)).any():
            raise InfeasibleCascade("Storage bound violation after execution")
        if (release < -1e-8).any() or (release > d.release_cap + 1e-8).any():
            raise InfeasibleCascade("Release capacity violation")
        generated = self.generation(release, storage, next_storage)
        power, energy = generated["power"], generated["energy"]
        if (np.abs(residual) > tolerance).any() or (power > d.power_cap + 1e-7).any():
            raise AssertionError("Water/energy calculation failed")
        system_residual = next_storage.sum() - storage.sum() - (external.sum() - release[-1]) * self.dt
        if abs(system_residual) > tolerance.sum():
            raise AssertionError("Cascade water conservation failed")
        result = dict(storage_start=storage.copy(), storage_end=next_storage, inflow=inflow,
                    external=external.copy(), actions=executed_actions, release=release,
                    low=low, high=high, controls=controls, residual=residual,
                    reward=float(energy.sum() / self.reference_energy),
                    system_residual=float(system_residual), **generated)
        if storage_lower is not None:
            result.update(storage_lower_m3=storage_min.copy(), storage_upper_m3=storage_max.copy())
        if storage_constraints is not None:
            result['storage_constraints'] = np.asarray(storage_constraints).copy()
        return result
