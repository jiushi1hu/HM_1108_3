"""Orchestrate complete multi-year episodes and final held-out evaluation."""
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
import subprocess

from harl.envs.cascade_reservoir.episodes import episode_environment, positive_integer, training_windows
from harl.envs.cascade_reservoir.periodic_evaluation import (
    PERIODIC_EVALUATION_DAYS, select_periodic_windows)
from harl.envs.cascade_reservoir.training_metrics import log_episode, StepMetricsWriter, STEP_METRIC_FIELDS
from harl.envs.cascade_reservoir.training_sampler import TrainingWindowSampler, validate_sampling_mode
from harl.envs.cascade_reservoir.xld_diagnostics import (
    XLDDispatchWriter, XLD_DIAGNOSTIC_FIELDS, aggregate_xld_summaries)
from harl.envs.cascade_reservoir.trajectory_diagnostics import (
    TrajectoryDiagnosticsWriter, TRAJECTORY_DIAGNOSTIC_FIELDS)
from harl.runners.cascade_reservoir_runner import (
    CascadeReservoirRunner, value_normalization_update_schedule)
from harl.envs.cascade_reservoir.cascade_reservoir_env import actor_observation_features

ROOT=Path(__file__).resolve().parents[1]


def step_metrics_paths(episode_log, seeds):
    """Keep each seed's compact daily table beside its shared summary log."""
    if episode_log is None:
        return {}
    path=Path(episode_log)
    return {seed:path.with_name(path.stem+'_seed'+str(seed)+'_steps.csv') for seed in seeds}


def xld_dispatch_paths(episode_log, seeds, enabled=False):
    """One compact final-policy XLD diagnostic table per seed."""
    if not enabled or episode_log is None:
        return {}
    path=Path(episode_log)
    return {seed:path.with_name(path.stem+'_seed'+str(seed)+'_xld_dispatch.csv') for seed in seeds}


def plot_output_paths(episode_log=None, plot_output=None):
    """Use an explicit figure stem, or retain the log-derived figure names."""
    stem = plot_output if plot_output is not None else episode_log
    if stem is None:
        return {}
    path=Path(stem)
    return dict(training=path.with_name(path.stem+'_training.png'),
                evaluator=path.with_name(path.stem+'_evaluator_return.png'),
                violations=path.with_name(path.stem+'_violation_rates.png'))


def trajectory_diagnostics_paths(episode_log, seeds, enabled=False):
    """Optional linked daily learning signals, kept separate from legacy metrics."""
    if not enabled or episode_log is None:
        return {}
    path = Path(episode_log)
    return {seed: path.with_name(path.stem+'_seed'+str(seed)+'_trajectory.csv') for seed in seeds}


def checkpoint_directories(episode_log, seeds, enabled=False):
    """New per-seed directories for inference checkpoints; never training resume."""
    if not enabled or episode_log is None:
        return {}
    path = Path(episode_log)
    return {seed: path.with_name(path.stem+'_seed'+str(seed)+'_checkpoints') for seed in seeds}


def counterfactual_credit_paths(episode_log, seeds, enabled=False):
    if episode_log is None or not enabled:
        return {}
    path = Path(episode_log)
    return {seed:path.with_name(path.stem+'_seed'+str(seed)+'_counterfactual_credit.csv.gz') for seed in seeds}


def check_episode_inputs(data,physical,years,days,bins):
    """Validate the episode view and reset observations, without a future-flow solver."""
    env=episode_environment(data,physical,years,days,bins)
    if len(env.data.dates) != days or env.stop-env.start != days:
        raise ValueError('Episode inputs must contain exactly episode_days daily rows')
    if env.actor_observations().shape != (5,env.observation_size) or env.state().shape != (env.state_size,):
        raise ValueError('Episode observation dimensions do not match the configured layout')


def _final_xld_summary(runners, evaluations, algorithm, skip_final_evaluation):
    """Pool final-policy XLD counts, keeping training scenarios and held-out apart."""
    final_rounds=[runner.periodic_evaluation_history[-1] for runner in runners]
    fixed_summaries=[row.get('xld_dispatch_diagnostics') for row in final_rounds]
    if (all(row.get('status')=='ok' for row in final_rounds) and
            all(summary is not None for summary in fixed_summaries)):
        fixed=aggregate_xld_summaries(fixed_summaries)
    else:
        fixed=dict(status='incomplete',expected_seed_count=len(runners),
            complete_seed_count=sum(summary is not None for summary in fixed_summaries),
            message='Final fixed-window diagnosis is incomplete; no pooled rate is reported.')
    held_out=None
    if not skip_final_evaluation:
        if len(evaluations)!=len(runners) or any('xld_dispatch_diagnostics' not in row for row in evaluations):
            held_out=dict(status='incomplete',
                message='Held-out diagnosis is incomplete; no pooled rate is reported.')
        else:
            summaries=[dict(row['xld_dispatch_diagnostics'], status='ok',
                            episode_horizon_days=row['episode_horizon_days']) for row in evaluations]
            held_out=aggregate_xld_summaries(summaries)
    result=dict(fixed_training_windows=fixed,held_out=held_out)
    labels=dict(actor_missed_feasible_discrete='A meeting discrete command existed but was not selected',
                discrete_grid_gap='Only a continuous command could meet the requirement (grid gap)',
                same_state_continuous_unattainable='No continuous command could meet it in the reached state')
    for scope, summary in result.items():
        if summary is None:
            continue
        if summary['status']!='ok':
            print('XLD {} | {}: {}'.format(algorithm.upper(),scope,summary['message']),flush=True)
            continue
        rate=summary['violation_rate']
        print('XLD {} | {} | violations={}/{} ({})'.format(algorithm.upper(),scope,
            summary['violation_count'],summary['active_days'],
            'N/A' if rate is None else '{:.2%}'.format(rate)),flush=True)
        for group,label in labels.items():
            fraction=summary['group_fractions_among_violations'][group]
            print('  {}: {} ({} of XLD violation days)'.format(label,summary['group_counts'][group],
                'N/A' if fraction is None else '{:.2%}'.format(fraction)),flush=True)
        limits={key:value for key,value in summary['classification_counts'].items()
                if key.startswith('same_state_')}
        print('  Current-state limit categories: '+json.dumps(limits,sort_keys=True),flush=True)
        print('  Scope: same starting state and other reservoirs\' commands; '
              'this does not assess future water value or global inevitability.',flush=True)
    return result


def run_parallel_experiment(data,physical,config,args,*,runner_cls=CascadeReservoirRunner,
                            algorithm_name=None):
    if algorithm_name is not None:
        if not isinstance(algorithm_name,str) or algorithm_name.lower() not in ('happo','mappo'):
            raise ValueError('algorithm_name must be happo or mappo')
        algorithm_name=algorithm_name.lower()
    days=positive_integer(config['train']['episode_length'],'episode_days')
    count=positive_integer(config['train']['n_rollout_threads'],'parallel_envs')
    episodes=positive_integer(config['experiment']['episodes'],'episodes')
    interval=positive_integer(config['experiment']['log_interval'],'log_interval')
    eval_interval=positive_integer(config['experiment'].get('eval_interval',1),'eval_interval')
    if episodes % count:
        raise ValueError('episodes must be divisible by parallel_envs')
    windows=training_windows(data,days)
    sampling_mode=validate_sampling_mode(config['experiment'].get('window_sampling','coverage_balanced'))
    training_samplers={seed:TrainingWindowSampler(data,days,episodes,count,seed,mode=sampling_mode)
                       for seed in config['experiment']['seeds']}
    bins=config['model']['action_bins']
    credit_mode=config['algo'].get('happo_credit_assignment','team_gae')
    if credit_mode not in ('team_gae','counterfactual'):
        raise ValueError('happo_credit_assignment must be team_gae or counterfactual')
    if credit_mode == 'counterfactual' and getattr(runner_cls,'algorithm_name','happo') != 'happo':
        raise ValueError('Counterfactual credit is a HAPPO-only variant')
    xld_enabled=config['experiment'].get('xld_diagnostics',False)
    if not isinstance(xld_enabled,bool):
        raise ValueError('xld_diagnostics must be a boolean')
    if xld_enabled and data.service_requirements is None:
        raise ValueError('XLD diagnostics require loaded service requirements')
    trajectory_enabled=config['experiment'].get('trajectory_diagnostics',False)
    checkpoints_enabled=config['experiment'].get('save_checkpoints',False)
    if not isinstance(trajectory_enabled,bool) or not isinstance(checkpoints_enabled,bool):
        raise ValueError('trajectory_diagnostics and save_checkpoints must be booleans')
    if (trajectory_enabled or checkpoints_enabled) and args.episode_log is None:
        raise ValueError('Trajectory diagnostics and checkpoints require --episode-log')
    checkpoint_interval=config['experiment'].get('checkpoint_interval',0)
    if (isinstance(checkpoint_interval,bool) or not isinstance(checkpoint_interval,int) or
            checkpoint_interval < 0 or (checkpoint_interval and not checkpoints_enabled)):
        raise ValueError('checkpoint_interval must be nonnegative and requires checkpoint output when positive')
    daily_paths=step_metrics_paths(args.episode_log,config['experiment']['seeds'])
    xld_paths=xld_dispatch_paths(args.episode_log,config['experiment']['seeds'],xld_enabled)
    trajectory_paths=trajectory_diagnostics_paths(args.episode_log,config['experiment']['seeds'],trajectory_enabled)
    checkpoint_dirs=checkpoint_directories(args.episode_log,config['experiment']['seeds'],checkpoints_enabled)
    credit_paths=counterfactual_credit_paths(args.episode_log,config['experiment']['seeds'],credit_mode=='counterfactual')
    figure_paths={} if args.no_plot else plot_output_paths(
        args.episode_log, getattr(args, 'plot_output', None))
    periodic_windows=select_periodic_windows(data)
    output_paths=(([] if args.episode_log is None else [args.episode_log])+
                  list(daily_paths.values())+list(xld_paths.values())+list(figure_paths.values())+
                  list(trajectory_paths.values())+list(checkpoint_dirs.values())+list(credit_paths.values()))
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError('Experiment output paths must be distinct')
    for path in output_paths:
        if path.exists():
            raise ValueError('Output file already exists: '+str(path))
    evaluation_years=list(data.evaluation_years)
    print('Checking {} training windows{} ...'.format(len(windows),
          ' and the evaluation sequence' if not args.skip_final_evaluation else ''),flush=True)
    for years in list(windows.values())+([] if args.skip_final_evaluation else [evaluation_years]):
        check_episode_inputs(data,physical,years,days,bins)
    for window in periodic_windows:
        check_episode_inputs(data,physical,window['source_years'],PERIODIC_EVALUATION_DAYS,bins)
    sources=sorted(list((ROOT/'harl').rglob('*.py'))+list((ROOT/'examples').glob('*.py'))+
                   list((ROOT/'harl/configs').rglob('*.yaml')))
    if (not isinstance(data.hashes,dict) or not data.hashes or
            any(not isinstance(key,str) or not isinstance(value,str) or not value
                for key,value in data.hashes.items())):
        raise ValueError('Comparison context requires nonempty input data_hashes')
    shared_algo=dict(config['algo'])
    # Parameter sharing is the explicit architectural difference being tested.
    # Keep every other optimization, physical and budget setting comparable.
    shared_algo.pop('share_param',None)
    sharing=bool(getattr(runner_cls,'share_actor_parameters',False))
    critic_layout=getattr(runner_cls,'critic_reward_layout','mixed_by_reservoir_then_legacy_team')
    team_advantage=critic_layout=='shared_team'
    learning_objective=dict(
        advantage_source='shared_team' if team_advantage else 'individual_mixed',
        reward_formula=('0.5 * sum(local rewards) + 0.5 * global generation reward'
                        if team_advantage else '0.5 * own local reward + 0.5 * global generation reward'),
        critic_reward_layout=critic_layout,
        actor_value_channels=[0]*5 if team_advantage else list(range(5)),
        normalization=('once_over_common_rollout_rows' if team_advantage
                       else 'per_reservoir_over_common_rollout_rows'
                       if config['algo'].get('normalize_shared_advantages',False)
                       else 'per_reservoir_over_controllable_rows'))
    raw_observation_dim=len(actor_observation_features(physical.get('use_daily_level_limits',False)))
    actor_architecture=dict(parameter_sharing=sharing,network_count=1 if sharing else 5,
        optimizer_count=1 if sharing else 5,raw_observation_dim=raw_observation_dim,
        network_observation_dim=raw_observation_dim+(5 if sharing else 0),
        identity_features='one_hot_reservoir_id' if sharing else 'implicit_in_separate_actor',
        critic_output_dim=1 if team_advantage else 6)
    comparison_context=json.loads(json.dumps(dict(shared_settings=dict(
        episode_horizon_days=days,parallel_environments=count,episodes_per_seed=episodes,
        gamma=config['algo']['gamma'],action_bins=bins,seeds=config['experiment']['seeds'],
        actor_temporal_weighting='uniform_rollout_rows',
        evaluation_interval=eval_interval,
        training_years=list(data.training_years),evaluation_source_years=evaluation_years,
        window_sampling=sampling_mode,physical_config=physical,periodic_windows=periodic_windows,
        device=config['device'],train=config['train'],model=config['model'],algo=shared_algo),
        data_hashes=data.hashes),sort_keys=True,allow_nan=False))
    metadata=dict(record_type='run_configuration',collection_mode='five_year_parallel',
        value_normalization_update=value_normalization_update_schedule(
            getattr(runner_cls,'algorithm_name',algorithm_name or 'happo'),
            config['train']['use_valuenorm']),
        learning_reward_architecture=dict(alpha=0.5,
            actor_reward='0.5 * own local reward + 0.5 * global generation reward',
            local_reward='own power / installed capacity - own weighted relative service deficits '
                '- guaranteed_output_violation_weight * own guaranteed-output violation indicator; '
                'all service costs are disabled when reward penalties are disabled',
            system_reward='0.5 * sum(local rewards) + 0.5 * global generation reward',
            critic='one centralized network: five mixed-return outputs plus one legacy-team diagnostic output',
            actor_advantages='own mixed-return GAE and own value output; normalize each channel independently',
            actor_temporal_weighting='uniform_rollout_rows',
            actor_objective_reduction='mean over all rollout rows; policy active masks zero forced rows',
            actor_discount_scope='gamma remains in GAE, Monte Carlo returns and reward accounting; '
                'no additional gamma**t multiplier on policy or entropy objectives',
            legacy_metrics='existing reward, CSV metrics and plot curves retain their team-reward definitions',
            update_interpretation='individual-objective sequential clipped surrogates with original likelihood factors; '
                'the common-return HAPPO improvement guarantee is not asserted for system accounting'),
        episode_horizon_days=days,parallel_environments=count,collector_backend='spawn worker processes',
        episodes_per_seed=episodes,updates_per_seed=episodes//count,
        collector_steps_per_update=days*count,collector_steps_per_seed=days*episodes,
        training_years=list(data.training_years),training_year_count=len(data.training_years),
        evaluation_source_years=evaluation_years,evaluation_year_count=len(evaluation_years),
        evaluation_labels=list(data.evaluation_labels),training_window_starts=list(windows),
        training_window_source_years=windows,policy_source='final_training_update',
        training_window_sampling=dict(mode=sampling_mode,
            coverage_scope='completed training episodes only; evaluation and partial failed batches excluded',
            seed_plans=[sampler.metadata() for sampler in training_samplers.values()]),
        periodic_evaluation=dict(eval_interval=eval_interval,
            episode_horizon_days=PERIODIC_EVALUATION_DAYS,windows=periodic_windows,
            schedule='round 0, every eval_interval completed updates, and the final update',
            action_selection='deterministic argmax; fresh environments; no training samples',
            aggregation='arithmetic mean of all three complete windows; null metrics if any window fails',
            violation_rate_definition='violating active reservoir-days / all active reservoir-days; '
                'counts pooled over all three complete windows; undiscounted; null if a window fails or no requirement is active',
            violation_rate_units='fraction (plots display percent)',
            role='fixed training-only convergence diagnostics; held-out years are reserved for final evaluation'),
        figure_paths={name:str(path.resolve()) for name,path in figure_paths.items()},
        trajectory_diagnostics=dict(enabled=trajectory_enabled,
            scope='collected training trajectories only; one row per day, environment lane and reservoir',
            files={str(seed):str(path.resolve()) for seed,path in trajectory_paths.items()},
            columns=list(TRAJECTORY_DIAGNOSTIC_FIELDS),
            values='frozen before optimization and before ValueNorm statistics change',
            advantages='own mixed reward channel; exact configured normalization and actor-entry HAPPO factor',
            actor_temporal_weighting='uniform_rollout_rows',
            episode_time_weight='always 1.0; column retained for CSV compatibility',
            policy_objective_weight='policy active mask only; HAPPO likelihood factor is recorded separately',
            monte_carlo='realized sampled reward-to-go, not true expected values or true advantages',
            summary_windows=['all_days','first_14_days','october_may'],
            updates='observational only; no reward, action, optimizer or normalization changes'),
        policy_checkpoints=dict(enabled=checkpoints_enabled, interval_updates=checkpoint_interval,
            directories={str(seed):str(path.resolve()) for seed,path in checkpoint_dirs.items()},
            schedule='final completed update and every N updates if interval is positive',
            purpose='deterministic evaluation replay; training resume is not implemented'),
        xld_dispatch_diagnostics=dict(enabled=xld_enabled, reservoir='XLD',
            scope='every XLD day in the final scheduled periodic evaluation and final held-out evaluation only',
            training_rows_written=False, files={str(seed):str(path.resolve()) for seed,path in xld_paths.items()},
            daily_csv_enabled=bool(xld_paths), console_summary_enabled=xld_enabled,
            expected_rows_per_seed=(3*PERIODIC_EVALUATION_DAYS+(0 if args.skip_final_evaluation else days))
                if xld_paths else 0,
            expected_observed_days_per_seed=(3*PERIODIC_EVALUATION_DAYS+(0 if args.skip_final_evaluation else days))
                if xld_enabled else 0,
            columns=list(XLD_DIAGNOSTIC_FIELDS),
            counterfactual='same current state and action-date bounds; all other reservoir actions held fixed',
            interpretation='same-day attainability under the executed terminal reachability constraints; '
                'no action replacement or claim of long-term optimality',
            data_precision='10 significant digits in CSV; internal calculations and JSON summaries retain full precision'),
        calendar='fixed consecutive daily steps with leap days retained; storage carries across source-year boundaries',
        release_bounds_scope=('episode_terminal_reachability' if physical.get(
            'terminal_storage_equal_initial', False) else 'current_transition'),
        terminal_storage_requirement=('each reservoir equals its episode-start storage after the final decision only'
            if physical.get('terminal_storage_equal_initial', False) else 'disabled'),
        daily_bound_alignment='action_date=t constrains the state after action t; no additional date shift',
        step_logging=dict(enabled=args.episode_log is not None,
            format='compact_csv_v1',
            record_scope='one row per environment day; scalar team metrics over all five reservoirs',
            files={str(seed):str(path.resolve()) for seed,path in daily_paths.items()},
            columns=list(STEP_METRIC_FIELDS),
            episode_step_index='one-based; restarts for each episode',
            daily_reward_discounting='undiscounted; apply gamma**(episode_step-1) for episode totals',
            shortfall_aggregation='physical deficits are summed over the five reservoirs; penalties retain reward weights and means',
            missing_service_values='empty CSV cells; known zero deficits are written as 0',
            float_precision='10 significant digits in CSV only; internal calculations retain full precision',
            output='one CSV per seed; JSONL contains summaries and configuration'),
        evaluation='held-out representative years in configured order; storage carries across boundaries; '
                   'one sequence scored after training, without policy ranking',
        config=config,physical_config=physical,
        service_requirements=(data.service_requirements.metadata()
                              if data.service_requirements is not None else None),
        data_hashes=data.hashes,
        source_code_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    metadata['comparison_context']=comparison_context
    metadata['actor_architecture']=actor_architecture
    metadata['learning_objective']=learning_objective
    if credit_mode == 'counterfactual':
        learning_objective.update(advantage_source='counterfactual_shared_team',
            normalization='common_controllable_rms_without_centering')
        metadata['counterfactual_credit'] = dict(enabled=True,
            actor_estimator='one_step_counterfactual', critic_target='existing team GAE/discounted returns',
            q_definition='exact current system reward + gamma * frozen denormalized V(next state); terminal continuation is zero',
            advantage='selected Q minus old-policy probability-weighted Q over all own commands; other original commands fixed',
            normalization='one common RMS over controllable actor rows; no centering',
            batch_size=config['algo'].get('counterfactual_batch_size',4096),
            files={str(seed):str(path.resolve()) for seed,path in credit_paths.items()},
            role='model-assisted HAPPO variant; no monotonic-improvement or physical optimality claim')
    if team_advantage:
        metadata['learning_reward_architecture'].update(
            actor_reward=learning_objective['reward_formula'],
            critic='one centralized scalar value for the shared team return',
            actor_advantages='one team-reward GAE normalized once over common rollout rows and shared by all five actors',
            update_interpretation='sequential HAPPO clipped surrogates use one shared team advantage; '
                'the global reward is counted once in the team reward')
        metadata['trajectory_diagnostics']['advantages']='shared team reward/value channel 0; normalized once over common rollout rows; actor-entry HAPPO factor remains separate'
        if credit_mode == 'counterfactual':
            metadata['learning_reward_architecture'].update(
                actor_advantages='actor-specific frozen one-step system-Q minus its old-policy counterfactual baseline; common RMS scaling',
                update_interpretation='model-assisted HAPPO surrogate with actor-specific counterfactual credit and sequential likelihood factors')
            metadata['trajectory_diagnostics']['advantages']='frozen one-step counterfactual raw/normalized actor credit; critic GAE fields remain separate'
    if algorithm_name is not None:
        metadata['algorithm']=algorithm_name
    if algorithm_name=='mappo':
        metadata['learning_reward_architecture']['update_interpretation']=(
            'one shared actor trained on pooled reservoir samples and individual mixed-return '
            'clipped PPO surrogates with a centralized critic; '
            'system reward remains an accounting metric rather than the actor training target')
    print(json.dumps(metadata,indent=2,allow_nan=False),flush=True)
    if args.check_data_only:
        print('Data, episode calendar and observation checks passed; '
              'future trajectory feasibility is not certified. No workers or training started.',flush=True)
        return
    if not args.no_plot:
        subprocess.run([args.plot_python,str(ROOT/'examples/plot_cascade.py'),'--check-backend'],check=True)
    with ExitStack() as outputs:
        stream=None
        daily_writers={}
        xld_writers={}
        trajectory_writers={}
        credit_writers={}
        if args.episode_log is not None:
            args.episode_log.parent.mkdir(parents=True,exist_ok=True)
            stream=outputs.enter_context(args.episode_log.open('x',encoding='utf-8'))
            daily_writers={seed:StepMetricsWriter(outputs.enter_context(
                path.open('x',encoding='utf-8',newline=''))) for seed,path in daily_paths.items()}
            xld_writers={seed:XLDDispatchWriter(outputs.enter_context(
                path.open('x',encoding='utf-8',newline=''))) for seed,path in xld_paths.items()}
            trajectory_writers={seed:outputs.enter_context(TrajectoryDiagnosticsWriter(path))
                                for seed,path in trajectory_paths.items()}
            if credit_paths:
                from harl.envs.cascade_reservoir.counterfactual_logging import CounterfactualCreditWriter
                credit_writers={seed:outputs.enter_context(CounterfactualCreditWriter(path))
                                for seed,path in credit_paths.items()}
        log_episode(metadata,stream)
        runners=[]
        for seed in config['experiment']['seeds']:
            credit_options={'counterfactual_log':credit_writers[seed]} if seed in credit_writers else {}
            runner=runner_cls(data,physical,config,seed,episode_log=stream,
                                          step_log=daily_writers.get(seed),periodic_windows=periodic_windows,
                                          training_sampler=training_samplers[seed],xld_log=xld_writers.get(seed),
                                          trajectory_log=trajectory_writers.get(seed),
                                          checkpoint_dir=checkpoint_dirs.get(seed), **credit_options)
            runner.run(episodes,interval)
            runners.append(runner)
        evaluations=[] if args.skip_final_evaluation else [row for runner in runners for row in runner.final_evaluation()]
        xld_summary=(_final_xld_summary(runners,evaluations,algorithm_name or 'happo',
            args.skip_final_evaluation) if xld_enabled else None)
        result=dict(record_type='run_complete',seeds=config['experiment']['seeds'],
            episode_horizon_days=days,parallel_environments=count,episodes_per_seed=episodes,
            training_steps_per_seed=[runner.training_steps for runner in runners],
            completed_episodes_per_seed=[runner.completed_episodes for runner in runners],
            updates_per_seed=[runner.update_count for runner in runners],policy_source='final_training_update',
            periodic_evaluations_per_seed=[len(runner.periodic_evaluation_history) for runner in runners],
            periodic_evaluation_failures_per_seed=[sum(row['status']=='failed'
                for row in runner.periodic_evaluation_history) for runner in runners],
            training_sampling_coverage_per_seed=[runner.training_coverage_summary() for runner in runners],
            final_policy_metadata=[runner.evaluation_policy_metadata() for runner in runners],
            policy_checkpoints=[checkpoint for runner in runners for checkpoint in runner.checkpoint_history],
            evaluation_skipped=bool(args.skip_final_evaluation),evaluations=evaluations,
            learning_objective=learning_objective)
        if xld_summary is not None:
            result['xld_dispatch_summary']=xld_summary
        log_episode(result,stream)
        payload=json.loads(json.dumps(dict(episode_horizon_days=days,gamma=config['algo']['gamma'],
            seeds=config['experiment']['seeds'],updates_per_seed=episodes//count,
            periodic_evaluation=dict(windows=periodic_windows),
            histories=[runner.training_history for runner in runners],
            evaluation_histories=[runner.periodic_evaluation_history for runner in runners],
            algorithm=algorithm_name or 'happo',comparison_context=comparison_context,
            actor_architecture=actor_architecture,learning_objective=learning_objective),allow_nan=False))
        if xld_summary is not None:
            payload['xld_dispatch_summary']=xld_summary
        if not args.no_plot:
            plot_command=[args.plot_python,str(ROOT/'examples/plot_cascade_parallel.py')]
            if figure_paths:
                plot_command.extend(['--output',str(figure_paths['training']),
                                     '--evaluator-output',str(figure_paths['evaluator']),
                                     '--violations-output',str(figure_paths['violations'])])
            subprocess.run(plot_command,
                input=json.dumps(payload,allow_nan=False),text=True,check=True)
    print('Finished multi-year experiment. Training resume is not implemented; '
          'evaluation checkpoints are saved when --save-checkpoints is enabled. '
          'JSONL summaries and per-seed daily metrics CSVs are saved with --episode-log. '
          'Figures are saved with --plot-output or --episode-log when plotting is enabled.',flush=True)
    return payload
