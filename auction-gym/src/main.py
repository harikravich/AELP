import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

from Agent import Agent
from AuctionAllocation import * # FirstPrice, SecondPrice
from Auction import Auction
from Bidder import *  # EmpiricalShadedBidder, TruthfulBidder
from BidderAllocation import *  #  LogisticTSAllocator, OracleAllocator


def parse_kwargs(kwargs):
    parsed = ','.join([f'{key}={value}' for key, value in kwargs.items()])
    return ',' + parsed if parsed else ''


def parse_config(path):
    with open(path) as f:
        config = json.load(f)

    # Set up Random Number Generator
    rng = np.random.default_rng(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Number of runs
    num_runs = config['num_runs'] if 'num_runs' in config.keys() else 1

    # Max. number of slots in every auction round
    # Multi-slot is currently not fully supported.
    max_slots = 1

    # Technical parameters for distribution of latent embeddings
    embedding_size = config['embedding_size']
    embedding_var = config['embedding_var']
    obs_embedding_size = config['obs_embedding_size']

    # Expand agent-config if there are multiple copies
    agent_configs = []
    num_agents = 0
    for agent_config in config['agents']:
        if 'num_copies' in agent_config.keys():
            for i in range(1, agent_config['num_copies'] + 1):
                agent_config_copy = deepcopy(agent_config)
                agent_config_copy['name'] += f' {num_agents + 1}'
                agent_configs.append(agent_config_copy)
                num_agents += 1
        else:
            agent_configs.append(agent_config)
            num_agents += 1

    # First sample item catalog (so it is consistent over different configs with the same seed)
    # Agent : (item_embedding, item_value)
    agents2items = {
        agent_config['name']: rng.normal(0.0, embedding_var, size=(agent_config['num_items'], embedding_size))
        for agent_config in agent_configs
    }

    agents2item_values = {
        agent_config['name']: rng.lognormal(0.1, 0.2, agent_config['num_items'])
        for agent_config in agent_configs
    }

    # Add intercepts to embeddings (Uniformly in [-4.5, -1.5], this gives nicer distributions for P(click))
    for agent, items in agents2items.items():
        agents2items[agent] = np.hstack((items, - 3.0 - 1.0 * rng.random((items.shape[0], 1))))

    return rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size


def instantiate_agents(rng, agent_configs, agents2item_values, agents2items):
    # Store agents to be re-instantiated in subsequent runs
    # Set up agents with competitive intelligence capabilities
    agents = [
        Agent(rng=rng,
              name=agent_config['name'],
              num_items=agent_config['num_items'],
              item_values=agents2item_values[agent_config['name']],
              allocator=eval(f"{agent_config['allocator']['type']}(rng=rng{parse_kwargs(agent_config['allocator']['kwargs'])})"),
              bidder=eval(f"{agent_config['bidder']['type']}(rng=rng{parse_kwargs(agent_config['bidder']['kwargs'])})"),
              memory=(0 if 'memory' not in agent_config.keys() else agent_config['memory']),
              use_competitive_intel=agent_config.get('use_competitive_intel', True),
              quality_score=agent_config.get('quality_score', 7.0),
              daily_budget=agent_config.get('daily_budget', 1000.0))
        for agent_config in agent_configs
    ]

    for agent in agents:
        if isinstance(agent.allocator, OracleAllocator):
            agent.allocator.update_item_embeddings(agents2items[agent.name])

    return agents


def instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots, embedding_size, embedding_var, obs_embedding_size):
    return (Auction(rng,
                    eval(f"{config['allocation']}()"),
                    agents,
                    agents2items,
                    agents2item_values,
                    max_slots,
                    embedding_size,
                    embedding_var,
                    obs_embedding_size,
                    config['num_participants_per_round'],
                    enable_competitive_intel=config.get('enable_competitive_intel', True)),
            config['num_iter'], config['rounds_per_iter'], config['output_dir'])


def simulation_run():
    for i in range(num_iter):
        print(f'==== ITERATION {i} ====')

        for _ in tqdm(range(rounds_per_iter)):
            auction.simulate_opportunity()

        names = [agent.name for agent in auction.agents]
        net_utilities = [agent.net_utility for agent in auction.agents]
        gross_utilities = [agent.gross_utility for agent in auction.agents]

        result = pd.DataFrame({'Name': names, 'Net': net_utilities, 'Gross': gross_utilities})

        print(result)
        print(f'\tAuction revenue: \t {auction.revenue}')

        for agent_id, agent in enumerate(auction.agents):
            agent.update(iteration=i, plot=True, figsize=FIGSIZE, fontsize=FONTSIZE)

            agent2net_utility[agent.name].append(agent.net_utility)
            agent2gross_utility[agent.name].append(agent.gross_utility)

            agent2allocation_regret[agent.name].append(agent.get_allocation_regret())
            agent2estimation_regret[agent.name].append(agent.get_estimation_regret())
            agent2overbid_regret[agent.name].append(agent.get_overbid_regret())
            agent2underbid_regret[agent.name].append(agent.get_underbid_regret())

            agent2CTR_RMSE[agent.name].append(agent.get_CTR_RMSE())
            agent2CTR_bias[agent.name].append(agent.get_CTR_bias())

            if isinstance(agent.bidder, PolicyLearningBidder) or isinstance(agent.bidder, DoublyRobustBidder):
                agent2gamma[agent.name].append(torch.mean(torch.Tensor(agent.bidder.gammas)).detach().item())
            elif not agent.bidder.truthful:
                agent2gamma[agent.name].append(np.mean(agent.bidder.gammas))

            best_expected_value = np.mean([opp.best_expected_value for opp in agent.logs])
            agent2best_expected_value[agent.name].append(best_expected_value)

            print('Average Best Value for Agent: ', best_expected_value)
            
            # Monitor agent performance for safety
            if hasattr(agent, 'get_performance_metrics'):
                perf_metrics = agent.get_performance_metrics()
                print(f"Agent {agent.name} Performance: Win Rate: {perf_metrics['recent_win_rate']:.3f}, "
                      f"Budget Utilization: {perf_metrics['budget_utilization']:.3f}, "
                      f"Avg CPC: {perf_metrics['recent_avg_cpc']:.2f}")
            
            agent.clear_utility()
            agent.clear_logs()
            
        # Display competitive intelligence summary
        if auction.enable_competitive_intel and i % 5 == 0:  # Every 5 iterations
            ci_summary = auction.get_competitive_intelligence_summary()
            if 'error' not in ci_summary:
                print(f"\n--- Competitive Intelligence Summary (Iteration {i}) ---")
                market_metrics = ci_summary.get('competition_metrics', {})
                if market_metrics:
                    print(f"Market Win Rate: {market_metrics.get('win_rate', 0):.3f}")
                    print(f"Average CPC: ${market_metrics.get('average_cpc', 0):.2f}")
                    print(f"Competition Intensity: {market_metrics.get('competition_intensity', 0):.3f}")
                print(f"Total Auctions Analyzed: {ci_summary.get('market_overview', {}).get('total_auctions_analyzed', 0)}")
                print("---")

        auction_revenue.append(auction.revenue)
        auction.clear_revenue()
        
        # Safety check: monitor for anomalous behavior
        if i > 5:  # After some warm-up iterations
            for agent in auction.agents:
                if hasattr(agent, 'get_performance_metrics'):
                    metrics = agent.get_performance_metrics()
                    
                    # Check for budget violations
                    if metrics['budget_utilization'] > 1.1:  # 10% over budget
                        print(f"⚠️  SAFETY ALERT: Agent {agent.name} exceeded budget by {(metrics['budget_utilization']-1)*100:.1f}%")
                    
                    # Check for abnormally low win rates (possible overbidding issues)
                    if metrics['recent_win_rate'] < 0.05 and len(agent.win_rate_history) > 10:
                        print(f"⚠️  PERFORMANCE ALERT: Agent {agent.name} has very low win rate: {metrics['recent_win_rate']:.3f}")
                    
                    # Check for abnormally high CPCs (possible bidding war)
                    if len(agent.avg_cpc_history) > 10:
                        recent_cpc = metrics['recent_avg_cpc']
                        historical_avg = np.mean(agent.avg_cpc_history[:-10]) if len(agent.avg_cpc_history) > 20 else recent_cpc
                        if recent_cpc > historical_avg * 2.0:  # CPC doubled
                            print(f"⚠️  COST ALERT: Agent {agent.name} CPC increased significantly: ${recent_cpc:.2f} vs ${historical_avg:.2f}")

if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    # Parse configuration file
    rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size = parse_config(args.config)

    # Plotting config
    FIGSIZE = (8, 5)
    FONTSIZE = 14

    # Placeholders for summary statistics over all runs
    run2agent2net_utility = {}
    run2agent2gross_utility = {}
    run2agent2allocation_regret = {}
    run2agent2estimation_regret = {}
    run2agent2overbid_regret = {}
    run2agent2underbid_regret = {}
    run2agent2best_expected_value = {}

    run2agent2CTR_RMSE = {}
    run2agent2CTR_bias = {}
    run2agent2gamma = {}

    run2auction_revenue = {}
    
    # Safety monitoring across runs
    safety_alerts_count = 0
    total_budget_violations = 0

    # Repeated runs
    for run in range(num_runs):
        # Reinstantiate agents and auction per run
        agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
        auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots, embedding_size, embedding_var, obs_embedding_size)

        # Placeholders for summary statistics per run
        agent2net_utility = defaultdict(list)
        agent2gross_utility = defaultdict(list)
        agent2allocation_regret = defaultdict(list)
        agent2estimation_regret = defaultdict(list)
        agent2overbid_regret = defaultdict(list)
        agent2underbid_regret = defaultdict(list)
        agent2best_expected_value = defaultdict(list)

        agent2CTR_RMSE = defaultdict(list)
        agent2CTR_bias = defaultdict(list)
        agent2gamma = defaultdict(list)

        auction_revenue = []

        # Run simulation (with global parameters -- fine for the purposes of this script)
        simulation_run()

        # Store
        run2agent2net_utility[run] = agent2net_utility
        run2agent2gross_utility[run] = agent2gross_utility
        run2agent2allocation_regret[run] = agent2allocation_regret
        run2agent2estimation_regret[run] = agent2estimation_regret
        run2agent2overbid_regret[run] = agent2overbid_regret
        run2agent2underbid_regret[run] = agent2underbid_regret
        run2agent2best_expected_value[run] = agent2best_expected_value

        run2agent2CTR_RMSE[run] = agent2CTR_RMSE
        run2agent2CTR_bias[run] = agent2CTR_bias
        run2agent2gamma[run] = agent2gamma

        run2auction_revenue[run] = auction_revenue

    # Make sure we can write results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def measure_per_agent2df(run2agent2measure, measure_name):
        df_rows = {'Run': [], 'Agent': [], 'Iteration': [], measure_name: []}
        for run, agent2measure in run2agent2measure.items():
            for agent, measures in agent2measure.items():
                for iteration, measure in enumerate(measures):
                    df_rows['Run'].append(run)
                    df_rows['Agent'].append(agent)
                    df_rows['Iteration'].append(iteration)
                    df_rows[measure_name].append(measure)
        return pd.DataFrame(df_rows)

    def plot_measure_per_agent(run2agent2measure, measure_name, cumulative=False, log_y=False, yrange=None, optimal=None):
        # Generate DataFrame for Seaborn
        if type(run2agent2measure) != pd.DataFrame:
            df = measure_per_agent2df(run2agent2measure, measure_name)
        else:
            df = run2agent2measure

        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
        min_measure, max_measure = 0.0, 0.0
        sns.lineplot(data=df, x="Iteration", y=measure_name, hue="Agent", ax=axes)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        if optimal is not None:
            plt.axhline(optimal, ls='--', color='gray', label='Optimal')
            min_measure = min(min_measure, optimal)
        if log_y:
            plt.yscale('log')
        if yrange is None:
            factor = 1.1 if min_measure < 0 else 0.9
            # plt.ylim(min_measure * factor, max_measure * 1.1)
        else:
            plt.ylim(yrange[0], yrange[1])
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.legend(loc='upper left', bbox_to_anchor=(-.05, -.15), fontsize=FONTSIZE, ncol=3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.pdf", bbox_inches='tight')
        # plt.show()
        return df

    net_utility_df = plot_measure_per_agent(run2agent2net_utility, 'Net Utility').sort_values(['Agent', 'Run', 'Iteration'])
    net_utility_df.to_csv(f'{output_dir}/net_utility_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)

    net_utility_df['Net Utility (Cumulative)'] = net_utility_df.groupby(['Agent', 'Run'])['Net Utility'].cumsum()
    plot_measure_per_agent(net_utility_df, 'Net Utility (Cumulative)')

    gross_utility_df = plot_measure_per_agent(run2agent2gross_utility, 'Gross Utility').sort_values(['Agent', 'Run', 'Iteration'])
    gross_utility_df.to_csv(f'{output_dir}/gross_utility_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)

    gross_utility_df['Gross Utility (Cumulative)'] = gross_utility_df.groupby(['Agent', 'Run'])['Gross Utility'].cumsum()
    plot_measure_per_agent(gross_utility_df, 'Gross Utility (Cumulative)')

    plot_measure_per_agent(run2agent2best_expected_value, 'Mean Expected Value for Top Ad')

    plot_measure_per_agent(run2agent2allocation_regret, 'Allocation Regret')
    plot_measure_per_agent(run2agent2estimation_regret, 'Estimation Regret')
    overbid_regret_df = plot_measure_per_agent(run2agent2overbid_regret, 'Overbid Regret')
    overbid_regret_df.to_csv(f'{output_dir}/overbid_regret_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)
    underbid_regret_df = plot_measure_per_agent(run2agent2underbid_regret, 'Underbid Regret')
    underbid_regret_df.to_csv(f'{output_dir}/underbid_regret_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)

    plot_measure_per_agent(run2agent2CTR_RMSE, 'CTR RMSE', log_y=True)
    plot_measure_per_agent(run2agent2CTR_bias, 'CTR Bias', optimal=1.0) #, yrange=(.5, 5.0))

    shading_factor_df = plot_measure_per_agent(run2agent2gamma, 'Shading Factors')

    def measure2df(run2measure, measure_name):
        df_rows = {'Run': [], 'Iteration': [], measure_name: []}
        for run, measures in run2measure.items():
            for iteration, measure in enumerate(measures):
                df_rows['Run'].append(run)
                df_rows['Iteration'].append(iteration)
                df_rows[measure_name].append(measure)
        return pd.DataFrame(df_rows)

    def plot_measure_overall(run2measure, measure_name):
        # Generate DataFrame for Seaborn
        if type(run2measure) != pd.DataFrame:
            df = measure2df(run2measure, measure_name)
        else:
            df = run2measure
        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
        sns.lineplot(data=df, x="Iteration", y=measure_name, ax=axes)
        min_measure = min(0.0, np.min(df[measure_name]))
        max_measure = max(0.0, np.max(df[measure_name]))
        plt.xlabel('Iteration', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        factor = 1.1 if min_measure < 0 else 0.9
        plt.ylim(min_measure * factor, max_measure * 1.1)
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.pdf", bbox_inches='tight')
        # plt.show()
        return df

    auction_revenue_df = plot_measure_overall(run2auction_revenue, 'Auction Revenue')

    net_utility_df_overall = net_utility_df.groupby(['Run', 'Iteration'])['Net Utility'].sum().reset_index().rename(columns={'Net Utility': 'Social Surplus'})
    plot_measure_overall(net_utility_df_overall, 'Social Surplus')

    gross_utility_df_overall = gross_utility_df.groupby(['Run', 'Iteration'])['Gross Utility'].sum().reset_index().rename(columns={'Gross Utility': 'Social Welfare'})
    plot_measure_overall(gross_utility_df_overall, 'Social Welfare')

    auction_revenue_df['Measure Name'] = 'Auction Revenue'
    net_utility_df_overall['Measure Name'] = 'Social Surplus'
    gross_utility_df_overall['Measure Name'] = 'Social Welfare'

    columns = ['Run', 'Iteration', 'Measure', 'Measure Name']
    auction_revenue_df.columns = columns
    net_utility_df_overall.columns = columns
    gross_utility_df_overall.columns = columns

    pd.concat((auction_revenue_df, net_utility_df_overall, gross_utility_df_overall)).to_csv(f'{output_dir}/results_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.csv', index=False)
    
    # Save competitive intelligence insights if available
    if hasattr(auction, 'get_competitive_intelligence_summary'):
        try:
            ci_summary = auction.get_competitive_intelligence_summary()
            if 'error' not in ci_summary:
                with open(f'{output_dir}/competitive_intelligence_summary_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs.json', 'w') as f:
                    import json
                    # Convert datetime objects to strings for JSON serialization
                    def serialize_datetime(obj):
                        if hasattr(obj, 'isoformat'):
                            return obj.isoformat()
                        return obj
                    
                    json.dump(ci_summary, f, default=serialize_datetime, indent=2)
                print(f"Competitive intelligence summary saved to {output_dir}/competitive_intelligence_summary_*.json")
        except Exception as e:
            print(f"Warning: Could not save competitive intelligence summary: {e}")
