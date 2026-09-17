"""Local EL development matrix and explicitly labelled static comparisons."""
import argparse
import csv
import json
from pathlib import Path

from tdl.decentralized.epidemic import run


def summarize(results_dir):
    rows, diagnostics = [], []
    for alpha in [10.0,1.0,0.3]:
        for topology in ['ring_degree_4','random_regular_degree_4','small_world_degree_4','fully_connected']:
            path=results_dir/f'decentralized_{topology}_alpha_{alpha:g}.json'
            report=json.loads(path.read_text(encoding='utf-8'))
            final=report['rounds'][-1]
            rows.append(comparison_row(topology,'static_undirected','sample_count_weighted_self_plus_neighbors',alpha,final,report['topology']['directed_model_transmissions_per_round'],0,path.name))
        for prefix in ['epidemic_local','epidemic_sample_weighted_control']:
            path=results_dir/f'{prefix}_k4_alpha_{alpha:g}.json'
            if not path.exists():
                if prefix=='epidemic_local': raise FileNotFoundError(path)
                continue
            report=json.loads(path.read_text(encoding='utf-8'))
            final=report['rounds'][-1]
            static=json.loads((results_dir/f'decentralized_ring_degree_4_alpha_{alpha:g}.json').read_text(encoding='utf-8'))
            if report['partition']!=static['partition']:
                raise ValueError('Static and EL partitions differ.')
            keys=['seed','num_clients','local_epochs','rounds','batch_size','optimizer','learning_rate','momentum','num_workers','min_samples_per_client','data_dir','device','alpha']
            if any(report['config'][key]!=static['config'][key] for key in keys):
                raise ValueError('Comparison training configurations differ.')
            if report['initial_node_evaluation']!=static['initial_node_evaluation']:
                raise ValueError('Initial model evaluations differ.')
            rows.append(comparison_row(prefix+'_k4',report['metadata']['topology_type'],report['metadata']['aggregation_rule'],alpha,final,final['directed_model_transmissions'],final['topology']['zero_in_degree_count'],path.name))
            if prefix=='epidemic_sample_weighted_control':
                paper=json.loads((results_dir/f'epidemic_local_k4_alpha_{alpha:g}.json').read_text(encoding='utf-8'))
                if any(a['topology']!=b['topology'] for a,b in zip(report['rounds'],paper['rounds'],strict=True)):
                    raise ValueError('Control graphs must be exactly the same as paper EL graphs.')
    for k in [3,7]:
        path=results_dir/f'epidemic_local_k{k}_alpha_0.3.json'
        if path.exists():
            report=json.loads(path.read_text(encoding='utf-8'))
            for item in report['rounds']:
                diagnostics.append({'k':k,'round':item['round'],'aggregation_rule':report['metadata']['aggregation_rule'],
                                   'mean_accuracy':item['mean_node_test_accuracy'],'worst_accuracy':item['worst_node_test_accuracy'],
                                   'accuracy_std':item['node_test_accuracy_std'],'disagreement':item['mean_pairwise_rms_parameter_distance'],
                                   'topology':item['topology']})
    summary={'interpretation':'Seed 42, 10 nodes, three rounds, development-only; no confidence intervals or significance claims. Not a replication of the paper.',
             'comparison_caveat':'Paper EL vs static changes topology dynamics AND aggregation weighting. The sample-weighted EL variant is a separate control on identical directed graphs. Directionality/incoming degree also differ from static undirected graphs.',
             'accuracy_units':'fractions; std is population std across nodes', 'rows':rows,'connectivity_diagnostics':diagnostics}
    (results_dir/'epidemic_development_comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (results_dir/'epidemic_development_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows: print(f"{row['alpha']:g} | {row['algorithm']} | {row['aggregation_rule']} | mean={row['mean_accuracy']:.2%} | worst={row['worst_accuracy']:.2%} | std={row['accuracy_std']:.2%} | disagreement={row['disagreement']:.8f}")
    return summary


def comparison_row(algorithm,topology_type,aggregation_rule,alpha,final,traffic,zero,report):
    return {'algorithm':algorithm,'topology_type':topology_type,'aggregation_rule':aggregation_rule,'alpha':alpha,
            'round':final['round'],'mean_accuracy':final['mean_node_test_accuracy'],'worst_accuracy':final['worst_node_test_accuracy'],
            'accuracy_std':final['node_test_accuracy_std'],'mean_test_loss':final['mean_node_test_loss'],
            'disagreement':final['mean_pairwise_rms_parameter_distance'],'transmissions_per_round':traffic,
            'zero_in_degree_count':zero,'report':report}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/epidemic.yaml'))
    parser.add_argument('--results-dir',type=Path,default=Path('results'))
    parser.add_argument('--summary-only',action='store_true')
    parser.add_argument('--with-control',action='store_true')
    parser.add_argument('--diagnostics',action='store_true')
    args=parser.parse_args()
    if not args.summary_only:
        jobs=[('epidemic_local',4,alpha,'paper_epidemic') for alpha in [10.,1.,0.3]]
        if args.with_control: jobs += [('epidemic_sample_weighted_control',4,alpha,'epidemic_sample_weighted_control') for alpha in [10.,1.,0.3]]
        if args.diagnostics: jobs += [('epidemic_local',k,0.3,'paper_epidemic') for k in [3,7]]
        for prefix,k,alpha,mode in jobs:
            path=args.results_dir/f'{prefix}_k{k}_alpha_{alpha:g}.json'
            if path.exists(): raise FileExistsError(path)
        for prefix,k,alpha,mode in jobs:
            run(args.config,args.results_dir/f'{prefix}_k{k}_alpha_{alpha:g}.json',alpha,k,mode)
    summarize(args.results_dir)


if __name__=='__main__':main()
