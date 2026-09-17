"""Synthetic EL-Local protocol, barrier, RNG, and integration checks."""
from copy import deepcopy
import random

import networkx as nx
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tdl.decentralized import epidemic, runner, training


@pytest.mark.parametrize('k', [1, 3, 4, 7, 9])
def test_directed_protocol_counts_and_candidate_pool(k):
    graph = epidemic.sample_communication(10, k, 42, 1)
    assert graph.is_directed()
    assert not nx.number_of_selfloops(graph)
    assert graph.number_of_edges() == 10*k
    assert all(graph.out_degree(node) == k for node in graph)
    if k == 9:
        assert all(set(graph.successors(node)) == set(range(10))-{node} for node in graph)
    summary = epidemic.communication_summary(graph)
    assert summary['transmissions_per_round'] == 10*k
    assert summary['mean_in_degree'] == k
    assert summary['edge_jaccard_previous_round'] is None


def test_reproducibility_round_churn_and_global_rng_independence():
    random.seed(12)
    state = random.getstate()
    a = epidemic.sample_communication(10, 4, 42, 1)
    assert random.getstate() == state
    assert set(a.edges()) == set(epidemic.sample_communication(10, 4, 42, 1).edges())
    b = epidemic.sample_communication(10, 4, 42, 2)
    assert set(a.edges()) != set(b.edges())
    assert epidemic.communication_summary(b, a)['edge_jaccard_previous_round'] == len(set(a.edges()) & set(b.edges()))/len(set(a.edges()) | set(b.edges()))


def test_variable_incoming_and_zero_possible():
    graphs = [epidemic.sample_communication(10, 1, 42, r) for r in range(1, 11)]
    assert any(len(set(dict(g.in_degree()).values())) > 1 for g in graphs)
    assert any(g.in_degree(node)==0 for g in graphs for node in g)
    # For fixed sender, the allowed pool is not its old ring neighborhood.
    peers = set().union(*(set(g.successors(0)) for g in [epidemic.sample_communication(10, 4, 42, r) for r in range(1, 51)]))
    assert peers == set(range(1,10))


@pytest.mark.parametrize('args', [(10,0,42,1),(10,10,42,1),(10,True,42,1),(1,1,42,1),(10,4,-1,1),(10,4,42,0),(10,4.5,42,1)])
def test_invalid_protocol(args):
    with pytest.raises(ValueError): epidemic.sample_communication(*args)


@pytest.mark.parametrize('mode', epidemic.MODES)
def test_incoming_mean_zero_semantics_and_immutability(mode):
    graph = nx.DiGraph(); graph.add_nodes_from(range(4)); graph.add_edges_from([(0,1),(2,1),(1,3),(3,1)])
    states = {node: {'weight': torch.tensor([value])} for node,value in enumerate([2.,10.,20.,30.])}
    original = deepcopy(states)
    counts = {0:1,1:2,2:3,3:4}
    mixed = epidemic.aggregate_incoming(states,counts,graph,mode)
    expected = 62/4 if mode=='paper_epidemic' else (2+20+60+120)/10
    assert mixed[1]['weight'].item() == pytest.approx(expected)
    # Nodes 0 and 2 have no incoming; their outgoing neighbors must not contribute.
    for node in [0,2]:
        assert torch.equal(mixed[node]['weight'],states[node]['weight'])
        assert mixed[node]['weight'].data_ptr()!=states[node]['weight'].data_ptr()
    for node in states: assert torch.equal(states[node]['weight'],original[node]['weight'])
    reversed_mix=epidemic.aggregate_incoming(states,counts,graph,mode,[3,2,1,0])
    assert all(torch.equal(mixed[n]['weight'],reversed_mix[n]['weight']) for n in graph)


def tiny_nodes():
    torch.manual_seed(42)
    models=training.initialize_node_models(nn.Sequential(nn.Linear(2,4),nn.Dropout(0.2),nn.Linear(4,2)),5)
    loaders={node:DataLoader(TensorDataset(torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]]),torch.tensor([0,1,0,1])),batch_size=2,shuffle=True,generator=torch.Generator()) for node in range(5)}
    return models,loaders


@pytest.mark.parametrize('mode', epidemic.MODES)
def test_synchronous_barriers_model_isolation_and_order(mode,monkeypatch):
    models,loaders=tiny_nodes(); reverse,reverse_loaders=tiny_nodes()
    original={n:deepcopy(m.state_dict()) for n,m in models.items()}
    train=training.train_client; calls=[]
    def observe(model,*args):
        assert all(torch.equal(models[n].state_dict()[key],value) for n in models for key,value in original[n].items())
        assert all(model is not live for live in models.values())
        calls.append(1)
        return train(model,*args)
    monkeypatch.setattr(training,'train_client',observe)
    sample=epidemic.sample_communication
    def after_barrier(*args):
        assert len(calls)==5
        return sample(*args)
    monkeypatch.setattr(epidemic,'sample_communication',after_barrier)
    aggregate=epidemic.aggregate_incoming
    def frozen(states,counts,graph,*args):
        assert len(calls)==5
        assert all(torch.equal(models[n].state_dict()[key],value) for n in models for key,value in original[n].items())
        expected=aggregate(states,counts,graph,*args)
        assert any(not torch.equal(states[n][key],original[n][key]) for n in models for key in states[n])
        return expected
    monkeypatch.setattr(epidemic,'aggregate_incoming',frozen)
    config={'seed':42,'epidemic_k':2,'aggregation_mode':mode,'local_epochs':1,'learning_rate':0.1,'momentum':0.9,'optimizer':'SGD'}
    forward,graph=epidemic.run_epidemic_round(models,loaders,config,torch.device('cpu'),1)
    monkeypatch.setattr(training,'train_client',train);monkeypatch.setattr(epidemic,'sample_communication',sample);monkeypatch.setattr(epidemic,'aggregate_incoming',aggregate)
    backward,other=epidemic.run_epidemic_round(reverse,reverse_loaders,config,torch.device('cpu'),1,[4,3,2,1,0])
    assert forward==backward and set(graph.edges())==set(other.edges())
    assert all(torch.equal(models[n].state_dict()[key],reverse[n].state_dict()[key]) for n in models for key in models[n].state_dict())
    assert len({m[0].weight.data_ptr() for m in models.values()})==5
    # A second round begins with distinct node-specific models, not shared initialization.
    forward, graph = epidemic.run_epidemic_round(models, loaders, config, torch.device('cpu'), 2)
    backward, other = epidemic.run_epidemic_round(reverse, reverse_loaders, config, torch.device('cpu'), 2, [4,3,2,1,0])
    assert forward == backward and set(graph.edges()) == set(other.edges())
    assert all(torch.equal(models[n].state_dict()[key], reverse[n].state_dict()[key]) for n in models for key in models[n].state_dict())


@pytest.mark.parametrize('mode', epidemic.MODES)
def test_tiny_end_to_end_and_reproducibility(tmp_path,monkeypatch,mode):
    labels=torch.tensor([0,1]*20)
    dataset=TensorDataset(torch.nn.functional.one_hot(labels,2).float(),labels);dataset.targets=labels
    monkeypatch.setattr(runner,'load_fashion_mnist',lambda _: (dataset,dataset))
    monkeypatch.setattr(runner,'FashionMNISTCNN',lambda:nn.Linear(2,2))
    config=tmp_path/'config.yaml'
    config.write_text('seed: 42\nnum_clients: 5\nalpha: 1.0\nmin_samples_per_client: 1\ndata_dir: data\nbatch_size: 4\nlocal_epochs: 1\nrounds: 2\noptimizer: SGD\nlearning_rate: 0.1\nmomentum: 0.9\ndevice: cpu\nnum_workers: 0\nepidemic_k: 1\n')
    first=epidemic.run(config,tmp_path/'a.json',aggregation_mode=mode)
    assert epidemic.run(config,tmp_path/'b.json',aggregation_mode=mode)==first
    assert first['partition']['coverage_complete']
    assert first['initial_node_evaluation']['mean_pairwise_rms_parameter_distance']==0
    for round_ in first['rounds']:
        assert round_['directed_model_transmissions']==5
        assert len(round_['nodes'])==5
        assert sum(n['num_local_training_samples'] for n in round_['nodes'])==40
        zeros=round_['topology']['zero_in_degree_nodes']
        if zeros:
            assert round_['zero_in_degree_mean_test_accuracy']==pytest.approx(sum(n['test_accuracy'] for n in round_['nodes'] if n['node_id'] in zeros)/len(zeros))
        else: assert round_['zero_in_degree_mean_test_accuracy'] is None
    with pytest.raises(FileExistsError): epidemic.run(config,tmp_path/'a.json')
@pytest.mark.parametrize('override', [{'epidemic_k': 5}, {'rounds': 0}, {'aggregation_mode': 'static_weighted'}, {'seed': -1}])
def test_runner_invalid_config_before_data(tmp_path, monkeypatch, override):
    import yaml
    config = {'seed': 42, 'num_clients': 5, 'epidemic_k': 1, 'rounds': 2,
              'local_epochs': 1, 'batch_size': 4, 'num_workers': 0, 'min_samples_per_client': 1,
              'aggregation_mode': 'paper_epidemic'}
    config.update(override)
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(config))
    def forbidden(_):
        raise AssertionError('Invalid configuration must fail before dataset loading')
    monkeypatch.setattr(runner, 'load_fashion_mnist', forbidden)
    with pytest.raises(ValueError):
        epidemic.run(path, tmp_path / 'report.json')


@pytest.mark.parametrize('graph', [nx.Graph([(0, 1)]), nx.MultiDiGraph([(0, 1)]), nx.DiGraph([(0, 0), (0, 1)])])
def test_invalid_communication_graph(graph):
    with pytest.raises(ValueError):
        epidemic.aggregate_incoming({0: {'w': torch.tensor([1.])}, 1: {'w': torch.tensor([2.])}},
                                    {0: 1, 1: 1}, graph)
