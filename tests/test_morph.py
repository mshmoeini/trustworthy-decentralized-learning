from copy import deepcopy
import random

import networkx as nx
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from tdl.decentralized import morph, morph_runner, runner, training


def config(**overrides):
    values = dict(seed=42, num_clients=6, k=2, beta=10., topology_refresh_interval=1,
                  random_peer_count=1, selection_mode="paper_resample", outgoing_capacity=2,
                  rounds=3, local_epochs=1, batch_size=2, num_workers=0,
                  min_samples_per_client=1, learning_rate=.1, momentum=.9, optimizer="SGD")
    values.update(overrides)
    return values


def states(n=6):
    return {i: {"w": torch.tensor([1., float(i)])} for i in range(n)}


@pytest.mark.parametrize("sign,expected", [(1, 1), (-1, -1)])
def test_cosine_identity_and_opposite(sign, expected):
    a = {"w": torch.tensor([1., 2.])}
    assert morph.layerwise_cosine(a, {"w": sign*a["w"]}) == pytest.approx(expected)


def test_equal_tensor_weights_and_buffers_excluded():
    a = {"large": torch.ones(10000), "bias": torch.ones(1), "buffer": torch.ones(4)}
    b = {"large": torch.ones(10000), "bias": -torch.ones(1), "buffer": -torch.ones(4)}
    assert morph.layerwise_cosine(a, b, ["large", "bias"]) == pytest.approx(0.)


def test_zero_and_large_norms():
    assert morph.layerwise_cosine({"w": torch.zeros(2)}, {"w": torch.zeros(2)}) == -1
    assert morph.layerwise_cosine({"w": torch.tensor([1e300, 1e300], dtype=torch.float64)},
                                 {"w": torch.tensor([1e300, 1e300], dtype=torch.float64)}) == pytest.approx(1.)


@pytest.mark.parametrize("other", [{"w": torch.ones(3)}, {"w": torch.tensor([float('nan'), 1.])}, {}])
def test_invalid_cosine(other):
    with pytest.raises(ValueError):
        morph.layerwise_cosine({"w": torch.ones(2)}, other)


def test_local_view_isolation_and_owned_models():
    a, b = morph.PeerView(0, [1]), morph.PeerView(1, [0, 2])
    a.receive(1, states()[1], b.metadata(), 1)
    assert a.known == {1, 2} and b.known == {0, 2}
    source = states()[1]
    a.receive(1, source, b.metadata(), 2)
    source["w"].zero_()
    assert a.models[1]["w"].sum() > 0


def test_transitive_history_limit_no_ttl_and_direct_precedence():
    view = morph.PeerView(0, [1])
    for r in range(1, 8):
        view.receive(1, states()[1], {"known_peers": [2], "similarities": {2: r/10}}, r)
    assert [r for r, _, _ in view.history[2]] == [3, 4, 5, 6, 7]
    view.cache[1] = .5
    scores = view.scores(states()[0], ["w"])
    assert scores[2] == pytest.approx(.25)
    view.receive(2, states()[2], {"known_peers": [], "similarities": {}}, 1000)
    assert 2 not in view.history
    assert view.scores(states()[0], ["w"])[2] == pytest.approx(1/(5**.5))
    view.models.pop(2)
    assert view.scores(states()[0], ["w"])[2] == pytest.approx(1/(5**.5))


def test_beta_and_dissimilarity_probability():
    scores = {1: -.9, 2: .9, 3: 0.}
    p = morph.selection_probabilities(scores, 1.)
    assert p[1] > p[3] > p[2]
    assert morph.selection_probabilities(scores, 10.)[1] > p[1]
    assert list(morph.selection_probabilities(scores, 0.).values()) == pytest.approx([1/3]*3)
    assert sum(morph.selection_probabilities(scores, 1e4).values()) == pytest.approx(1)


def test_sampling_deterministic_stochastic_without_replacement():
    scores = {n: n/20 for n in range(20)}
    a = morph.sample_peers(scores, 5, 2., random.Random(42))
    assert a == morph.sample_peers(dict(reversed(list(scores.items()))), 5, 2., random.Random(42))
    assert len(set(a)) == 5
    assert a != morph.sample_peers(scores, 5, 2., random.Random(43))


def test_directed_uncapped_requests_and_capped_displacement():
    preferences = {0: [2, 1], 1: [2, 0], 2: [0, 1]}
    scores = {0: {2: .8, 1: .5}, 1: {2: -.8, 0: .5}, 2: {0: .5, 1: .5}}
    uncapped, _ = morph.negotiate(preferences, scores, 1)
    assert set(uncapped.edges()) == {(2, 0), (2, 1), (0, 2)}
    assert uncapped.out_degree(2) == 2 and uncapped.in_degree(2) == 1
    capped, metrics = morph.negotiate(preferences, scores, 1, 1)
    assert (2, 1) in capped.edges() and (2, 0) not in capped.edges()
    assert metrics["replaced_connections"] == 1 and metrics["declined_requests"] == 1
    assert not metrics["incoming_deficits"]
    assert all(capped.out_degree(n) == capped.in_degree(n) == 1 for n in capped)
    other, _ = morph.negotiate(dict(reversed(list(preferences.items()))), scores, 1, 1)
    assert set(capped.edges()) == set(other.edges())


def test_matching_deficit_reported_without_global_fallback():
    graph, metrics = morph.negotiate({0: [2], 1: [2], 2: [0]}, {0: {2: 0}, 1: {2: 1}, 2: {0: 0}}, 1, 1)
    assert metrics["incoming_deficits"] == {1: 1}
    assert (1, 0) not in graph.edges()


def test_bootstrap_no_global_candidate_leakage_and_gossip_barrier():
    p = morph.MorphProtocol(nx.cycle_graph(6), config())
    assert p.views[0].known == {1, 5}
    p.refresh(states(), ["w"], 1)
    assert set(p.graph.predecessors(0)) <= {1, 5}
    p.exchange(states(), 1)
    assert p.views[0].known == {1, 2, 4, 5}
    assert 3 not in p.views[0].known
    p.exchange(states(), 2)
    assert p.views[0].known == {1, 2, 3, 4, 5}


def test_refresh_interval_and_round_seed():
    p = morph.MorphProtocol(nx.cycle_graph(6), config(topology_refresh_interval=2, outgoing_capacity=None))
    edges = set(p.graph.edges())
    assert not p.refresh(states(), ["w"], 1)["topology_refreshed"]
    assert set(p.graph.edges()) == edges
    p.exchange(states(), 1)
    clone = deepcopy(p)
    assert p.refresh(states(), ["w"], 2) == clone.refresh(states(), ["w"], 2)
    assert set(p.graph.edges()) == set(clone.graph.edges())
    assert not p.refresh(states(), ["w"], 3)["topology_refreshed"]


def test_selection_changes_with_round_and_seed_for_identical_state():
    base = morph.MorphProtocol(nx.cycle_graph(6), config(outgoing_capacity=None, beta=0.))
    for n, view in base.views.items():
        view.known = set(range(6)) - {n}
    a, b, c = deepcopy(base), deepcopy(base), deepcopy(base)
    a.refresh(states(), ["w"], 2)
    b.refresh(states(), ["w"], 3)
    c.config["seed"] = 43
    c.refresh(states(), ["w"], 2)
    assert set(a.graph.edges()) != set(b.graph.edges())
    assert set(a.graph.edges()) != set(c.graph.edges())


def test_report_ingestion_gate_and_stale_history_retention():
    p = morph.MorphProtocol(nx.cycle_graph(6), config(topology_refresh_interval=5))
    p.views[1].cache[2] = .5
    p.exchange(states(), 1)
    assert not p.views[0].history[2]
    p.exchange(states(), 5)
    assert list(p.views[0].history[2]) == [(5, 1, .5)]
    p.views[0].cache[1] = .4
    p.exchange(states(), 6)  # gated off, retains earlier report
    assert list(p.views[0].history[2]) == [(5, 1, .5)]
    assert p.views[0].scores(states()[0], ["w"])[2] == pytest.approx(.2)


def test_deficient_refresh_raises_and_preserves_graph():
    p = morph.MorphProtocol(nx.cycle_graph(6), config(outgoing_capacity=1))
    old = set(p.graph.edges())
    with pytest.raises(RuntimeError, match="cannot fill target"):
        p.refresh(states(), ["w"], 1)
    assert set(p.graph.edges()) == old


def test_requester_identity_discovered_by_sender_after_selection():
    p = morph.MorphProtocol(nx.cycle_graph(6), config(outgoing_capacity=None, random_peer_count=0, beta=1e4))
    p.views[0].known.add(3)
    p.views[0].cache = {1: .9, 5: .9, 3: -.9}
    assert 0 not in p.views[3].known
    p.refresh(states(), ["w"], 1)
    assert (3, 0) in p.graph.edges()
    assert 0 in p.views[3].known


@pytest.mark.parametrize("random_count", [0, 1, 2])
def test_guided_random_hybrid_components(random_count):
    p = morph.MorphProtocol(nx.cycle_graph(6), config(random_peer_count=random_count, outgoing_capacity=None))
    p.refresh(states(), ["w"], 1)
    assert len(p.random_edges) == 6*random_count
    assert all(p.graph.in_degree(n) == 2 for n in p.graph)
    assert all(s in p.views[r].known for s, r in p.graph.edges())


def test_official_single_swap_and_similar_removal():
    c = config(selection_mode="official_swap", random_peer_count=0, outgoing_capacity=None, beta=1e4)
    p = morph.MorphProtocol(nx.cycle_graph(6), c)
    for n, view in p.views.items():
        view.known = set(range(6)) - {n}
        view.cache = {peer: (.9 if peer == (n+1)%6 else -.9 if peer == (n+3)%6 else 0.) for peer in view.known}
    old = {n: set(p.graph.predecessors(n)) for n in p.graph}
    p.refresh(states(), ["w"], 1)
    for n in p.graph:
        new = set(p.graph.predecessors(n))
        assert new - old[n] == {(n+3)%6}
        assert old[n] - new == {(n+1)%6}


def tiny_nodes():
    torch.manual_seed(42)
    models = training.initialize_node_models(nn.Sequential(nn.Linear(2, 4), nn.Dropout(.2), nn.Linear(4, 2)), 6)
    data = TensorDataset(torch.tensor([[1., 0.], [0., 1.]]*2), torch.tensor([0, 1]*2))
    loaders = {n: DataLoader(TensorDataset(data.tensors[0].repeat(n+1, 1), data.tensors[1].repeat(n+1)),
                            batch_size=2, shuffle=True, generator=torch.Generator()) for n in models}
    return models, loaders


def test_snapshot_barriers_uniform_aggregation_execution_order(monkeypatch):
    models, loaders = tiny_nodes()
    reverse, reverse_loaders = tiny_nodes()
    c = config(outgoing_capacity=None)
    p, q = morph.MorphProtocol(nx.cycle_graph(6), c), morph.MorphProtocol(nx.cycle_graph(6), c)
    original = {n: deepcopy(m.state_dict()) for n, m in models.items()}
    aggregate = morph.aggregate_incoming
    calls = []
    train = training.train_client
    def observed_train(model, *args):
        assert all(torch.equal(models[n].state_dict()[key], v) for n in models for key, v in original[n].items())
        calls.append(1)
        return train(model, *args)
    def observed_aggregate(snapshot, counts, graph, **kwargs):
        assert len(calls) == 6
        assert all(torch.equal(models[n].state_dict()[key], v) for n in models for key, v in original[n].items())
        updates = aggregate(snapshot, counts, graph, **kwargs)
        for node in graph:
            peers = sorted({node, *graph.predecessors(node)})
            for key in snapshot[node]:
                expected = sum(snapshot[peer][key]/len(peers) for peer in peers)
                torch.testing.assert_close(updates[node][key], expected)
        return updates
    monkeypatch.setattr(training, "train_client", observed_train)
    monkeypatch.setattr(morph, "aggregate_incoming", observed_aggregate)
    forward = morph.run_morph_round(models, loaders, p, c, torch.device("cpu"), 1)
    monkeypatch.setattr(training, "train_client", train)
    monkeypatch.setattr(morph, "aggregate_incoming", aggregate)
    backward = morph.run_morph_round(reverse, reverse_loaders, q, c, torch.device("cpu"), 1, list(reversed(range(6))))
    assert forward == backward
    for r in [2, 3]:
        assert morph.run_morph_round(models, loaders, p, c, torch.device("cpu"), r) == morph.run_morph_round(reverse, reverse_loaders, q, c, torch.device("cpu"), r, list(reversed(range(6))))
    assert all(torch.equal(models[n].state_dict()[key], reverse[n].state_dict()[key]) for n in models for key in models[n].state_dict())


@pytest.mark.parametrize("mode,capacity", [("morph_code_faithful", None), ("morph_paper_capped", 2)])
def test_tiny_end_to_end(tmp_path, monkeypatch, mode, capacity):
    labels = torch.tensor([0, 1]*20)
    dataset = TensorDataset(torch.nn.functional.one_hot(labels, 2).float(), labels)
    dataset.targets = labels
    monkeypatch.setattr(runner, "load_fashion_mnist", lambda _: (dataset, dataset))
    monkeypatch.setattr(runner, "FashionMNISTCNN", lambda: nn.Linear(2, 2))
    c = config(data_dir="unused", device="cpu", alpha=1., bootstrap_topology="ring_degree_2", morph_mode=mode, outgoing_capacity=capacity)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c))
    a = morph_runner.run(path, tmp_path / "a.json")
    assert morph_runner.run(path, tmp_path / "b.json") == a
    assert a["metadata"]["aggregation_rule"] == "uniform"
    assert a["metadata"]["morph_mode"] == mode
    assert a["partition"]["coverage_complete"]
    assert all(r["directed_model_transmissions"] == 12 for r in a["rounds"])
    with pytest.raises(FileExistsError):
        morph_runner.run(path, tmp_path / "a.json")


@pytest.mark.parametrize("override", [{"beta": -1}, {"k": 6}, {"random_peer_count": 3}, {"seed": True}, {"topology_refresh_interval": 0}, {"outgoing_capacity": 0}, {"selection_mode": "unknown"}])
def test_invalid_config(override):
    with pytest.raises(ValueError):
        morph.validate_config(config(**override))


def test_similarity_source_provenance_and_cached_scores():
    view = morph.PeerView(0, [1, 2, 3])
    view.scores(states()[0], ["w"])
    assert view.score_sources == {1: "fallback", 2: "fallback", 3: "fallback"}
    view.receive(1, states()[1], {"known_peers": [2, 3], "similarities": {2: .5}}, 1)
    view.scores(states()[0], ["w"])
    assert view.score_sources == {1: "direct", 2: "indirect", 3: "fallback"}
    summary = morph.similarity_source_summary({(1, 0), (2, 0), (3, 0)}, {0: view})
    assert summary["counts"] == {"direct": 1, "indirect": 1, "fallback": 1}
    assert list(summary["fractions"].values()) == pytest.approx([1/3]*3)
    view.models.pop(1)
    view.scores(states()[0], ["w"])
    assert view.score_sources[1] == "direct" and view.score_cached[1]
    empty = morph.similarity_source_summary(set(), {0: view})
    assert empty["guided_count"] == 0 and set(empty["fractions"].values()) == {None}


def test_similarity_diagnostic_does_not_change_selection_or_metadata(monkeypatch):
    c = config(outgoing_capacity=None)
    p = morph.MorphProtocol(nx.cycle_graph(6), c)
    q = deepcopy(p)
    for round_number in range(1, 11):
        diagnostics = p.refresh(states(), ["w"], round_number)
        with monkeypatch.context() as patch:
            patch.setattr(morph, "similarity_source_summary", lambda *args: {})
            q.refresh(states(), ["w"], round_number)
        assert set(p.graph.edges()) == set(q.graph.edges())
        assert p.random_edges == q.random_edges
        sources = diagnostics["guided_similarity_sources"]
        assert sources["initial_guided_slots"]["guided_count"] == 6
        assert sources["accepted_guided_edges"]["guided_count"] == len(set(p.graph.edges()) - p.random_edges)
        assert sum(sources["accepted_guided_edges"]["fractions"].values()) == pytest.approx(1)
        p.exchange(states(), round_number)
        q.exchange(states(), round_number)
        assert all(p.views[n].metadata() == q.views[n].metadata() for n in p.views)


def test_code_faithful_serves_every_request_with_unbalanced_outgoing_degree():
    wanted = {0: [1, 2], 1: [0, 2], 2: [0, 3], 3: [0, 4], 4: [0, 5], 5: [0, 1]}
    graph, diagnostics = morph.serve_wanted_senders(wanted, 2)
    assert all(graph.in_degree(n) == 2 for n in graph)
    assert graph.out_degree(0) == 5
    assert len(set(dict(graph.out_degree()).values())) > 1
    assert graph.number_of_edges() == 12
    assert diagnostics["connection_requests"] == 12
    assert diagnostics["declined_requests"] == 0
    assert not diagnostics["incoming_deficits"]
    for receiver, senders in wanted.items():
        assert set(graph.predecessors(receiver)) == set(senders)


def test_code_faithful_has_no_matching_or_capacity_rejection(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Code-faithful serving must not call matching/flow")
    monkeypatch.setattr(morph, "negotiate", forbidden)
    monkeypatch.setattr(nx, "maximum_flow", forbidden)
    c = config(morph_mode="morph_code_faithful", outgoing_capacity=None)
    p = morph.MorphProtocol(nx.cycle_graph(6), c)
    q = deepcopy(p)
    for round_number in range(1, 11):
        a = p.refresh(states(), ["w"], round_number)
        b = q.refresh(states(), ["w"], round_number)
        assert a == b and set(p.graph.edges()) == set(q.graph.edges())
        assert all(len(p.wanted_senders[n]) == p.graph.in_degree(n) == 2 for n in p.graph)
        assert all(s in p.views[r].known for s, r in p.graph.edges())
        assert a["declined_requests"] == 0
        p.exchange(states(), round_number)
        q.exchange(states(), round_number)


@pytest.mark.parametrize("mode,capacity", [("morph_code_faithful", 2), ("morph_paper_capped", None), ("ambiguous", None)])
def test_protocol_modes_reject_ambiguous_capacity_settings(mode, capacity):
    with pytest.raises(ValueError):
        morph.validate_config(config(morph_mode=mode, outgoing_capacity=capacity))


def test_capped_round6_regression_has_feasible_assignment_but_raises():
    import json
    from pathlib import Path
    fixture = json.loads((Path(__file__).parent / "fixtures/morph_capped_round6.json").read_text())
    preferences = {int(n): peers for n, peers in fixture["preferences"].items()}
    scores = {int(n): {int(p): s for p, s in row.items()} for n, row in fixture["scores"].items()}
    graph, diagnostics = morph.negotiate(preferences, scores, 4, 4)
    assert graph.number_of_edges() == 39 and diagnostics["incoming_deficits"] == {8: 1}
    assert graph.in_degree(8) == graph.out_degree(8) == 3
    feasible = nx.DiGraph()
    feasible.add_nodes_from(range(10))
    feasible.add_edges_from(fixture["feasible_assignment"])
    assert feasible.number_of_edges() == 40
    assert all(feasible.in_degree(n) == feasible.out_degree(n) == 4 for n in feasible)
    assert all(s in preferences[r] and s != r for s, r in feasible.edges())
    c = config(num_clients=10, k=4, beta=500., outgoing_capacity=4, morph_mode="morph_paper_capped")
    bootstrap = nx.watts_strogatz_graph(10, 4, 0)
    p = morph.MorphProtocol(bootstrap, c)
    for n, view in p.views.items():
        view.known = set(preferences[n])
        view.seen_real = set(preferences[n])
        view.cache = dict(scores[n])
    before = set(p.graph.edges())
    with pytest.raises(RuntimeError, match="cannot fill target k.*8"):
        p.refresh(states(10), ["w"], 6)
    assert set(p.graph.edges()) == before
