"""Morph local-view protocol. Source reconciliation is in docs/morph.md."""
from collections import defaultdict, deque
from copy import deepcopy
import math
from numbers import Integral
import random

import networkx as nx
import torch
from tdl.byzantine.attacks import start_snapshots, outgoing_payloads
from tdl.byzantine.diagnostics import delivery_summary

from tdl.decentralized.epidemic import aggregate_incoming, communication_summary
from tdl.decentralized.training import train_node_snapshots, summarize_local_training

SOURCE_REVISION = "57d73b921c317b0f0d7d9f4a71a6db8051beaf82"
MORPH_MODES = ("morph_code_faithful", "morph_paper_capped")


def protocol_mode(config):
    """Legacy configurations retain their previous capacity semantics."""
    return config.get("morph_mode", "morph_paper_capped" if config.get("outgoing_capacity") is not None else "morph_code_faithful")


def layerwise_cosine(left, right, parameter_names=None):
    """Equal tensor weights; skip zero norms, return -1 if none are usable.

    Callers pass named-parameter keys so model buffers are never included.
    Float64 normalized vectors prevent large finite float32 values overflowing.
    """
    names = list(left) if parameter_names is None else list(parameter_names)
    scores = []
    if not names:
        raise ValueError("At least one parameter is required")
    for name in names:
        if name not in left or name not in right or left[name].shape != right[name].shape:
            raise ValueError(f"Incompatible parameter: {name}")
        a, b = left[name].detach().cpu().double().flatten(), right[name].detach().cpu().double().flatten()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Parameters must be finite")
        if not len(a):
            continue
        # Scale before normalization, including for very large float64 inputs.
        sa, sb = a.abs().max(), b.abs().max()
        if sa == 0 or sb == 0:
            continue
        a, b = a / sa, b / sb
        score = torch.dot(a / a.norm(), b / b.norm()).item()
        scores.append(max(-1., min(1., score)))
    return sum(scores) / len(scores) if scores else -1.


def selection_probabilities(scores, beta, prefer_dissimilar=True):
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if not scores or any(not math.isfinite(s) or abs(s) > 1 for s in scores.values()):
        raise ValueError("Finite cosine scores in [-1, 1] required")
    sign = -1 if prefer_dissimilar else 1
    logits = {p: sign * beta * s for p, s in sorted(scores.items())}
    maximum = max(logits.values())
    weights = {p: math.exp(s - maximum) for p, s in logits.items()}
    total = sum(weights.values())
    return {p: w / total for p, w in weights.items()}


def sample_peers(scores, count, beta, rng, prefer_dissimilar=True):
    if count < 0 or count > len(scores):
        raise ValueError("Insufficient discovered candidates")
    remaining, selected = dict(scores), []
    for _ in range(count):
        probabilities = selection_probabilities(remaining, beta, prefer_dissimilar)
        peer = rng.choices(list(probabilities), weights=list(probabilities.values()), k=1)[0]
        selected.append(peer)
        del remaining[peer]
    return selected


class PeerView:
    """Owned local discovery, latest received models, and five-report histories."""
    def __init__(self, node, bootstrap):
        self.node = node
        self.known = set(bootstrap) - {node}
        self.models = {}
        self.seen_real = set()
        self.cache = {}
        # Observational provenance only; never shared or used for selection.
        self.score_sources = {}
        self.score_cached = {}
        self.history = defaultdict(lambda: deque(maxlen=5))

    def scores(self, local, parameter_names):
        fallback = sum(self.cache.values()) / len(self.cache) if self.cache else 0.
        # Frozen previous cache mirrors official indirect-estimator lookup.
        old = dict(self.cache)
        result = {}
        sources, cached = {}, {}
        for peer in sorted(self.known):
            if peer in self.models:
                result[peer] = layerwise_cosine(local, self.models[peer], parameter_names)
                sources[peer], cached[peer] = "direct", False
            elif peer not in self.seen_real and self.history[peer]:
                def intermediary_score(via):
                    if via in old:
                        return old[via]
                    if via in self.models:
                        old[via] = layerwise_cosine(local, self.models[via], parameter_names)
                        return old[via]
                    return fallback
                result[peer] = sum(intermediary_score(via) * sim for _, via, sim in self.history[peer]) / len(self.history[peer])
                sources[peer], cached[peer] = "indirect", False
            else:
                result[peer] = old.get(peer, fallback)
                sources[peer] = self.score_sources.get(peer, "fallback") if peer in old else "fallback"
                cached[peer] = peer in old
        self.cache.update(result)
        self.score_sources.update(sources)
        self.score_cached.update(cached)
        return result

    def metadata(self):
        return {"known_peers": sorted(self.known),
                "similarities": {p: s for p, s in sorted(self.cache.items()) if p in self.known and s != 0}}

    def receive(self, sender, state, metadata, round_number, record_reports=True):
        self.known.update(set(metadata["known_peers"]) - {self.node})
        self.known.add(sender)
        self.models[sender] = {k: v.detach().clone() for k, v in state.items()}
        if sender not in self.seen_real:
            self.history.pop(sender, None)
            self.seen_real.add(sender)
        if record_reports:
            for target, similarity in sorted(metadata["similarities"].items()):
                if target != self.node and target not in self.seen_real and target in self.known:
                    if not math.isfinite(similarity) or abs(similarity) > 1:
                        raise ValueError("Invalid metadata similarity")
                    self.history[target].append((round_number, sender, similarity))


def similarity_source_summary(edges, views):
    """Classify the score used, including cached measurements/estimates.

    Fractions use the named stage's guided edges as denominator. Random edges
    are excluded; an empty stage has null fractions rather than invented zeros.
    """
    counts = {source: 0 for source in ["direct", "indirect", "fallback"]}
    cached_counts = dict(counts)
    for sender, receiver in sorted(edges):
        source = views[receiver].score_sources.get(sender, "fallback")
        counts[source] += 1
        if views[receiver].score_cached.get(sender, False):
            cached_counts[source] += 1
    total = sum(counts.values())
    return {"guided_count": total, "counts": counts, "cached_counts": cached_counts,
            "fractions": {source: count / total if total else None for source, count in counts.items()},
            "four_way_fractions": {
                "fresh_direct": (counts["direct"] - cached_counts["direct"]) / total if total else None,
                "cached_direct": cached_counts["direct"] / total if total else None,
                "indirect": counts["indirect"] / total if total else None,
                "fallback": counts["fallback"] / total if total else None}}


def negotiate(preferences, scores, k, outgoing_capacity=None):
    """Receiver-proposing matching using only requester's reported similarity.

    Uncapped mode accepts every request, as official DPSGD_REQ does. Capped
    mode follows the paper: senders keep the most dissimilar requesters;
    rejected/displaced receivers continue through their local preference list.
    """
    nodes = set(preferences)
    if outgoing_capacity is not None and outgoing_capacity < 1:
        raise ValueError("Capacity must be positive or None")
    for receiver, peers in preferences.items():
        if len(peers) != len(set(peers)) or receiver in peers or not set(peers) <= nodes:
            raise ValueError("Invalid preference list")
        if not set(peers) <= set(scores[receiver]):
            raise ValueError("Missing local similarity")
    accepted = {p: set() for p in nodes}
    incoming = {p: set() for p in nodes}
    cursors = {p: 0 for p in nodes}
    requests = declines = replacements = 0
    request_edges = []
    while True:
        active = [r for r in sorted(nodes) if len(incoming[r]) < k and cursors[r] < len(preferences[r])]
        if not active:
            break
        for receiver in active:
            sender = preferences[receiver][cursors[receiver]]
            cursors[receiver] += 1
            requests += 1
            request_edges.append([sender, receiver])
            candidates = accepted[sender] | {receiver}
            # Lower reported similarity wins; deterministic node ID breaks ties.
            ranked = sorted(candidates, key=lambda r: (scores[r][sender], r))
            keep = set(ranked if outgoing_capacity is None else ranked[:outgoing_capacity])
            for rejected in candidates - keep:
                declines += 1
                if rejected in accepted[sender]:
                    incoming[rejected].remove(sender)
                    replacements += 1
            if receiver in keep:
                incoming[receiver].add(sender)
            accepted[sender] = keep
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(nodes))
    graph.add_edges_from((s, r) for r in sorted(nodes) for s in sorted(incoming[r]))
    missing = {r: k - len(incoming[r]) for r in sorted(nodes) if len(incoming[r]) != k}
    return graph, {"connection_requests": requests, "declined_requests": declines,
                   "replaced_connections": replacements, "incoming_deficits": missing,
                   "connection_request_edges": request_edges}


def serve_wanted_senders(wanted_senders, k):
    """Serve every local pull request, as official DPSGD_REQ execution does.

    The registry validates delivery IDs only. It cannot add candidates, match
    requests globally, rank receivers, impose outgoing capacity, or reject peers.
    """
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(wanted_senders))
    requests = []
    for receiver, peers in sorted(wanted_senders.items()):
        peers = list(peers)
        if len(peers) != k or len(set(peers)) != k or receiver in peers or not set(peers) <= set(wanted_senders):
            raise ValueError("Every receiver needs k distinct reachable wanted senders")
        for sender in peers:
            graph.add_edge(sender, receiver)
            requests.append([sender, receiver])
    return graph, {"connection_requests": len(requests), "declined_requests": 0,
                   "replaced_connections": 0, "incoming_deficits": {},
                   "connection_request_edges": requests}


def validate_config(config):
    for key, minimum in [("num_clients", 2), ("k", 1), ("seed", 0), ("topology_refresh_interval", 1),
                         ("random_peer_count", 0), ("rounds", 1), ("local_epochs", 1), ("batch_size", 1),
                         ("num_workers", 0), ("min_samples_per_client", 1)]:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"Invalid {key}")
    if config["k"] >= config["num_clients"] or config["random_peer_count"] > config["k"]:
        raise ValueError("Invalid peer counts")
    if config["selection_mode"] not in {"official_swap", "paper_resample"}:
        raise ValueError("Invalid selection_mode")
    if config["selection_mode"] == "official_swap" and config["random_peer_count"]:
        raise ValueError("Random component requires paper_resample")
    selection_probabilities({0: 0.}, config["beta"])
    cap = config.get("outgoing_capacity")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, Integral) or cap < 1):
        raise ValueError("Invalid outgoing_capacity")
    mode = protocol_mode(config)
    if mode not in MORPH_MODES:
        raise ValueError("Invalid morph_mode")
    if mode == "morph_code_faithful" and cap is not None:
        raise ValueError("morph_code_faithful serves all requesters; outgoing_capacity must be null")
    if mode == "morph_paper_capped" and cap is None:
        raise ValueError("morph_paper_capped requires outgoing_capacity")


class MorphProtocol:
    def __init__(self, bootstrap, config):
        validate_config(config)
        if bootstrap.is_multigraph() or set(bootstrap) != set(range(config["num_clients"])) or nx.number_of_selfloops(bootstrap):
            raise ValueError("Invalid bootstrap graph")
        self.config = dict(config)
        self.graph = bootstrap.to_directed()
        if any(self.graph.in_degree(n) != config["k"] for n in self.graph):
            raise ValueError("Bootstrap incoming degree must equal k")
        if not nx.is_weakly_connected(self.graph):
            raise ValueError("Bootstrap must be weakly connected")
        self.views = {n: PeerView(n, set(self.graph.predecessors(n)) | set(self.graph.successors(n))) for n in sorted(self.graph)}
        self.random_edges = set()
        self.wanted_senders = {n: set(self.graph.predecessors(n)) for n in self.graph}

    def refresh(self, snapshot, names, round_number, selection_observer=None):
        c = self.config
        if isinstance(round_number, bool) or not isinstance(round_number, Integral) or round_number < 1:
            raise ValueError("Round numbers start at one")
        refresh = round_number % c["topology_refresh_interval"] == 0
        diagnostics = {"topology_refreshed": refresh, "connection_requests": 0, "declined_requests": 0,
                       "replaced_connections": 0, "incoming_deficits": {}, "connection_request_edges": []}
        if not refresh:
            return diagnostics
        scores = {n: v.scores(snapshot[n], names) for n, v in sorted(self.views.items())}
        preferences, random_proposals, guided_proposals = {}, set(), set()
        for node, view in sorted(self.views.items()):
            rng = random.Random((c["seed"] + round_number) * len(self.views) + node)
            available = scores[node]
            current = set(self.graph.predecessors(node))
            random_count = c["random_peer_count"]
            # Random exploration uses discovered peers only, independently of scores.
            random_selected = rng.sample(sorted(available), random_count)
            random_proposals.update((p, node) for p in random_selected)
            guided = {p: s for p, s in available.items() if p not in random_selected}
            if c["selection_mode"] == "official_swap" and not random_count:
                add = {p: s for p, s in available.items() if p not in current}
                wanted = set(current)
                if len(current) > 1 and add:
                    wanted.add(sample_peers(add, 1, c["beta"], rng)[0])
                    wanted.remove(sample_peers({p: available[p] for p in current}, 1, c["beta"], rng, False)[0])
                preferred = sorted(wanted)
            else:
                preferred = random_selected + sample_peers(guided, c["k"] - random_count, c["beta"], rng)
            remaining = {p: s for p, s in available.items() if p not in preferred}
            preferences[node] = preferred + sample_peers(remaining, len(remaining), c["beta"], rng)
            guided_proposals.update((p, node) for p in preferred if p not in random_selected)
        if selection_observer is not None:
            diagnostics["peer_selection_observations"] = selection_observer(
                self.views, scores, preferences, random_proposals, guided_proposals)
        if protocol_mode(c) == "morph_code_faithful":
            wanted = {n: preferences[n][:c["k"]] for n in sorted(self.views)}
            graph, negotiation = serve_wanted_senders(wanted, c["k"])
        else:
            # Experimental paper-capacity path is preserved unchanged.
            graph, negotiation = negotiate(preferences, scores, c["k"], c["outgoing_capacity"])
        if negotiation["incoming_deficits"]:
            raise RuntimeError(f"Local-view matching cannot fill target k: {negotiation['incoming_deficits']}")
        # Capture all stages before gossip or peer-weight removal. No extra draws.
        diagnostics["guided_similarity_sources"] = {
            "initial_guided_slots": similarity_source_summary(guided_proposals, self.views),
            "guided_requests": similarity_source_summary(
                {tuple(edge) for edge in negotiation["connection_request_edges"]} - random_proposals, self.views),
            "accepted_guided_edges": similarity_source_summary(set(graph.edges()) - random_proposals, self.views)}
        for sender, receiver in negotiation["connection_request_edges"]:
            self.views[sender].known.add(receiver)
        for receiver, view in self.views.items():
            for removed in set(self.graph.predecessors(receiver)) - set(graph.predecessors(receiver)):
                view.models.pop(removed, None)
        self.graph = graph
        self.wanted_senders = {n: set(graph.predecessors(n)) for n in graph}
        self.random_edges = set(graph.edges()) & random_proposals
        return {**diagnostics, **negotiation}

    def exchange(self, snapshot, round_number):
        # Metadata and models are frozen before any node learns new identities.
        metadata = {n: deepcopy(v.metadata()) for n, v in sorted(self.views.items())}
        delta = self.config["topology_refresh_interval"]
        record = round_number > 1 and (round_number - 1) % max(1, delta - 1) == 0
        for receiver in sorted(self.views):
            for sender in sorted(self.graph.predecessors(receiver)):
                self.views[receiver].receive(sender, snapshot[sender], metadata[sender], round_number, record)


def run_morph_round(models, loaders, protocol, config, device, round_number, node_order=None, selection_observer=None):
    starts = start_snapshots(models, config)
    snapshot, counts, local = train_node_snapshots(models, loaders, config, device, round_number, node_order)
    outgoing = outgoing_payloads(starts, snapshot, config) if starts is not None else snapshot
    names = [n for n, _ in models[min(models)].named_parameters()]
    previous = protocol.graph.copy()
    diagnostics = protocol.refresh(snapshot, names, round_number) if selection_observer is None else protocol.refresh(snapshot, names, round_number, selection_observer)
    # Selection uses only previously received peer models and metadata.
    protocol.exchange(outgoing, round_number)
    updates = aggregate_incoming(outgoing, counts, protocol.graph, node_order=node_order,
                                 **({"self_snapshot": snapshot} if starts is not None else {}))
    similarities = [layerwise_cosine(snapshot[r], outgoing[s], names) for s, r in sorted(protocol.graph.edges())]
    for node in sorted(models):
        models[node].load_state_dict(updates[node])
    edges, old = set(protocol.graph.edges()), set(previous.edges())
    if "byzantine_nodes" in config or "attack" in config:
        diagnostics["byzantine_deliveries"] = delivery_summary(protocol.graph, config.get("byzantine_nodes", []), starts is not None)
    return {**summarize_local_training(round_number, counts, local), **diagnostics,
            "morph_mode": protocol_mode(config),
            "aggregation_rule": "uniform", "topology": communication_summary(protocol.graph, previous),
            "known_peer_counts": {str(n): len(v.known) for n, v in sorted(protocol.views.items())},
            "mean_known_peer_count": sum(len(v.known) for v in protocol.views.values()) / len(protocol.views),
            "topology_churn": len(edges ^ old) / len(edges | old),
            "mean_selected_peer_similarity": sum(similarities) / len(similarities),
            "random_peer_transmissions": len(protocol.random_edges),
            "guided_peer_transmissions": len(edges - protocol.random_edges),
            "directed_model_transmissions": len(edges)}
