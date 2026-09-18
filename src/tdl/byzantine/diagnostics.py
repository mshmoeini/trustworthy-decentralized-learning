"""Offline observations; attacker identities never influence peer selection."""
import statistics


def observe_candidates(views, scores, preferences, random_proposals, guided_proposals, byzantine_nodes):
    """Observe exactly the local scores already computed for this refresh.

    Lower similarity ranks first. Ties retain rank bounds rather than inventing
    a preference. No scoring method or random generator is called here.
    """
    attackers = set(byzantine_nodes)
    records = []
    for receiver, view in sorted(views.items()):
        if receiver in attackers:
            continue
        candidates = scores[receiver]
        reserved = {s for s, r in random_proposals if r == receiver}
        for sender in sorted(attackers):
            known = sender in candidates
            score = candidates.get(sender)
            less = sum(v < score for v in candidates.values()) if known else None
            tied = sum(v == score for v in candidates.values()) if known else None
            source = view.score_sources.get(sender) if known else None
            cached = view.score_cached.get(sender) if known else None
            records.append({"receiver": receiver, "attacker": sender, "known": known,
                "directly_received_before_selection": sender in view.seen_real,
                "stored_received_model_available": sender in view.models,
                "similarity": score, "similarity_source": source, "cached": cached,
                "candidate_count": len(candidates), "rank_min": 1 + less if known else None,
                "rank_max": less + tied if known else None,
                "relative_rank": less / (len(candidates) - 1) if known and len(candidates) > 1 else None,
                "guided_eligible": known and sender not in reserved,
                "random_proposal": (sender, receiver) in random_proposals,
                "guided_proposal": (sender, receiver) in guided_proposals})
    honest = set(views) - attackers
    return {"timing": "Before current exchange; scores use honest current local weights and previous received payloads/cache",
        "rank_definition": "Ascending similarity among locally known candidates; 1 most dissimilar; ties shown as bounds",
        "honest_nodes_knowing_any_attacker": sum(bool(attackers & views[n].known) for n in honest),
        "honest_nodes_with_direct_receipt_of_any_attacker": sum(bool(attackers & views[n].seen_real) for n in honest),
        "honest_nodes_using_direct_similarity_to_any_attacker": len({r['receiver'] for r in records if r['similarity_source'] == 'direct'}),
        "guided_proposals_to_attackers": sum(r["guided_proposal"] for r in records),
        "random_proposals_to_attackers": sum(r["random_proposal"] for r in records),
        "receivers": records}


def delivery_summary(graph, byzantine_nodes, attack_active):
    attackers = set(byzantine_nodes)
    honest_edges = [(s, r) for s, r in graph.edges() if s in attackers and r not in attackers]
    count = sum(graph.out_degree(n) for n in attackers)
    total = graph.number_of_edges()
    return {"byzantine_nodes": sorted(attackers),
        "attacker_out_degrees": {str(n): graph.out_degree(n) for n in sorted(attackers)},
        "attacker_in_degrees": {str(n): graph.in_degree(n) for n in sorted(attackers)},
        "deliveries_originating_from_attackers": count,
        "fraction_deliveries_originating_from_attackers": count / total if total else 0.,
        "honest_receivers_selecting_attackers": len({r for _, r in honest_edges}),
        "attacker_to_honest_edges": [list(e) for e in sorted(honest_edges)],
        "poisoned_model_deliveries": count if attack_active else 0,
        "poisoned_model_delivery_fraction": count / total if attack_active and total else 0.,
        "poisoned_deliveries_to_honest_nodes": len(honest_edges) if attack_active else 0}


def impact_summary(evaluation, byzantine_nodes):
    attackers = set(byzantine_nodes)
    honest = [n["test_accuracy"] for n in evaluation["nodes"] if n["node_id"] not in attackers]
    malicious = {str(n["node_id"]): n["test_accuracy"] for n in evaluation["nodes"] if n["node_id"] in attackers}
    return {"attacker_test_accuracies": malicious,
        "honest_only_mean_accuracy": statistics.mean(honest) if honest else None,
        "honest_only_worst_accuracy": min(honest) if honest else None,
        "honest_only_accuracy_std": statistics.pstdev(honest) if honest else None,
        "global_mean_accuracy": evaluation["mean_node_test_accuracy"],
        "disagreement": evaluation["mean_pairwise_rms_parameter_distance"]}
