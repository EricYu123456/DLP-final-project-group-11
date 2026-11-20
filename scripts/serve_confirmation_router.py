"""Build a shared probe panel, fit capability embeddings and route prompts through the confirmed model."""

import argparse
import csv
import gc
import json
from functools import lru_cache
from itertools import islice
from pathlib import Path
import sys

import joblib
import numpy as np

from route_query import ROOT
from coldstart.conditional_capability import assign_regions, calibrated_scores, score_local
from coldstart.conditional_text import consensus_features, text_features
from coldstart.paired_distillation import lookup_pair_space, pair_scores
from coldstart.rescue_router import rescue_scores
from coldstart.skill_embedding import skill_coordinates, skill_scores


NOMINAL_METHODS = (
    "nominal-200", "nominal-una", "nominal-peers", "nominal-word", "nominal-word-gated",
    "nominal-ensemble", "nominal-ensemble-three", "nominal-ensemble-cautious",
    "nominal-ensemble-full", "nominal-ensemble-three-full", "nominal-ensemble-cautious-full",
)


def readout_file(path, cfg):
    parts = [cfg["algorithm"], cfg["name"]]
    parts.extend(f"{key}{int(cfg[key])}" for key in ("n_regions", "classes") if cfg.get(key) is not None)
    if cfg["algorithm"] in ("types", "contrast", "distilled"):
        parts.append(f"step{int(cfg['steps'])}")
    if cfg["algorithm"] == "cascade" and cfg["threshold"] >= 0.5:
        parts.append("cpu")
    return path / "serving" / ("_".join(parts) + ".joblib")


@lru_cache(maxsize=1)
def sentence_encoder(device="cpu"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
        device=device,
    )


def encode_prompts(texts, device="cpu", precision="float32"):
    encoder = sentence_encoder(device)
    if precision == "float32":
        return encoder.encode(texts, show_progress_bar=False)
    import torch

    with torch.autocast(device, dtype=torch.bfloat16):
        return encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False).float().cpu().numpy()


def encode(texts, proxies, rubrics, expert, x=None, known=None):
    from coldstart.answer_evidence import encode_evidence
    import torch

    if x is None:
        x = encode_prompts(texts)
    evidence, axes = encode_evidence(proxies, texts, 12, known=known), {}
    for name in sorted(rubrics):
        from coldstart.skill_axes import SkillAxes

        model = SkillAxes(name)
        axes[name] = model.encode_axes(texts, batch=12).reshape(len(texts), -1)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    frozen = None
    if expert:
        from coldstart.question_views import FrozenQuestionViews

        frozen = FrozenQuestionViews("expert").encode(texts, batch=12)
    return x, evidence, axes, frozen


def fitted_scores(cfg, payload, shared, raw, p, mem, coordinates, evidence):
    kind = cfg["algorithm"]
    dom = mem.argmax(1)
    member = np.eye(mem.shape[1])[dom]
    if kind in ("regional", "local"):
        reg = shared["region"]
        group = assign_regions(p, dom, reg["models"])
        means, gd, pop = reg["means"], reg["group_domain"], reg["pop"]
        if kind == "regional":
            from train_conditional_router import score_vectors

            return score_vectors(payload["z"], p, group, means, gd, pop, cfg)
        mat = score_local(payload["z"], p, group, means, pop, gd, cfg)
        return (
            calibrated_scores(mat, payload["calibration"], cfg["calibration"]) if cfg["calibration"] else mat
        )
    if kind == "skills":
        from train_skill_embeddings import blend_relative

        bank = shared["bank"]
        xx, rel = skill_coordinates(bank, raw, p, member)
        return skill_scores(payload["z"], xx, blend_relative(p, rel), member, bank["means"], cfg)
    if kind == "switch":
        from coldstart.switch_calibration import switch_scores

        return switch_scores(payload, p, dom, cfg)
    if kind == "survey":
        from coldstart.survey_policy import route_policy

        return route_policy(payload, p, dom, cfg)
    if kind == "distilled":
        from coldstart.distilled_text import distilled_scores

        return distilled_scores(payload, coordinates, dom, shared["student"], cfg["signal"])
    if kind == "coregionalized":
        from coldstart.coregionalized import panel_cross, query, scores

        model = shared["model"]
        pred, k = query(model, coordinates, raw, dom)
        cross = panel_cross(k, model["domains"], payload["idx"], len(dom))
        return scores(payload, pred, cross, dom, cfg)
    if kind == "answers":
        from coldstart.latent_answers import answer_arrays, predict_answers
        from train_latent_answers import assembled_scores

        data = answer_arrays([evidence[name] for name in cfg["proxies"].split()])
        values = predict_answers(shared["net"], payload["z"], data, dom)
        return assembled_scores(payload, values, dom, p, cfg)
    if kind == "reference":
        from coldstart.reference_outcomes import features, probability, scores, state_effects, state_scores

        verifier = shared["verifier"]
        x = features([evidence[name] for name in cfg["proxies"].split()], p, dom, len(verifier["means"]))
        if cfg.get("conditional_states"):
            return state_scores(
                verifier,
                payload["z"],
                probability(verifier, x),
                p,
                dom,
                state_effects(verifier["state_readouts"], x),
                cfg["signal"],
                cfg["calibration"],
            )
        return scores(
            verifier,
            payload["z"],
            probability(verifier, x),
            p,
            dom,
            dict(cfg, **payload.get("calibration", {})),
        )
    if kind == "nominal":
        from coldstart.latent_answers import answer_arrays
        from coldstart.nominal_response import add_choice_signal, outputs, query, scores

        raw_votes = answer_arrays([evidence[key] for key in cfg["proxies"].split()])
        q = query(shared["net"], raw_votes, dom)
        prob = outputs(shared["net"], payload["z"], q, dom)
        matrix = scores(payload, prob, p, dom, shared["means"], cfg)
        return add_choice_signal(matrix, shared["net"], payload, q, dom, raw_votes)
    if kind == "proxy-confusion":
        from coldstart.latent_answers import answer_arrays
        from coldstart.proxy_confusion import coordinates as judge_coordinates, judge_probs, score

        raw_votes = answer_arrays([evidence[name] for name in cfg["proxies"].split()])
        pack = shared["judge"]
        xx = judge_coordinates(
            pack, raw_votes, judge_probs(pack["judge"], raw_votes, dom, pack["domains"]), dom
        )
        return score(payload, xx, p, dom, shared["means"], cfg)
    if kind in ("paired", "rescue", "tabular"):
        basis = lookup_pair_space(raw, dom, shared["space"])
        if kind == "paired":
            return pair_scores(payload["z"], basis, cfg["temperature"])
        if kind == "rescue":
            return rescue_scores(
                shared["ctx"],
                payload["pf"],
                payload["pair"],
                basis,
                dom,
                shared["model"],
                cfg["alpha"],
                cfg["trees"],
            )
        from coldstart.tabular_capability import query_skills

        return payload["z"] @ query_skills(basis, dom, shared["tabular"]).T
    if kind in ("moments", "contrast"):
        proxies = (cfg.get("proxies") or "").split()
        if proxies:
            coordinates = np.column_stack((coordinates, consensus_features([evidence[k] for k in proxies])))
        if kind == "contrast":
            from coldstart.contrastive_routing import contrast_scores, features

            return contrast_scores(payload, features(shared["net"], coordinates, dom), dom, cfg)
        from coldstart.paired_moments import score_moments

        return score_moments(payload["z"], shared["model"].predict(coordinates, dom), dom, cfg["signal"])
    if kind == "forest":
        from coldstart.policy_forest import forest_scores, leaf_similarity

        model = shared["model"]
        pred, leaves = model.predict(raw, dom)
        return forest_scores(
            payload["z"], model.means, pred, dom, leaf_similarity(leaves, payload["support"]), cfg
        )
    if kind == "types":
        from coldstart.latent_skills import ItemTypeGate, gate_scores, route_types

        gate = ItemTypeGate(coordinates.shape[1], member.shape[1], int(cfg["classes"]))
        gate.load_state_dict(shared["gate"])
        probs = gate_scores(gate.eval(), coordinates, dom)
        return route_types(payload["z"], p, probs, dom, shared["means"], cfg)
    if kind == "joint":
        import torch
        from coldstart.joint_pool import JointPool

        dim = member.shape[1]
        net = JointPool(
            coordinates.shape[1], np.full((1, dim), 0.5), np.full((1, dim), 0.5), np.ones((1, 1)), 0.2
        ).eval()
        net.encoder.load_state_dict(payload["encoder"])
        with torch.inference_mode():
            phi, offset = net.coordinates(
                torch.from_numpy(coordinates), torch.from_numpy(member.astype(np.float32))
            )
            return (phi @ torch.from_numpy(payload["z"]).T + offset).sigmoid().numpy().T
    raise ValueError(f"Unknown routing algorithm {kind}")


def model_scores(out, texts, configs, profiles, packs, x=None, known=None):
    from train_joint_pool import joint_coordinates

    proxies = set().union(*(set(p["head"]["proxies"]) for p in packs))
    for cfg in configs:
        proxies.update((cfg.get("proxies") or "").split())
    rubrics = {cfg["axes"] for cfg in configs if cfg.get("axes")}
    x, evidence, axes, frozen = encode(texts, proxies, rubrics, any("coordinates" in p for p in packs), x, known)
    scores = []
    students = {}
    deferred = None
    for cfg, model, shared in zip(configs, profiles, packs):
        head = shared["head"]
        features = text_features(head["pack"], x, texts, [evidence[k] for k in head["proxies"]])
        p, mem = head["readout"].predict(x, features)
        raw = np.column_stack((features, axes[cfg["axes"]])) if cfg.get("axes") else features
        coords = joint_coordinates(raw, frozen, shared["coordinates"]) if "coordinates" in shared else None
        if cfg["algorithm"] == "coregionalized":
            coords = x
        if cfg["algorithm"] == "distilled":
            from train_distilled_text import load_model
            from coldstart.distilled_text import encode_tokens, tokenize
            import torch

            key = (cfg["name"], int(cfg["steps"]))
            if key not in students:
                net, limit = load_model(out / key[0] / f"encoder_{key[1]}.pt")
                seq, pad = tokenize(texts, limit)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                students[key] = encode_tokens(net.to(device), seq, pad, device)
                del net
            coords = students[key]
        if cfg["algorithm"] == "cascade":
            from coldstart.cascade import approximate, decision, teacher_config

            dom = mem.argmax(1)
            peer, prob = approximate(x, features, p, dom, shared["student"])
            mat, risk = decision(model["z"], peer, prob, dom, shared["means"], shared["student"], cfg)
            deferred = risk > cfg["threshold"]
            if deferred.any():
                selected = [text for text, yes in zip(texts, deferred) if yes]
                full, _ = model_scores(
                    out, selected, [teacher_config(cfg)], [model], [shared["teacher_model"]], x[deferred]
                )
                mat[:, deferred] = full[0]
            scores.append(mat)
        else:
            scores.append(fitted_scores(cfg, model, shared, raw, p, mem, coords, evidence))
    return scores, deferred


def portfolio_scores(args, texts, pool, packs, known=None, x=None):
    from coldstart.switch_calibration import rate_challengers

    if not packs:
        packs.extend(joblib.load(readout_file(args.out, cfg)) for cfg in pool["configs"])
    if x is None:
        x = encode_prompts(texts, precision=args.encoder_precision)
    scores, deferred = model_scores(args.out, texts, pool["configs"], pool["profiles"], packs, x, known)
    mat = np.average(scores, axis=0, weights=pool["weights"])
    return rate_challengers(
        mat,
        pool.get("uncertainty"),
        pool.get("allowance", 0.0),
        specialization=pool.get("specialization", 0.0),
    ), deferred


def route(args, texts, pool, packs, estimators=None):
    if pool.get("nominal_ensemble"):
        from ensemble_nominal_profiles import raw_scores

        mat, deferred = raw_scores(args, pool, texts), None
    elif "word_weights" in pool:
        from distill_nominal_embeddings import item_codes, load_student, read_scores
        from panel_text_kernel import score_margin

        student = load_student(args.out)
        mat = read_scores(pool, student, item_codes(student, texts))
        deferred = np.zeros(len(texts), dtype=bool)
        if pool["method"] == "nominal-word-gated":
            deferred = score_margin(mat) < pool["word_threshold"]
            if deferred.any():
                active = [t for t, needed in zip(texts, deferred) if needed]
                mat[:, deferred], _ = portfolio_scores(args, active, pool, packs)
    elif pool.get("algorithm") == "peer-states":
        from fit_peer_states import predict, question_features

        evidence = []
        if pool["helpers"]:
            from coldstart.answer_evidence import encode_evidence

            raw = encode_evidence(pool["helpers"], texts, 12)
            evidence = [raw[name] for name in pool["helpers"]]
        x = encode_prompts(texts, precision=args.encoder_precision)
        features, domain = question_features(pool["basis"], x, texts, evidence)
        mat, deferred = predict(pool, features, domain), None
    elif pool.get("algorithm") in ("tabular-panel", "text-kernel"):
        from fit_roster_panel import predict

        x = (
            None
            if pool["algorithm"] == "text-kernel"
            else encode_prompts(texts, args.panel_device, args.encoder_precision)
        )
        mat, deferred = predict(pool, x, texts, estimators), None
    elif "fast_panel" in pool or "student" in pool or "helper_student" in pool:
        known, encoded = None, None
        if "helper_student" in pool:
            from coldstart.answer_evidence import encode_evidence
            from distill_helper_gate import SMALL, SEMANTIC_BUDGETS, decision, inputs, semantic_inputs
            from distill_conditional_router import load_basis, project

            lexical_views = project(load_basis(args.out), texts)
            if pool["method"] in SEMANTIC_BUDGETS:
                encoded = encode_prompts(texts, precision=args.encoder_precision)
                view = semantic_inputs(args.out, encoded, texts, lexical_views)
            else:
                known = encode_evidence(SMALL, texts, 12)
                view = inputs(lexical_views, [known[k] for k in SMALL])
            mat, deferred = decision(pool["helper_student"], view, pool["method"])
        elif "student" in pool:
            from distill_conditional_router import early_scores, load_basis, project

            mat, deferred = early_scores(pool["student"], project(load_basis(args.out), texts))
        else:
            from fit_roster_panel import predict
            from panel_text_kernel import score_margin

            mat = predict(pool["fast_panel"], None, texts)
            deferred = score_margin(mat) < pool["accept_margin"]
        if deferred.any():
            active = [t for t, needed in zip(texts, deferred) if needed]
            if known is not None:
                known = {key: (prob[deferred], words[deferred]) for key, (prob, words) in known.items()}
            mat[:, deferred], _ = portfolio_scores(args, active, pool, packs, known, None if encoded is None else encoded[deferred])
    else:
        mat, deferred = portfolio_scores(args, texts, pool, packs)
    for index, v in enumerate(mat.T):
        winner = int(v.argmax())
        result = dict(
            selected=str(pool["model_names"][winner]).replace("__", "/"),
            probes=len(pool["probe_ids"]),
            method=args.method,
        )
        if deferred is not None:
            result["large_decoder_used"] = bool(deferred[index])
        if args.scores:
            result["scores"] = {
                str(name).replace("__", "/"): float(s) for name, s in zip(pool["model_names"], v)
            }
        print(json.dumps(result, ensure_ascii=False), flush=True)


def measurements(path, roster, bank_ids):
    pos = {int(v): i for i, v in enumerate(bank_ids)}
    if path:
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not {"model_name", "prompt_id", "response"}.issubset(rows[0] if rows else {}):
            raise ValueError("The response CSV needs model_name, prompt_id and response columns.")
    else:
        rows = []
    names = list(dict.fromkeys([*roster, *(r["model_name"].strip() for r in rows)]))
    if not names or any(not n for n in names):
        raise ValueError("Provide model names for the common panel.")
    y = np.full((len(names), len(bank_ids)), np.nan, dtype=np.float32)
    for r in rows:
        value = r["response"].strip()
        if not value:
            continue
        if value not in ("0", "1"):
            raise ValueError("Use 1 for a correct answer and 0 for an incorrect answer.")
        prompt_id = int(r["prompt_id"])
        if prompt_id not in pos:
            raise ValueError(f"Prompt {prompt_id} is outside the selected probe bank.")
        i, j = names.index(r["model_name"].strip()), pos[prompt_id]
        if np.isfinite(y[i, j]):
            raise ValueError("Each candidate can have only one recorded response per prompt.")
        y[i, j] = float(value)
    return names, y


def write_probes(path, names, ids, benchmarks, texts, grades):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "prompt_id", "benchmark", "prompt", "response"])
        for name, row in zip(names, grades):
            for item, domain, text, value in zip(ids, benchmarks, texts, row):
                writer.writerow([name, int(item), domain, text, int(value) if np.isfinite(value) else ""])


def onboard(args):
    from confirmation_models import ConfirmationModels
    from evaluate_capability_router import input_context
    from select_confirmation_router import orders

    d, s, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        manifest = dict(f)
    tr = manifest["training"]
    ids = d.prompt_ids[tr]
    names, y = measurements(args.responses, args.model_name, ids)
    only_panel = args.method in ("panel-120", "panel-200", "panel-text-120", "panel-text-200")
    peer_states = args.method.startswith("peer-states")
    active_peer = args.method in ("active-peer", "active-peer-stop")
    nominal = args.method in NOMINAL_METHODS
    replicated_panel = nominal or args.method == "roster-200"
    stop_reason = None
    if replicated_panel:
        with np.load(args.out / "nominal-replication/study.npz") as study:
            rows = study["panels"][list(study["seeds"]).index(args.seed)]
            training_rows = tr if args.method.endswith("-full") else study["train"]
        if getattr(args, "ensemble_probes", None) is not None:
            from evaluate_panel_budget import nested_order

            prefix = nested_order(d.domains[rows], args.seed)[: args.ensemble_probes]
            rows = rows[np.sort(prefix)]
        positions = {int(v): i for i, v in enumerate(tr)}
        idx = np.asarray([positions[int(v)] for v in rows])
        cfg = {"policy": "fixed-nominal", "probe_count": len(idx)}
    elif active_peer:
        from acquire_peer_states import acquire, active_context
        from threadpoolctl import threadpool_limits

        positions = {int(v): i for i, v in enumerate(tr)}
        with threadpool_limits(limits=3):
            ctx = active_context(args, d, s, manifest)
            rows, acquired_pool, _, stop_reason = acquire(
                ctx, d, names, args.seed, lambda chosen: y[:, [positions[int(v)] for v in chosen]],
                stop_early=args.method == "active-peer-stop",
            )
        idx = np.asarray([positions[int(v)] for v in rows])
        cfg = {"policy": "active-peer", "probe_count": 200}
    elif peer_states:
        cfg = {"policy": "balanced", "probe_count": 200}
    elif only_panel:
        cfg = {"policy": "strict-panel", "probe_count": int(args.method.rsplit("-", 1)[-1])}
    else:
        selection = args.selection
        full_method = args.method
        if args.method in ("distilled-conditional", "small-helper", "small-helper-economical", "semantic-conditional", "semantic-economical"):
            full_method = "conditional"
        if args.method == "gated-conditional":
            gate = json.loads(
                selection.with_name("conditional_cost_selection.json").read_text(encoding="utf-8")
            )
            full_method = gate["full_method"]
        if selection.name == "selection.json" and args.method.startswith("reference-"):
            selection = selection.with_name("reference_profiles.json")
        choice = json.loads(selection.read_text(encoding="utf-8"))[full_method]
        runner = ConfirmationModels(args, d, s, manifest, manifest["development"][:1], "development", True)
        runner.target_names = np.asarray(names)
        balanced = orders(d, s, manifest, runner.ref).orders["balanced", args.seed]
        cfg = choice["configs"][0]
    if only_panel or peer_states:
        positions = {v: i for i, v in enumerate(tr)}
        idx = np.asarray([positions[v] for v in manifest[f"balanced_{args.seed}"][: cfg["probe_count"]]])
    elif cfg["policy"] == "stop-evpi":
        from panel_stopping import select_prefix

        head = runner.head(cfg.get("readout", cfg["name"]))
        idx, _, stop_reason = select_prefix(
            runner.h,
            y,
            balanced,
            runner.dt,
            head["oof"],
            args.seed,
            int(cfg["probe_count"]),
            cfg["tolerance"],
            live=True,
        )
    elif cfg["policy"] == "adaptive":
        from coldstart.conditional_capability import choose_panel

        reg = runner.component(cfg)["region"]
        for n in (70, 120, 160, int(cfg["probe_count"])):
            n = min(n, int(cfg["probe_count"]))
            idx = choose_panel(
                runner.h, y, reg["groups"], reg["group_domain"], reg["means"], "adaptive", args.seed, n
            )
            if not np.isfinite(y[:, idx]).all() or n == int(cfg["probe_count"]):
                break
    elif cfg["policy"] == "opportunity" and not np.isfinite(y[:, balanced[:95]]).all():
        idx = balanced[:95]
    elif not (active_peer or replicated_panel):
        idx = runner.panel(cfg, y, args.seed, balanced)
    if np.any(np.isfinite(y[:, np.setdiff1d(np.arange(len(ids)), idx)])):
        raise ValueError("The responses include prompts outside this method's current probe panel.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "plan":
        from embed_new_model import read_prompts

        prompts = read_prompts(ids[idx], ROOT / "data/raw/question_order.csv")
        write_probes(
            args.output, names, ids[idx], [d.benchmark_names[v] for v in d.domains[tr[idx]]],
            [prompts[int(item)] for item in ids[idx]], y[:, idx],
        )
        print(
            json.dumps(
                dict(
                    questions_per_model=len(idx),
                    models=len(names),
                    file=str(args.output),
                    stop_reason=stop_reason,
                )
            )
        )
        return
    if (len(idx) != int(cfg["probe_count"]) and stop_reason not in ("evpi", "estimated-gain")) or not np.isfinite(y[:, idx]).all():
        raise ValueError("Complete the current panel for each candidate before fitting the pool.")
    if args.method == "roster-200":
        from fit_roster_panel import fit_pool
        from panel_capabilities import panel_text

        pool, _ = fit_pool(
            None, d.domains[tr[idx]], y[:, idx], panel_text(ids[idx], ROOT / "data/raw"),
            ids[idx], names, args.seed, text_only=True,
        )
    elif nominal:
        from replicate_nominal_profiles import package
        import torch

        torch.set_num_threads(2)
        local = {int(row): i for i, row in enumerate(training_rows)}
        selected = np.asarray([local[int(row)] for row in tr[idx]])
        graded = np.full((len(names), len(training_rows)), np.nan, np.float32)
        graded[:, selected] = y[:, idx]
        folder = args.out / "nominal-replication"
        if args.method.endswith("-full"):
            folder = folder / "full-ensemble"
        if args.method.startswith("nominal-ensemble") and (folder / "early-stop/ensemble_nets.joblib").is_file():
            folder = folder / "early-stop"
        filename = f"{args.method}_fit.joblib" if args.method in ("nominal-una", "nominal-peers") else "fit.joblib"
        shared = joblib.load(folder / filename)
        if args.method.startswith("nominal-ensemble"):
            from ensemble_nominal_profiles import METHODS, fit_ensemble

            count = METHODS[args.method.removesuffix("-full")][0]
            members = [joblib.load(folder / f"ensemble_member_{j}.joblib") for j in range(count)]
            pool = fit_ensemble(
                shared, members, graded, selected, names, ids[idx], args.seed, args.method, device=args.panel_device,
            )
            pool["shared_early_stop"] = folder.name == "early-stop"
        else:
            pool = package(shared, graded, selected, names, ids[idx], args.seed)
        if args.method.startswith("nominal-word"):
            from distill_nominal_embeddings import attach, calibration_data, load_student

            attach(pool, load_student(args.out), calibration_data(args.out), d.group_ids[tr[idx]])
    elif active_peer:
        pool = acquired_pool
    elif peer_states:
        from fit_peer_states import context, fit_pool
        from threadpoolctl import threadpool_limits

        mode = args.method.removeprefix("peer-states").strip("-") or "modern"
        with threadpool_limits(limits=3):
            ctx = context(args, d, s, manifest, mode)
            history = d.labels[np.ix_(ctx["ref"], tr[idx])]
            pool = fit_pool(ctx, history, y[:, idx], d.domains[tr[idx]], names, ids[idx], args.seed, mode)
    elif only_panel:
        from fit_roster_panel import fit_pool
        from panel_capabilities import panel_text

        texts = panel_text(ids[idx], ROOT / "data/raw")
        pool, _ = fit_pool(
            d.embeddings[tr[idx]],
            d.domains[tr[idx]],
            y[:, idx],
            texts,
            ids[idx],
            names,
            args.seed,
            device=args.panel_device,
            text_only=args.method.startswith("panel-text-"),
        )
    else:
        from coldstart.switch_calibration import paired_uncertainty

        profiles = [runner.score(config, y, idx, args.seed)[1] for config in choice["configs"]]
        pool = dict(
            configs=choice["configs"],
            weights=choice["weights"],
            profiles=profiles,
            model_names=np.array(names),
            probe_ids=ids[idx],
            seed=args.seed,
            refit=True,
            allowance=choice.get("allowance", 0.0),
            specialization=choice.get("specialization", 0.0),
            uncertainty=paired_uncertainty(runner.h, y, idx, runner.dt),
        )
        if args.method == "gated-conditional":
            from fit_roster_panel import fit_pool
            from panel_capabilities import panel_text

            fast, _ = fit_pool(
                None,
                d.domains[tr[idx]],
                y[:, idx],
                panel_text(ids[idx], ROOT / "data/raw"),
                ids[idx],
                names,
                args.seed,
                text_only=True,
            )
            pool.update(fast_panel=fast, accept_margin=gate["development"]["threshold"])
        elif args.method in ("distilled-conditional", "small-helper", "small-helper-economical", "semantic-conditional", "semantic-economical"):
            from distill_conditional_router import train_for_pool, training_inputs
            from threadpoolctl import threadpool_limits

            teacher = ConfirmationModels(
                args, d, s, manifest, np.r_[tr, manifest["development"]], "development", True
            )
            teacher.target_names = np.asarray(names)
            with threadpool_limits(limits=3):
                if args.method == "distilled-conditional":
                    views = training_inputs(args.out, d, manifest)
                    pool["student"], _ = train_for_pool(teacher, views, choice, y, idx, args.seed)
                else:
                    from distill_conditional_router import teacher_for_pool
                    from distill_helper_gate import SEMANTIC_BUDGETS, fit_student, training_data

                    semantic = args.method in SEMANTIC_BUDGETS
                    x = training_data(args, d, manifest, semantic)
                    target = teacher_for_pool(teacher, choice, y, idx, args.seed)
                    pool["helper_student"], _ = fit_student(target, x, len(tr), d.group_ids[manifest["development"]], semantic)
    pool["method"] = args.method
    joblib.dump(pool, args.output, compress=2)
    print(
        json.dumps(
            dict(
                questions_per_model=len(idx),
                models=len(names),
                file=str(args.output),
                stop_reason=stop_reason,
            )
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan, fit, serve = [sub.add_parser(k) for k in ("plan", "fit", "route")]
    for cmd in (plan, fit, serve):
        cmd.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
        cmd.add_argument(
            "--method",
            choices=(
                "primary",
                "conditional",
                *NOMINAL_METHODS,
                "ensemble-budget",
                "roster-200",
                "roster-stop",
                "roster-stop-economical",
                "gated-conditional",
                "distilled-conditional",
                "small-helper",
                "small-helper-economical",
                "semantic-conditional",
                "semantic-economical",
                "peer-states",
                "peer-states-text",
                "peer-states-live",
                "peer-states-mc",
                "peer-states-option",
                "peer-states-option-control",
                "active-peer",
                "active-peer-stop",
                "economical",
                "adaptive",
                "cascade",
                "cascade-cpu",
                "cascade-budget",
                "reference-modern",
                "reference-live",
                "reference-panel",
                "reference-joint-modern",
                "reference-joint-live",
                "panel-120",
                "panel-200",
                "panel-text-120",
                "panel-text-200",
            )
            + (("measured", "measured-binary", "measured-choice") if cmd is serve else ()),
        )
        cmd.add_argument(
            "--selection", type=Path, default=ROOT / "reports/routing/capability/confirmation/selection.json"
        )
    for cmd in (plan, fit):
        cmd.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
        cmd.add_argument("--seed", type=int)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--responses", type=Path, required=cmd is fit)
        cmd.add_argument("--model-name", action="append", default=[])
    serve.add_argument("--roster", choices=("audited_13", "expanded_37"), default="audited_13")
    serve.add_argument("--pool", type=Path)
    serve.add_argument("--scores", action="store_true")
    fit.add_argument(
        "--panel-device", choices=("cpu", "cuda"), default="cpu", help="Device for panel-only fitting."
    )
    serve.add_argument("--panel-device", choices=("cpu", "cuda"), default="cpu")
    serve.add_argument("--encoder-precision", choices=("float32", "bfloat16"), default="float32")
    serve.add_argument(
        "--batch-size", type=int, default=1, help="Number of incoming prompts to route together."
    )
    group = serve.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--stdin", action="store_true")
    group.add_argument("--stdin-json", action="store_true")
    args = parser.parse_args(argv)
    pool = joblib.load(args.pool) if args.action == "route" and args.pool else None
    if args.method is None and pool is not None:
        args.method = pool.get("method")
    if args.method is None:
        args.method = json.loads(args.selection.read_text(encoding="utf-8"))["default_method"]
    if args.action in ("plan", "fit"):
        if args.method == "ensemble-budget":
            budget = json.loads(args.selection.with_name("ensemble_probe_budget_selection.json").read_text(encoding="utf-8"))
            args.method, args.ensemble_probes = budget["method"], budget["questions"]
        available = (11, 23, 37)
        if args.method in NOMINAL_METHODS or args.method.startswith("roster-"):
            from replicate_nominal_profiles import SEEDS as available
        if args.seed is None:
            args.seed = available[0]
        if args.seed not in available:
            parser.error(f"Available onboarding seeds for {args.method}: {', '.join(map(str, available))}")
    if args.action == "route":
        from route_query import json_prompts

        if args.batch_size < 1:
            parser.error("--batch-size must be positive.")
        if args.stdin_json:
            source = json_prompts(sys.stdin, parser)
        elif args.stdin:
            source = (i.rstrip("\r\n") for i in sys.stdin if i.strip())
        else:
            source = iter([args.prompt])
        if pool is None:
            pool = joblib.load(args.out / f"{args.roster}_{args.method}.joblib")
        packs = []
        estimators = None
        if (
            pool.get("algorithm") == "tabular-panel"
            and len(pool["model_names"]) > 1
            and (args.prompt is None or args.panel_device == "cuda")
        ):
            from panel_tabular_skills import prepare_estimators

            estimators = prepare_estimators(pool["skill"], args.panel_device)
        while texts := list(islice(source, args.batch_size)):
            route(args, texts, pool, packs, estimators)
    elif args.method in ("roster-stop", "roster-stop-economical"):
        from stop_roster_budget import onboard as onboard_stopped

        onboard_stopped(args)
    else:
        onboard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
