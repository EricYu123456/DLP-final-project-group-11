"""Fit and apply the validation-selected capability models on a shared probe panel."""

from types import SimpleNamespace

import joblib
import numpy as np

from train_conditional_router import features_for, score_vectors, train as train_readout
from train_skill_embeddings import blend_relative, skill_measurements
from coldstart.conditional_capability import (
    assign_regions,
    calibrated_scores,
    calibrate_switches,
    fit_profile,
    local_fingerprints,
    make_regions,
    score_local,
)
from coldstart.paired_distillation import (
    behavioral_design,
    distill,
    infer_teacher,
    lookup_pair_space,
    pair_scores,
)
from coldstart.rescue_router import cohort_context, measured_pool, pair_profile, rescue_scores
from coldstart.skill_embedding import fit_skill_profile, fit_skill_space, skill_coordinates, skill_scores


class ConfirmationModels:
    def __init__(self, args, d, split, manifest, query, phase, refit):
        self.args, self.d, self.split, self.manifest = args, d, split, manifest
        self.train, self.query, self.phase = manifest["training"], query, phase
        self.dt = d.domains[self.train]
        self.refit = refit
        self.ref = np.r_[split.train_models, split.val_models] if refit else np.asarray(split.train_models)
        self.h = d.labels[np.ix_(self.ref, self.train)]
        self.heads, self.components, self.coordinates = {}, {}, {}

    def export_readout(self, cfg):
        """Keep the query-time decoder separate from candidate embeddings and historical training labels."""
        b = self.component(cfg)
        kind = b["kind"]
        pack = dict(head=b["head"]["readout"])
        if kind in ("regional", "local"):
            pack["region"] = {k: b["region"][k] for k in ("models", "group_domain", "means", "pop")}
        if kind == "skills":
            pack["bank"] = {k: v for k, v in b["bank"].items() if k != "cells"}
        if kind in ("paired", "rescue"):
            pack["space"] = b["space"]
        if kind == "rescue":
            pack["ctx"] = {k: b["ctx"][k] for k in ("absolute", "means")}
            pack["model"] = b["model"]
        if kind == "tabular":
            component = self.component(
                dict(name=cfg["pairs"], algorithm="paired", readout=cfg["readout"], axes=cfg.get("axes"))
            )
            pack["space"], pack["tabular"] = component["space"], b["pack"]
        if kind in ("joint", "types", "moments", "contrast"):
            with np.load(self.args.out / cfg.get("coordinates", cfg["name"]) / "coordinate_pack.npz") as f:
                pack["coordinates"] = dict(f)
        if kind == "types":
            import torch

            base = self.args.out / cfg["name"] / ("refit" if self.refit else "dev")
            pack["gate"] = torch.load(
                base / f"gate_{int(cfg['classes'])}_{int(cfg['steps'])}.pt",
                map_location="cpu",
                weights_only=True,
            )
            pack["means"] = b["pack"]["means"]
        if kind in ("moments", "forest", "coregionalized"):
            pack["model"] = b["model"]
        if kind in ("answers", "contrast"):
            pack["net"] = b["net"]
        if kind == "distilled":
            pack["student"] = {k: v for k, v in b["student"].items() if k not in ("history", "means")}
        if kind == "proxy-confusion":
            pack.update(judge=b["judge"], means=b["ref"]["means"])
        if kind == "nominal":
            pack.update(net=b["net"], means=b["means"])
        if kind == "reference":
            pack["verifier"] = b["verifier"]
        if kind == "cascade":
            from coldstart.cascade import teacher_config

            pack.update(student=b["student"], means=b["pack"]["means"])
            if cfg["threshold"] < 0.5:
                pack["teacher_model"] = self.export_readout(teacher_config(cfg))
        return pack

    def head(self, name):
        if name in self.heads:
            return self.heads[name]
        base = self.args.out / name
        original = joblib.load(base / "readout.joblib")
        directory = base / "refit75" if self.refit else base
        if not (directory / "readout.joblib").exists():
            directory.mkdir(exist_ok=True)
            opt = SimpleNamespace(**vars(self.args))
            opt.proxies, opt.name = original["proxies"], name
            train_readout(opt, self.d, SimpleNamespace(train_models=self.ref), self.manifest, directory)
        readout = joblib.load(directory / "readout.joblib") if self.refit else original
        features = features_for(
            self.args, self.d, self.query, readout["pack"], readout["proxies"], self.phase
        )
        p, mem = readout["readout"].predict(self.d.embeddings[self.query], features)
        with np.load(directory / "predictions.npz") as f:
            oof = f["historical"]
        result = dict(
            readout=readout, p=p, mem=mem, dom=mem.argmax(1), oof=oof, directory=directory, features=features
        )
        self.heads[name] = result
        return result

    def measurements(self, cfg, head, rows, phase):
        args = SimpleNamespace(**vars(self.args))
        args.axes = cfg.get("axes")
        return skill_measurements(args, self.d, rows, head["readout"], phase)

    def frozen_design(self, cfg, head):
        from train_joint_pool import frozen_vectors, joint_coordinates

        name = cfg.get("coordinates", cfg["name"])
        if name not in self.coordinates:
            base = self.args.out / name
            with np.load(base / "coordinate_pack.npz") as f:
                pack = dict(f)
            with np.load(base / "development_features.npz") as f:
                design = f["x"]
            if self.phase == "development" and np.array_equal(self.query, self.manifest["development"]):
                target = design[len(self.train) :]
            else:
                raw = self.measurements(cfg, head, self.query, self.phase)
                frozen = frozen_vectors(self.args, self.split, self.query)
                target = joint_coordinates(raw, frozen, pack)
            self.coordinates[name] = design[: len(self.train)], target
        return self.coordinates[name]

    def component(self, cfg):
        name = cfg["name"]
        kind = cfg.get("algorithm", "regional")
        key = (name, kind, cfg.get("n_regions"), cfg.get("classes"), cfg.get("steps"))
        if key in self.components:
            return self.components[key]
        head = self.head(cfg.get("readout", name))
        block = dict(head=head, kind=kind)
        if kind in ("regional", "local"):
            n = cfg["n_regions"]
            folder = head["directory"]
            file = folder / f"regions_{n}.joblib"
            if not file.exists():
                (g,), gd, models = make_regions(head["oof"], self.dt, n)
                means = np.stack([self.h[:, g == j].mean(1) for j in range(len(gd))])
                joblib.dump(
                    dict(models=models, group_domain=gd, groups=g, means=means, pop=np.bincount(g)),
                    file,
                    compress=2,
                )
            region = joblib.load(file)
            block.update(region=region, query_group=assign_regions(head["p"], head["dom"], region["models"]))
        elif kind == "skills":
            base = self.args.out / name
            directory = base / "refit75" if self.refit else base
            directory.mkdir(exist_ok=True)
            fpack, fdata = directory / "skills.npz", directory / "training_coordinates.npz"
            if not fpack.exists():
                from coldstart.skill_embedding import skill_strata

                inp = self.measurements(cfg, head, self.train, "development")
                member = np.eye(int(self.dt.max()) + 1)[self.dt]
                bank = fit_skill_space(inp, head["oof"], member, self.h)
                x, hr = skill_coordinates(bank, inp, head["oof"], member)
                means = np.stack([self.h[:, self.dt == j].mean(1) for j in range(member.shape[1])])
                np.savez_compressed(fpack, **bank, means=means, cells=skill_strata(x, self.h, self.dt, 22))
                np.savez_compressed(fdata, x=x, historical=blend_relative(head["oof"], hr))
            with np.load(fpack) as f:
                bank = dict(f)
            with np.load(fdata) as f:
                x, hist = f["x"], f["historical"]
            raw = self.measurements(cfg, head, self.query, self.phase)
            member = np.eye(int(self.dt.max()) + 1)[head["dom"]]
            xx, hr = skill_coordinates(bank, raw, head["p"], member)
            block.update(
                x=x, hist=hist, bank=bank, xx=xx, member=member, predicted=blend_relative(head["p"], hr)
            )
        elif kind in ("paired", "rescue"):
            dest = self.args.out / name
            if kind == "rescue":
                from catboost import CatBoostClassifier

                prefix = "refit" if self.refit else "development"
                ps, px = dest / f"{prefix}_space.npz", dest / f"{prefix}_cohort.npz"
                block["model"] = CatBoostClassifier().load_model(str(dest / f"{prefix}.cbm"))
            else:
                part = dest / ("refit" if self.refit else "dev")
                ps, px = part / "pair_space.npz", part / "pair_controls.npz"
            with np.load(ps) as f:
                space = dict(f)
            with np.load(px) as f:
                x = f["design" if kind == "rescue" else "x"]
                block["cells"] = f["cells"]
            raw = self.measurements(cfg, head, self.query, self.phase)
            block.update(
                ctx=cohort_context(self.h, self.dt, x),
                xx=lookup_pair_space(raw, head["dom"], space),
                space=space,
            )
        elif kind == "tabular":
            from coldstart.tabular_capability import query_skills

            paired = self.component(
                dict(name=cfg["pairs"], algorithm="paired", readout=cfg["readout"], axes=cfg.get("axes"))
            )
            file = "refit_targets.npz" if self.refit else "targets.npz"
            with np.load(self.args.out / name / file) as f:
                pack = dict(f)
            design = query_skills(paired["xx"], head["dom"], pack)
            block.update(pack=pack, xx=design)
        elif kind == "joint":
            x, xx = self.frozen_design(cfg, head)
            block.update(x=x, xx=xx)
        elif kind == "cascade":
            from coldstart.cascade import approximate

            sub = "refit" if self.refit else "dev"
            student = joblib.load(self.args.out / name / f"{sub}.joblib")
            with np.load(self.args.out / cfg["teacher"] / sub / f"types_{int(cfg['classes'])}.npz") as f:
                pack = dict(f)
            peer, prob = approximate(
                self.d.embeddings[self.query], head["features"], head["p"], head["dom"], student
            )
            block.update(pack=pack, student=student, peer=peer, posterior=prob)
        elif kind == "types":
            import torch
            from coldstart.latent_skills import ItemTypeGate, gate_scores

            x, xx = self.frozen_design(cfg, head)
            base = self.args.out / name / ("refit" if self.refit else "dev")
            k, step = int(cfg["classes"]), int(cfg["steps"])
            with np.load(base / f"types_{k}.npz") as f:
                pack = dict(f)
            gate = ItemTypeGate(x.shape[1], len(pack["means"]), k)
            gate.load_state_dict(
                torch.load(base / f"gate_{k}_{step}.pt", map_location="cpu", weights_only=True)
            )
            block.update(pack=pack, posterior=gate_scores(gate.eval(), xx, head["dom"]))
        elif kind == "moments":
            from train_response_moments import extra_features

            _, xx = self.frozen_design(cfg, head)
            proxies = (cfg.get("proxies") or "").split()
            extra = extra_features(self.args, self.d.prompt_ids[self.query], proxies, self.phase)
            if extra is not None:
                xx = np.column_stack((xx, extra))
            file = "refit.joblib" if self.refit else "development.joblib"
            model = joblib.load(self.args.out / name / file)
            block.update(model=model, coordinates=model.predict(xx, head["dom"]))
        elif kind == "contrast":
            from train_contrastive_router import load_decoder
            from train_response_moments import extra_features
            from coldstart.contrastive_routing import features

            x, xx = self.frozen_design(cfg, head)
            proxies = (cfg.get("proxies") or "").split()
            if proxies:
                x = np.column_stack(
                    (x, extra_features(self.args, self.d.prompt_ids[self.train], proxies, "development"))
                )
                xx = np.column_stack(
                    (xx, extra_features(self.args, self.d.prompt_ids[self.query], proxies, self.phase))
                )
            prefix = "refit" if self.refit else "development"
            net = load_decoder(self.args.out / name / f"{prefix}_{int(cfg['steps'])}.pt")
            block.update(net=net, x=features(net, x, self.dt), xx=features(net, xx, head["dom"]))
        elif kind == "forest":
            prefix = "refit" if self.refit else "development"
            model = joblib.load(self.args.out / name / f"{prefix}.joblib")
            with np.load(self.args.out / name / f"{prefix}_leaves.npz") as f:
                train_leaves = f["leaves"]
            raw = self.measurements(cfg, head, self.query, self.phase)
            pred, leaves = model.predict(raw, head["dom"])
            block.update(model=model, pred=pred, leaves=leaves, training_leaves=train_leaves)
        elif kind == "answers":
            import torch
            from train_conditional_router import proxy_data
            from coldstart.latent_answers import AnswerDecoder, answer_arrays

            filename = "refit.pt" if self.refit else "development.pt"
            state = torch.load(self.args.out / name / filename, map_location="cpu", weights_only=True)
            net = AnswerDecoder(state["experts"], state["domains"])
            net.load_state_dict(state["state"])
            proxies = cfg["proxies"].split()
            training = answer_arrays(
                proxy_data(self.args.out, self.d.prompt_ids[self.train], proxies, "development")
            )
            if cfg.get("answer_keys"):
                from train_latent_answers import historical_keys

                training["option_key"] = historical_keys(self.args.out, self.d.prompt_ids[self.train])
            query = answer_arrays(
                proxy_data(self.args.out, self.d.prompt_ids[self.query], proxies, self.phase)
            )
            block.update(net=net.eval(), hz=state["historical"], training=training, query=query)
        elif kind == "distilled":
            from coldstart.distilled_text import historical_embeddings

            step = int(cfg["steps"])
            with np.load(self.args.out / name / f"development_{step}.npz") as f:
                full = f["features"]
            x = full[: len(self.train)]
            if self.phase == "development":
                lookup = dict(zip(self.d.prompt_ids[self.manifest["development"]], range(len(full) - len(x))))
                xx = full[len(x) :][[lookup[v] for v in self.d.prompt_ids[self.query]]]
            else:
                with np.load(self.args.out / name / f"evaluation_{step}.npz") as f:
                    lookup = {int(v): i for i, v in enumerate(f["ids"])}
                    xx = f["features"][[lookup[int(v)] for v in self.d.prompt_ids[self.query]]]
            block.update(x=x, xx=xx, student=historical_embeddings(x, self.dt, self.h))
        elif kind == "coregionalized":
            from coldstart.coregionalized import query

            prefix = "refit" if self.refit else "development"
            model = joblib.load(self.args.out / name / f"{prefix}.joblib")
            raw = self.measurements(cfg, head, self.query, self.phase)
            pred, cross = query(model, self.d.embeddings[self.query], raw, head["dom"])
            block.update(model=model, pred=pred, cross=cross)
        elif kind == "reference":
            from coldstart.reference_outcomes import features, probability, state_effects
            from train_conditional_router import proxy_data

            sub = "refit" if self.refit else "dev"
            verifier = joblib.load(self.args.out / name / f"{sub}.joblib")
            ev = proxy_data(self.args.out, self.d.prompt_ids[self.query], cfg["proxies"].split(), self.phase)
            design = features(ev, head["p"], head["dom"], len(verifier["means"]))
            block.update(verifier=verifier, chance=probability(verifier, design))
            if "state_readouts" in verifier:
                block["state_effects"] = state_effects(verifier["state_readouts"], design)
        elif kind == "nominal":
            from coldstart.nominal_response import query
            from coldstart.latent_answers import answer_arrays
            from train_conditional_router import proxy_data
            from train_nominal_router import load_net

            dest = self.args.out / name
            sub = "refit" if self.refit else "dev"
            net = load_net(dest / f"{sub}.pt")
            proxies = cfg["proxies"].split()
            data = answer_arrays(
                proxy_data(self.args.out, self.d.prompt_ids[self.train], proxies, "development")
            )
            test = answer_arrays(
                proxy_data(self.args.out, self.d.prompt_ids[self.query], proxies, self.phase)
            )
            with np.load(dest / "keys.npz") as f:
                if not np.array_equal(f["prompt_ids"], self.d.prompt_ids[self.train]):
                    raise ValueError("Historical answer keys belong to a different training set.")
                keys = f["keys"]
            with np.load(dest / f"{sub}.npz") as f:
                history, means = f["history"], f["means"]
            block.update(
                net=net,
                history=history,
                means=means,
                keys=keys,
                training=query(net, data, self.dt),
                query=query(net, test, head["dom"]),
            )
        elif kind == "proxy-confusion":
            from coldstart.proxy_confusion import coordinates, judge_probs
            from coldstart.latent_answers import answer_arrays
            from train_conditional_router import proxy_data

            judge = joblib.load(self.args.out / name / "question.joblib")
            with np.load(self.args.out / name / "training.npz") as f:
                x, xp = f["x"], f["xp"]
            prefix = "refit" if self.refit else "development"
            with np.load(self.args.out / name / f"{prefix}.npz") as f:
                ref = dict(f)
            raw = answer_arrays(
                proxy_data(self.args.out, self.d.prompt_ids[self.query], cfg["proxies"].split(), self.phase)
            )
            xx = coordinates(
                judge, raw, judge_probs(judge["judge"], raw, head["dom"], judge["domains"]), head["dom"]
            )
            block.update(judge=judge, x=x, xp=xp, xx=xx, ref=ref)
        self.components[key] = block
        return block

    def panel(self, cfg, labels, seed, balanced):
        from coldstart.conditional_capability import choose_panel
        from coldstart.skill_embedding import stratified_cells

        n = int(cfg["probe_count"])
        if cfg["policy"] == "balanced":
            return balanced[:n]
        if cfg["policy"] == "stop-evpi":
            from panel_stopping import select_prefix

            head = self.head(cfg.get("readout", cfg["name"]))
            return select_prefix(self.h, labels, balanced, self.dt, head["oof"], seed, n, cfg["tolerance"])[0]
        if cfg["policy"] == "opportunity":
            from coldstart.switch_calibration import tail_panel

            head = self.head(cfg["readout"])
            return tail_panel(self.h, labels, balanced, self.dt, head["oof"], seed, n)
        part = self.component(cfg)
        if cfg["policy"] == "skill-stratified":
            cells = part["bank"]["cells"] if part["kind"] == "skills" else part["cells"]
            return stratified_cells(cells, self.dt, seed, n)
        reg = part["region"]
        return choose_panel(
            self.h, labels, reg["groups"], reg["group_domain"], reg["means"], cfg["policy"], seed, n
        )

    def score(self, cfg, labels, idx, seed):
        block = self.component(cfg)
        head, kind = block["head"], block["kind"]
        h, y = self.h, labels
        if kind in ("regional", "local"):
            reg = block["region"]
            g, gd, pop, means = reg["groups"], reg["group_domain"], reg["pop"], reg["means"]
            group = block["query_group"]
            if kind == "regional":
                z = fit_profile(h[:, idx], y[:, idx], g[idx], pop, gd, cfg["prior"])
                mat = score_vectors(z, head["p"], group, means, gd, pop, cfg)
                return mat, dict(z=z)
            z = local_fingerprints(h[:, idx], y[:, idx], g[idx], means, pop, gd, cfg["focus"], cfg["prior"])
            mat = score_local(z, head["p"], group, means, pop, gd, cfg)
            cal = {}
            if cfg["calibration"]:
                cal = calibrate_switches(h, y, idx, head["oof"], g, means, pop, gd, z, cfg, seed)
                mat = calibrated_scores(mat, cal, cfg["calibration"])
            return mat, dict(z=z, calibration=cal)
        if kind == "skills":
            cells = block["bank"]["cells"] if cfg["policy"] == "skill-stratified" else self.dt
            z = fit_skill_profile(
                h,
                y,
                idx,
                self.dt,
                block["x"],
                block["hist"],
                cells,
                cfg["strength"],
                cfg["observed"],
                cfg["rank_shrink"],
            )
            return skill_scores(
                z, block["xx"], block["predicted"], block["member"], block["bank"]["means"], cfg
            ), dict(z=z)
        if kind == "switch":
            from coldstart.switch_calibration import fit_switches, switch_scores

            pf = fit_switches(h, y, idx, self.dt, head["oof"], cfg["nearest"], cfg["prior"])
            return switch_scores(pf, head["p"], head["dom"], cfg), pf
        if kind == "survey":
            from coldstart.survey_policy import fit_policy, route_policy

            pf = fit_policy(h, y, idx, self.dt, head["oof"], cfg["nearest"], cfg["ridge"], seed)
            return route_policy(pf, head["p"], head["dom"], cfg), pf
        if kind == "distilled":
            from coldstart.distilled_text import distilled_scores, fit_distilled

            pf = fit_distilled(block["student"], block["x"], self.dt, h, y, idx, cfg["nearest"], cfg["ridge"])
            return distilled_scores(pf, block["xx"], head["dom"], block["student"], cfg["signal"]), pf
        if kind == "coregionalized":
            from coldstart.coregionalized import embed, panel_cross, scores
            from train_coregionalized import model_metadata

            meta = model_metadata(
                self.args, np.r_[np.asarray(self.d.model_names)[self.ref], self.target_names]
            )
            pf = embed(block["model"], h, y, idx, self.dt, meta, cfg["metadata"], seed)
            cross = panel_cross(block["cross"], self.dt, idx, len(self.query))
            return scores(pf, block["pred"], cross, head["dom"], cfg), pf
        if kind == "reference":
            from coldstart.reference_outcomes import embed, scores, state_embedding, state_scores

            verifier = block["verifier"]
            if cfg.get("conditional_states"):
                z = state_embedding(verifier, h, y, idx, self.dt, cfg["nearest"], cfg["shrink"])
                return state_scores(
                    verifier,
                    z,
                    block["chance"],
                    head["p"],
                    head["dom"],
                    block["state_effects"],
                    cfg["signal"],
                    cfg["calibration"],
                ), dict(z=z)
            z = embed(verifier, h, y, idx, self.dt, cfg["nearest"], cfg["shrink"])
            fitted = {}
            if cfg.get("panel_calibration"):
                from coldstart.reference_outcomes import calibrate

                sub = "refit" if self.refit else "dev"
                with np.load(self.args.out / cfg["name"] / f"{sub}_oof.npz") as f:
                    fitted = calibrate(verifier, h, y, idx, self.dt, head["oof"], f["chance"], cfg, seed)
            return scores(verifier, z, block["chance"], head["p"], head["dom"], dict(cfg, **fitted)), dict(
                z=z, calibration=fitted
            )
        if kind == "nominal":
            from coldstart.nominal_response import fit_new, outputs, scores

            dom = head["dom"]
            pf = fit_new(
                block["net"],
                block["history"],
                block["means"],
                block["training"],
                self.dt,
                block["keys"],
                h,
                y,
                idx,
                cfg["nearest"],
                cfg["prior"],
            )
            prob = outputs(block["net"], pf["z"], block["query"], dom)
            return scores(pf, prob, head["p"], dom, block["means"], cfg), pf
        if kind == "proxy-confusion":
            from coldstart.proxy_confusion import embed, score

            ref = block["ref"]
            pf = embed(ref, h, y, self.dt, idx, block["x"], block["xp"], cfg["nearest"], cfg["ridge"], seed)
            return score(pf, block["xx"], head["p"], head["dom"], ref["means"], cfg), pf
        if kind == "rescue":
            ctx = block["ctx"]
            pf, teacher = measured_pool(ctx, y, idx)
            pair = pair_profile(ctx, pf, teacher, y, idx, pf["aggregate"].argmax())
            r = rescue_scores(
                ctx, pf, pair, block["xx"], head["dom"], block["model"], cfg["alpha"], cfg["trees"]
            )
            return r, dict(pf=pf, pair=pair)
        if kind == "paired":
            ctx = block["ctx"]
            be = behavioral_design(h, self.dt, ctx["behavior"])
            teacher = infer_teacher(h, y, idx, self.dt, be, ctx["behavior"], cfg["teacher_prior"])
            strata = self.dt if cfg["policy"] == "balanced" else block["cells"]
            wt = np.bincount(strata)[strata[idx]] / np.bincount(strata[idx])[strata[idx]]
            z = distill(
                ctx["design"],
                teacher,
                y,
                idx,
                self.dt,
                wt / wt.sum(),
                cfg["pseudo_count"],
                cfg["probe_weight"],
                cfg["strength"],
            )
            return pair_scores(z, block["xx"], cfg["temperature"]), dict(z=z)
        if kind == "tabular":
            from coldstart.tabular_capability import fit_tabular

            prior, fit = fit_tabular(h, y, idx, self.dt, block["pack"])
            z = (1 - cfg["tabular_mix"]) * prior + cfg["tabular_mix"] * fit
            z[:, len(block["pack"]["center"]) :] *= cfg["signal"]
            return z @ block["xx"].T, dict(z=z)
        if kind == "cascade":
            from coldstart.cascade import decision, teacher_config
            from coldstart.latent_skills import measure_types, route_types

            pack = block["pack"]
            z = measure_types(
                h, y, idx, self.dt, pack["rates"], pack["posterior"], cfg["shrink"], cfg["neighbor"]
            )
            mat, risk = decision(
                z, block["peer"], block["posterior"], head["dom"], pack["means"], block["student"], cfg
            )
            defer = risk > cfg["threshold"]
            if defer.any():
                large = self.component(teacher_config(cfg))
                p, dom = large["head"]["p"][defer], large["head"]["dom"][defer]
                mat[:, defer] = route_types(z, p, large["posterior"][defer], dom, pack["means"], cfg)
            self.last_deferred = defer
            return mat, dict(z=z)
        if kind == "types":
            from coldstart.latent_skills import measure_types, route_types

            pack = block["pack"]
            z = measure_types(
                h, y, idx, self.dt, pack["rates"], pack["posterior"], cfg["shrink"], cfg["neighbor"]
            )
            return route_types(z, head["p"], block["posterior"], head["dom"], pack["means"], cfg), dict(z=z)
        if kind == "moments":
            from coldstart.paired_moments import measure, score_moments

            z = measure(h, y, idx, self.dt, block["model"], cfg["pseudo"], cfg["sharpness"])
            return score_moments(z, block["coordinates"], head["dom"], cfg["signal"]), dict(z=z)
        if kind == "contrast":
            from coldstart.contrastive_routing import contrast_scores, embed_challengers

            pf = embed_challengers(block["net"], block["x"], h, y, idx, self.dt, cfg["prior"], cfg["nearest"])
            return contrast_scores(pf, block["xx"], head["dom"], cfg), pf
        if kind == "forest":
            from coldstart.policy_forest import fit_forest_profile, forest_scores, leaf_similarity

            leaves = block["training_leaves"]
            z = fit_forest_profile(h, y, idx, self.dt, leaves, cfg["penalty"], cfg["nearest"])
            out = forest_scores(
                z,
                block["model"].means,
                block["pred"],
                head["dom"],
                leaf_similarity(block["leaves"], leaves[idx]),
                cfg,
            )
            return out, dict(z=z, support=leaves[idx])
        if kind == "answers":
            from train_latent_answers import assembled_scores, fit_answers
            from coldstart.latent_answers import predict_answers

            pf = fit_answers(
                block["net"], block["hz"], block["training"], self.dt, h, y, idx, cfg["prior"], cfg["nearest"]
            )
            v = predict_answers(block["net"], pf["z"], block["query"], head["dom"])
            return assembled_scores(pf, v, head["dom"], head["p"], cfg), pf
        if kind == "joint":
            import torch
            from train_joint_pool import fit_on_panel

            one = np.eye(int(self.dt.max()) + 1, dtype=np.float32)
            network = fit_on_panel(
                block["x"],
                one[self.dt],
                h,
                y,
                idx,
                seed,
                (int(cfg["steps"]),),
                lambda *_: None,
                cfg["real_weight"],
            )
            network = network.cpu().eval()
            with torch.inference_mode():
                _, logits = network(torch.from_numpy(block["xx"]), torch.from_numpy(one[head["dom"]]))
                z = network.vectors().numpy().copy()
            return logits.sigmoid().numpy().T, dict(z=z, encoder=network.encoder.state_dict())
        raise ValueError(f"Unsupported confirmation model: {kind}")
