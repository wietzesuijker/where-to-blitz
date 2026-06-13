"""BTG species-discovery experiment — offline GPU version.

Modifications vs exp_discovery_acquisition.py:
- Accepts --image-cache dir: if set, load images from disk instead of fetching URLs.
- Staging mode (--stage-only): pull observations + download all photos to --image-cache,
  save obs JSON, then exit. Run this on the login node.
- GPU mode (--obs-cache <json>): load saved obs JSON + images from cache, embed on GPU.
"""
from __future__ import annotations
import argparse, io, json, os, time, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np

INAT = "https://api.inaturalist.org/v1/observations"
PROJECT = 228908


def pull_observations(n_target=1500, per_page=200):
    params = {"project_id": PROJECT, "quality_grade": "research", "iconic_taxa": "Amphibia",
              "per_page": per_page, "order_by": "id", "order": "desc",
              "photos": "true", "identified": "true",
              "swlat": 41, "swlng": -141, "nelat": 84, "nelng": -52}
    import requests
    out, id_below = [], None
    while len(out) < n_target:
        p = dict(params)
        if id_below:
            p["id_below"] = id_below
        d = requests.get(INAT, params=p, timeout=60).json()
        res = d.get("results", [])
        if not res:
            break
        for o in res:
            taxon = o.get("taxon") or {}
            photos = o.get("photos") or []
            g = o.get("geojson") or {}
            if taxon.get("rank") == "species" and photos and g.get("coordinates"):
                url = photos[0].get("url", "").replace("/square.", "/small.")
                if url:
                    out.append({"id": o["id"], "species": taxon["name"],
                                "lat": g["coordinates"][1], "lon": g["coordinates"][0],
                                "photo": url})
        id_below = res[-1]["id"]
        if len(res) < per_page:
            break
    return out[:n_target]


def stage_images(records, cache_dir: Path):
    """Download all photos to cache_dir. Returns records with 'local_path' set."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i, rec in enumerate(records):
        local = cache_dir / f"{rec['id']}.jpg"
        rec["local_path"] = str(local)
        if local.exists():
            ok += 1
            continue
        try:
            with urllib.request.urlopen(rec["photo"], timeout=30) as r:
                data = r.read()
            local.write_bytes(data)
            ok += 1
        except Exception as e:
            print(f"  [stage] SKIP {rec['id']}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  staged {i+1}/{len(records)} ({ok} ok)")
    print(f"[stage] done: {ok}/{len(records)} images cached to {cache_dir}")
    return records


def load_image_from_disk(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def fetch_image_from_url(url: str):
    from PIL import Image
    with urllib.request.urlopen(url, timeout=30) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def embed_images(records, device, batch=64, use_cache=True):
    """Vision embeddings on GPU. Loads from local_path if available, else fetches URL."""
    import torch
    from PIL import Image
    backbone = None
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        backbone = "dinov2_vits14"
        import torchvision.transforms as T
        tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        def encode(batch_t):
            return model(batch_t)
    except Exception as e:
        print(f"[embed] DINOv2 unavailable ({e}); falling back to torchvision ResNet50")
        import torchvision
        from torchvision.models import ResNet50_Weights
        w = ResNet50_Weights.IMAGENET1K_V2
        m = torchvision.models.resnet50(weights=w)
        m.fc = torch.nn.Identity()
        model = m.to(device).eval()
        backbone = "resnet50_imagenet"
        tf = w.transforms()
        def encode(batch_t):
            return model(batch_t)

    embs, keep = [], []
    buf, idxs = [], []

    def flush():
        if not buf:
            return
        import torch
        with torch.no_grad():
            x = torch.stack(buf).to(device)
            z = encode(x).float().cpu().numpy()
        embs.append(z); keep.extend(idxs)
        buf.clear(); idxs.clear()

    for i, rec in enumerate(records):
        try:
            local = rec.get("local_path", "")
            if use_cache and local and os.path.exists(local):
                img = load_image_from_disk(local)
            else:
                img = fetch_image_from_url(rec["photo"])
            buf.append(tf(img)); idxs.append(i)
        except Exception as exc:
            print(f"  [embed] skip {i}: {exc}")
            continue
        if len(buf) >= batch:
            flush()
        if (i + 1) % 200 == 0:
            print(f"  embedded {i+1}/{len(records)}...")
    flush()
    E = np.concatenate(embs, 0) if embs else np.zeros((0, 1))
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    return E, keep, backbone


def discover(order_fn, recs, E, seed, budget):
    rng = np.random.default_rng(seed)
    n = len(recs)
    start = int(rng.integers(n))
    seen = [start]
    seen_species = {recs[start]["species"]}
    curve = [1]
    remaining = set(range(n)) - {start}
    while remaining and len(seen) < budget:
        nxt = order_fn(seen, remaining, recs, E, rng)
        remaining.discard(nxt)
        seen.append(nxt)
        seen_species.add(recs[nxt]["species"])
        curve.append(len(seen_species))
    return curve


def acq_random(seen, remaining, recs, E, rng):
    return int(rng.choice(list(remaining)))

def acq_embed_novelty(seen, remaining, recs, E, rng):
    rem = np.fromiter(remaining, int)
    sims = E[rem] @ E[seen].T
    nearest = sims.max(axis=1)
    return int(rem[int(np.argmin(nearest))])

def acq_spatial_coverage(seen, remaining, recs, E, rng):
    rem = np.fromiter(remaining, int)
    sl = np.array([[recs[i]["lat"], recs[i]["lon"]] for i in seen])
    rl = np.array([[recs[i]["lat"], recs[i]["lon"]] for i in rem])
    d = np.sqrt(((rl[:, None, :] - sl[None, :, :]) ** 2).sum(-1))
    return int(rem[int(np.argmax(d.min(axis=1)))])


def auc(curve, budget):
    c = np.array(curve, float)
    if len(c) < budget:
        c = np.concatenate([c, np.full(budget - len(c), c[-1])])
    return float(c[:budget].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=".")
    ap.add_argument("--image-cache", default=None,
                    help="Dir to read/write cached images. Set for offline mode.")
    ap.add_argument("--obs-cache", default=None,
                    help="Path to pre-saved observations JSON (from --stage-only run).")
    ap.add_argument("--stage-only", action="store_true",
                    help="Download obs+photos to --image-cache, save JSON, then exit.")
    args = ap.parse_args()

    cache_dir = Path(args.image_cache) if args.image_cache else None
    obs_json = Path(args.obs_cache) if args.obs_cache else None

    # STAGE MODE: run on login node
    if args.stage_only:
        print("[stage] pulling observations from iNaturalist...")
        recs = pull_observations(args.n)
        print(f"[stage] got {len(recs)} obs ({len(set(r['species'] for r in recs))} species)")
        assert cache_dir, "--image-cache required for --stage-only"
        recs = stage_images(recs, cache_dir)
        out_json = cache_dir / "observations.json"
        with open(out_json, "w") as f:
            json.dump(recs, f)
        print(f"[stage] saved {out_json}")
        return

    # GPU MODE
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"device={device} cuda={torch.cuda.is_available()} name={gpu_name}")

    if obs_json and obs_json.exists():
        print(f"[run] loading obs from {obs_json}")
        with open(obs_json) as f:
            recs = json.load(f)
    else:
        print("[run] no obs cache, pulling from iNat (need internet)...")
        recs = pull_observations(args.n)

    print(f"pulled {len(recs)} amphibian obs w/ photo+species, "
          f"{len(set(r['species'] for r in recs))} distinct species")

    use_cache = cache_dir is not None
    E, keep, backbone = embed_images(recs, device, use_cache=use_cache)
    recs = [recs[i] for i in keep]
    print(f"embedded {len(recs)} images with {backbone}, dim={E.shape[1]} in {time.time()-t0:.0f}s")
    n_species = len(set(r["species"] for r in recs))

    strategies = {"random": acq_random, "spatial_coverage": acq_spatial_coverage,
                  "embedding_novelty": acq_embed_novelty}
    budget = min(args.budget, len(recs))
    results = defaultdict(list)
    curves_mean = {}
    for name, fn in strategies.items():
        aucs, last = [], []
        all_curves = []
        for s in range(args.seeds):
            c = discover(fn, recs, E, seed=s, budget=budget)
            aucs.append(auc(c, budget)); last.append(c[-1])
            padded = np.array(c + [c[-1]] * (budget - len(c)))[:budget]
            all_curves.append(padded)
        results[name] = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                         "species_at_budget_mean": float(np.mean(last)),
                         "species_at_budget_std": float(np.std(last))}
        curves_mean[name] = np.mean(all_curves, 0).tolist()
        print(f"  {name:18s} AUC={results[name]['auc_mean']:.1f}±{results[name]['auc_std']:.1f} "
              f"species@{budget}={results[name]['species_at_budget_mean']:.1f}")

    base = results["spatial_coverage"]["species_at_budget_mean"]
    fancy = results["embedding_novelty"]["species_at_budget_mean"]
    rng_r = results["random"]["species_at_budget_mean"]
    verdict = ("embedding_novelty BEATS coverage" if fancy > base + 1 else
               "embedding_novelty does NOT beat coverage" if fancy < base - 1 else
               "embedding_novelty TIES coverage")
    print(f"\nVERDICT @budget={budget}: {verdict} "
          f"(novelty {fancy:.1f} vs coverage {base:.1f} vs random {rng_r:.1f}; total species {n_species})")

    out_data = {"meta": {"n_obs": len(recs), "n_species": n_species, "budget": budget,
                         "seeds": args.seeds, "backbone": backbone, "device": device,
                         "gpu_name": gpu_name,
                         "runtime_s": round(time.time() - t0, 1)},
                "results": dict(results), "curves_mean": curves_mean, "verdict": verdict}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "exp_discovery_results.json"), "w") as f:
        json.dump(out_data, f, indent=2)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        for name, c in curves_mean.items():
            plt.plot(range(1, len(c) + 1), c, label=name, lw=2)
        plt.xlabel("observations sampled"); plt.ylabel("distinct species discovered")
        plt.title(f"BTG amphibian discovery curves ({backbone}, n={len(recs)}, {args.seeds} seeds)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out, "exp_discovery_curves.png"), dpi=120)
        print("wrote exp_discovery_curves.png")
    except Exception as e:
        print(f"[plot] skipped: {e}")
    print(f"DONE in {time.time()-t0:.0f}s -> exp_discovery_results.json")


if __name__ == "__main__":
    main()
