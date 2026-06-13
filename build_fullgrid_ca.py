"""Canada-wide engine: build per-group webapp_data over EVERY Canadian land cell.

Mirror of build_fullgrid.py (the BC engine), extended from BC to all of Canada.
The structural difference is the *source of the per-cell weight*: the BC engine
has per-taxon iNaturalist record CSVs over a small grid; nationally we instead
read the published per-taxon iNaturalist density COGs (1 km, Canada LAEA) and
reproject them to the 0.25-deg WGS84 grid.

Per cell, per group:
  - discover     = norm(1/(density+eps))         REAL  (inverse iNat density)
  - env          = climate surprisal weighted by density   REAL (CHELSA + density)
  - staleness    = norm(1/(density+eps))         APPROX (proxy: no per-record dates)
  - conservation = 0                             APPROX (no national rarity/IUCN join)
  - urgency      = forest-loss fraction          REAL if ca_forestloss.tif present,
                                                 else 0 and flagged DEFERRED
  - travel_min   = mean Weiss travel-time over the cell     REAL
  - n_train      = round(density * cell_km2)     REAL-proxy (records implied by density)

Land vs ocean is decided by the Weiss travel-time raster (ocean = nodata -9999),
using the same 7x7 sub-grid land mask as the BC engine.

Row format EXACTLY matches BC: [lat,lon,discover,conservation,env,staleness,urgency,travel_min,n_train].
"""
import json, os
import numpy as np
import rasterio
from rasterio.windows import from_bounds

RES = 0.25
BBOX = (-141.0, 41.0, -52.0, 84.0)               # Canada: minlon,minlat,maxlon,maxlat
TRAVEL = "cluster_results/ca_travel_time.tif"    # Weiss 2018 (minutes to nearest city), Canada clip
CLIM = "cluster_results/ca_bioclim.tif"          # 3 bands: temp, seasonality, precip (CHELSA)
LOSS = "cluster_results/ca_forestloss.tif"       # optional per-0.05-deg Hansen loss fraction
DENS_DIR = "cluster_results/ca"                  # ca_density_<TAXA>.tif (0.25-deg WGS84, this grid)
OUT_DIR = "cluster_results/ca"
H = 1.0                                           # climate KDE bandwidth (standardized units)
EPS = 1e-3                                        # density floor for inverse-density discover

# Our 8 output groups -> COG taxon. Fungi/Reptilia have no national COG -> fall back to 'All'.
GROUP_TO_COG = {
    "Amphibia": "Amphibia",
    "Aves": "Aves",
    "Insecta": "Insecta",
    "Mammalia": "Mammalia",
    "Plantae": "Plantae",
    "Reptilia": "All",          # no Reptilia COG -> All-biodiversity density as proxy
    "Fungi": "All",             # no Fungi COG     -> All-biodiversity density as proxy
    "All biodiversity": "All",
}
GROUPS = ["Amphibia", "Aves", "Fungi", "Insecta", "Mammalia", "Plantae", "Reptilia", "All biodiversity"]


def norm(x):
    x = np.asarray(x, float)
    r = np.nanmax(x) - np.nanmin(x)
    return (x - np.nanmin(x)) / r if r else x * 0.0


def sample(path, lonlat, band=1):
    with rasterio.open(path) as ds:
        return np.array([v[band - 1] for v in ds.sample(lonlat)], float)


def read_density(group, lonlat):
    """Per-cell mean iNat density for a group's COG taxon, sampled at cell centroids
    from the already-reprojected (LAEA->WGS84, 0.25-deg average-resampled) raster."""
    taxon = GROUP_TO_COG[group]
    path = os.path.join(DENS_DIR, f"ca_density_{taxon}.tif")
    with rasterio.open(path) as ds:
        d = np.array([v[0] for v in ds.sample(lonlat)], float)
        nd = ds.nodata
    if nd is not None:
        d[d == nd] = np.nan
    d[d < 0] = np.nan
    return d


# 1) enumerate every cell centroid over the bbox, keep the land ones (Weiss != nodata).
gis = range(int(np.floor(BBOX[1] / RES)), int(np.floor(BBOX[3] / RES)) + 1)
gjs = range(int(np.floor(BBOX[0] / RES)), int(np.floor(BBOX[2] / RES)) + 1)
cells = [(gi, gj, (gi + 0.5) * RES, (gj + 0.5) * RES) for gi in gis for gj in gjs]
clat = np.array([c[2] for c in cells]); clon = np.array([c[3] for c in cells])

# Same robust land test as BC: 7x7 sub-grid, keep cell if any sub-point is land,
# travel cost = mean of the land sub-points.
SUB = list(np.linspace(-0.11, 0.11, 7))
n = len(SUB)
subpts = [(lo + dx, la + dy) for la, lo in zip(clat, clon) for dx in SUB for dy in SUB]
with rasterio.open(TRAVEL) as ds:
    vv = np.array([v[0] for v in ds.sample(subpts)], float).reshape(len(cells), n * n)
    tnd = ds.nodata
valid = (vv != tnd) & (vv >= 0) & np.isfinite(vv)
land_pts = valid.sum(1)
travel_min = np.where(valid.any(1), np.where(valid, vv, 0).sum(1) / np.maximum(land_pts, 1), np.nan)
keep = land_pts >= 1
clat, clon, travel_min = clat[keep], clon[keep], travel_min[keep]
lonlat = list(zip(clon, clat))
print(f"Canada land cells (multi-point Weiss mask): {len(clat)}")

# 2) shared climate layer for ALL land cells
with rasterio.open(CLIM) as ds:
    clim = np.array([list(ds.sample([(lon, lat)]))[0] for lon, lat in lonlat], float)
clim[clim < -1e30] = np.nan
Z = (clim - np.nanmean(clim, 0)) / (np.nanstd(clim, 0) + 1e-9)

# 3) urgency: real if a Canada Hansen loss-fraction raster exists, else deferred=0.
URGENCY_REAL = os.path.exists(LOSS)
if URGENCY_REAL:
    o_urgency = norm(sample(LOSS, lonlat))
    print("urgency: REAL (Hansen loss fraction)")
else:
    o_urgency = np.zeros(len(clat))
    print("urgency: DEFERRED (no ca_forestloss.tif) -> 0 for all cells")

# cell area in km2 (for density -> implied record count). 0.25-deg lat band, varies with lat.
KM_PER_DEG = 111.32
cell_km2 = (RES * KM_PER_DEG) * (RES * KM_PER_DEG * np.cos(np.radians(clat)))


def env_coverage(weights):
    """Climate surprisal: cells whose climate is rare among *where records exist*
    (weighted by density) score high. Percentile-ranked to spread 0..1 evenly.
    Identical math to build_fullgrid.env_coverage; here the weight is COG density.

    Vectorized + chunked: at ~38k Canada land cells the per-cell Python loop is
    O(N x R) and prohibitively slow, so the (cell x record) Gaussian kernel is
    evaluated in blocks of rows against all record-cells at once."""
    R = weights > 0
    if R.sum() == 0:
        return np.zeros(len(clat))
    ZR, wR = Z[R], weights[R]                       # (Rn,3), (Rn,)
    ok = ~np.any(np.isnan(Z), axis=1)
    s = np.full(len(clat), np.nan)
    idx = np.where(ok)[0]
    ZK = Z[idx]                                     # (Kn,3) query cells
    inv2h2 = 1.0 / (2 * H * H)
    BLK = 1024
    ZR2 = np.sum(ZR ** 2, axis=1)                   # (Rn,)
    for b in range(0, len(idx), BLK):
        Zq = ZK[b:b + BLK]                          # (m,3)
        # squared dist (m,Rn) = |q|^2 - 2 q.r + |r|^2
        d2 = (np.sum(Zq ** 2, axis=1)[:, None] - 2 * Zq @ ZR.T + ZR2[None, :])
        np.maximum(d2, 0, out=d2)
        dens = np.exp(-d2 * inv2h2) @ wR            # (m,)
        s[idx[b:b + len(Zq)]] = -np.log(dens + 1e-9)
    out = np.zeros(len(clat))
    v = ok & np.isfinite(s)
    if v.sum() > 1:
        out[np.where(v)[0]] = np.argsort(np.argsort(s[v])) / (v.sum() - 1)
    return out


def sr(x):
    return round(float(x), 3) if np.isfinite(x) else 0.0


def tr(x):
    return round(float(x), 0) if np.isfinite(x) else -1.0


# 4) per-group rows
data = {}
for group in GROUPS:
    dens = read_density(group, lonlat)            # per-cell iNat density (records/km2-ish)
    dens0 = np.where(np.isfinite(dens), dens, 0.0)
    inv = 1.0 / (dens0 + EPS)                      # under-recorded = high
    o_discover = norm(inv)                         # REAL: inverse density
    o_staleness = norm(inv)                        # APPROX: proxy (no per-record dates nationally)
    o_conservation = np.zeros(len(clat))           # APPROX: no national rarity/IUCN join -> gap default
    o_env = env_coverage(dens0)                    # REAL: climate surprisal weighted by density
    n_train = np.round(dens0 * cell_km2).astype(int)   # density -> implied record count

    rows = [[round(float(clat[i]), 3), round(float(clon[i]), 3),
             sr(o_discover[i]), sr(o_conservation[i]), sr(o_env[i]),
             sr(o_staleness[i]), sr(o_urgency[i]), tr(travel_min[i]), int(n_train[i])]
            for i in range(len(clat))]
    data[group] = rows
    rec = int((n_train > 0).sum())
    print(f"  {group:16s} (COG={GROUP_TO_COG[group]:9s}): {len(rows)} cells "
          f"({rec} with density, {len(rows)-rec} gaps)")

# 5) write per-group JSON + index
os.makedirs(OUT_DIR, exist_ok=True)
sizes = {}
for group in GROUPS:
    fn = os.path.join(OUT_DIR, f"webapp_data_{group.replace(' ', '_')}.json")
    json.dump({group: data[group]}, open(fn, "w"), separators=(",", ":"))
    sizes[group] = os.path.getsize(fn)

index = {
    "groups": GROUPS,
    "files": {g: f"webapp_data_{g.replace(' ', '_')}.json" for g in GROUPS},
    "n_cells": len(clat),
    "bbox": list(BBOX),
    "res": RES,
    "row_format": ["lat", "lon", "discover", "conservation", "env",
                   "staleness", "urgency", "travel_min", "n_train"],
    "axes_status": {
        "discover": "REAL (inverse iNat density COG)",
        "env": "REAL (CHELSA climate surprisal, density-weighted)",
        "travel_min": "REAL (Weiss 2018 travel-time)",
        "n_train": "REAL-proxy (density x cell km2)",
        "urgency": "REAL (Hansen loss fraction)" if URGENCY_REAL else "DEFERRED (0)",
        "staleness": "APPROX (inverse-density proxy, no per-record dates)",
        "conservation": "APPROX (0; no national rarity/IUCN join)",
    },
    "group_to_cog": GROUP_TO_COG,
}
json.dump(index, open(os.path.join(OUT_DIR, "index.json"), "w"), indent=2)

total = sum(sizes.values())
print(f"\nwrote {len(GROUPS)} group files + index.json to {OUT_DIR}/")
for g in GROUPS:
    print(f"  {g:16s} {sizes[g]//1024:5d} KB")
print(f"total per-group JSON: {total//1024} KB, {len(clat)} land cells each")
