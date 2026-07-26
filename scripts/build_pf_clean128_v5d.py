"""Build deterministic Clean128 v5d from the immutable Clean128 v5 artifact.

The sole scoring-behavior change is deterministic seeding of the two previously
unseeded single-PF branches. Their resampling helper is made deterministic by
moving its random draws into the explicitly seeded parent kernels.
"""

# ruff: noqa: E501

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "kaggle_notebooks" / "pf_clean128_v5" / "rogii-pf-clean128-v5.ipynb"
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "pf_clean128_v5d"
OUTPUT = OUTPUT_DIR / "rogii-pf-clean128-v5d.ipynb"


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def replace_once(value: str, old: str, new: str, label: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement, got {count}")
    return value.replace(old, new, 1)


def source_cell(cells: list[dict], source_index: int) -> dict:
    matches = [
        cell
        for cell in cells
        if cell.get("metadata", {}).get("clean128_source_index") == source_index
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one source-index {source_index} cell, got {len(matches)}")
    return matches[0]


def marked_cell(cells: list[dict], marker: str) -> dict:
    matches = [cell for cell in cells if marker in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one cell containing {marker!r}, got {len(matches)}")
    return matches[0]


SEED_HELPER_SOURCE = """# Cross-process-stable, order-independent PF seed derivation.
import hashlib as _pf_seed_hashlib

PF_DETERMINISTIC_SEEDING = True
PF_SEED_NAMESPACE = 'rogii-pf-clean128-v5d'


def _stable_pf_seed(well_id, branch):
    payload = (
        f'{PF_SEED_NAMESPACE}\\0{str(well_id)}\\0{str(branch)}'
    ).encode('utf-8')
    digest = _pf_seed_hashlib.sha256(payload).digest()
    # Numba's legacy np.random.seed is safest with a positive signed-31-bit
    # integer. SHA-256 makes this stable across processes and thread order.
    return int.from_bytes(digest[:8], 'little') % 2147483646 + 1


assert _stable_pf_seed('seed-audit-well', 'branch-a') == _stable_pf_seed(
    'seed-audit-well',
    'branch-a',
)
assert _stable_pf_seed('seed-audit-well', 'branch-a') != _stable_pf_seed(
    'seed-audit-well',
    'branch-b',
)
"""


def transform_compact_pf(value: str) -> str:
    value = replace_once(
        value,
        """@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1]=cum[j]+w[j]
    u0=np.random.uniform(0.,1./N)
    np2=np.empty(N); na=np.empty(N); ci=0
    for j in range(N):
        u=u0+j/N
        while ci<N-1 and cum[ci+1]<u: ci+=1
        np2[j]=pos[ci]+rp*np.random.randn()
        na[j] =aux[ci]+rv*np.random.randn()
    return np2,na
""",
        """@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv, u0, pos_noise, aux_noise):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1]=cum[j]+w[j]
    np2=np.empty(N); na=np.empty(N); ci=0
    for j in range(N):
        u=u0+j/N
        while ci<N-1 and cum[ci+1]<u: ci+=1
        np2[j]=pos[ci]+rp*pos_noise[j]
        na[j] =aux[ci]+rv*aux_noise[j]
    return np2,na
""",
        "compact deterministic resampler",
    )
    value = replace_once(
        value,
        """def _pf_ancc(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,
              ALPHA,RN,PN,IS,RP,RR,RESAMP):
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
""",
        """def _pf_ancc(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,
              ALPHA,RN,PN,IS,RP,RR,RESAMP,seed):
    np.random.seed(seed)
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
""",
        "compact ancc kernel seed",
    )
    value = replace_once(
        value,
        """        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            for j in range(N): w[j]=1./N
""",
        """        if 1./ne<RESAMP*N:
            u0=np.random.uniform(0.,1./N)
            pos_noise=np.empty(N); aux_noise=np.empty(N)
            for k in range(N):
                pos_noise[k]=np.random.randn()
                aux_noise[k]=np.random.randn()
            pos,rate=_resamp(pos,rate,w,N,RP,RR,u0,pos_noise,aux_noise)
            for j in range(N): w[j]=1./N
""",
        "compact ancc resample draws",
    )
    value = replace_once(
        value,
        """def _pf_z(md_v,z_v,gr_v,gr_sm_v,gg_p,gg_s,vmin,step,
          gs,ip,iv,beta,icpt,zsig,N,
          MOM,VN,PN,GR_WT,RP,RV,RESAMP):
    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N
""",
        """def _pf_z(md_v,z_v,gr_v,gr_sm_v,gg_p,gg_s,vmin,step,
          gs,ip,iv,beta,icpt,zsig,N,
          MOM,VN,PN,GR_WT,RP,RV,RESAMP,seed):
    np.random.seed(seed)
    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N
""",
        "compact z kernel seed",
    )
    value = replace_once(
        value,
        """        if 1./ne<RESAMP*N:
            pos,vel=_resamp(pos,vel,w,N,RP,RV)
            for j in range(N): w[j]=1./N
""",
        """        if 1./ne<RESAMP*N:
            u0=np.random.uniform(0.,1./N)
            pos_noise=np.empty(N); aux_noise=np.empty(N)
            for k in range(N):
                pos_noise[k]=np.random.randn()
                aux_noise[k]=np.random.randn()
            pos,vel=_resamp(pos,vel,w,N,RP,RV,u0,pos_noise,aux_noise)
            for j in range(N): w[j]=1./N
""",
        "compact z resample draws",
    )
    value = replace_once(
        value,
        "def run_pf_ancc(hw,tw_tvt,tw_gr,N=ANCC_N):",
        "def run_pf_ancc(hw,tw_tvt,tw_gr,N=ANCC_N,seed=0):",
        "compact ancc wrapper seed",
    )
    value = replace_once(
        value,
        "ANCC_ALPHA,ANCC_RN,ANCC_PN,ANCC_IS,ANCC_RP,ANCC_RR,PF_RESAMP)",
        "ANCC_ALPHA,ANCC_RN,ANCC_PN,ANCC_IS,ANCC_RP,ANCC_RR,PF_RESAMP,seed)",
        "compact ancc wrapper pass-through",
    )
    value = replace_once(
        value,
        "def run_pf_z(hw,tw_tvt,tw_gr,N=PF_N):",
        "def run_pf_z(hw,tw_tvt,tw_gr,N=PF_N,seed=0):",
        "compact z wrapper seed",
    )
    value = replace_once(
        value,
        "PF_MOM,PF_VN,PF_PN,PF_GR_WT,PF_ROUGH_P,PF_ROUGH_V,PF_RESAMP)",
        "PF_MOM,PF_VN,PF_PN,PF_GR_WT,PF_ROUGH_P,PF_ROUGH_V,PF_RESAMP,seed)",
        "compact z wrapper pass-through",
    )
    value = replace_once(
        value,
        "_pf_ancc(_md,_z,_gr,_gg,45.,0.1,20.,50.,0.,8,0.998,0.002,0.005,0.3,0.1,0.001,0.5)",
        "_pf_ancc(_md,_z,_gr,_gg,45.,0.1,20.,50.,0.,8,0.998,0.002,0.005,0.3,0.1,0.001,0.5,101)",
        "compact ancc warm-up seed",
    )
    value = replace_once(
        value,
        "_pf_z(_md,_z,_gr,_gr,_gg,_gg,45.,0.1,20.,50.,0.,-1.,0.,0.1,8,"
        "0.993,0.005,0.01,0.3,0.2,0.003,0.5)",
        "_pf_z(_md,_z,_gr,_gr,_gg,_gg,45.,0.1,20.,50.,0.,-1.,0.,0.1,8,"
        "0.993,0.005,0.01,0.3,0.2,0.003,0.5,202)",
        "compact z warm-up seed",
    )
    value = replace_once(
        value,
        "_beam_jit(np.random.randn(30),np.random.randn(50),25,8,15.,100.)",
        "_beam_jit(np.linspace(-1.,1.,30),np.linspace(-2.,2.,50),25,8,15.,100.)",
        "compact deterministic beam warm-up",
    )
    value = replace_once(
        value,
        """    pf_a,std_a=run_pf_ancc(hw,tw_tvt,tw_gr)
    if len(pf_a)==0: return None
    pf_z,std_z=run_pf_z(hw,tw_tvt,tw_gr)
""",
        """    seed_branch='sp45_train' if is_train else 'sp45_test'
    pf_a,std_a=run_pf_ancc(
        hw,tw_tvt,tw_gr,seed=_stable_pf_seed(wid,seed_branch+'_ancc')
    )
    if len(pf_a)==0: return None
    pf_z,std_z=run_pf_z(
        hw,tw_tvt,tw_gr,seed=_stable_pf_seed(wid,seed_branch+'_z')
    )
""",
        "compact per-well scoring seeds",
    )
    return value


def transform_spaced_pf(value: str) -> str:
    value = replace_once(
        value,
        """@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1] = cum[j]+w[j]
    u0 = np.random.uniform(0., 1./N); np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0+j/N
        while ci < N-1 and cum[ci+1] < u: ci += 1
        np2[j] = pos[ci]+rp*np.random.randn(); na[j] = aux[ci]+rv*np.random.randn()
    return np2, na
""",
        """@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv, u0, pos_noise, aux_noise):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1] = cum[j]+w[j]
    np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0+j/N
        while ci < N-1 and cum[ci+1] < u: ci += 1
        np2[j] = pos[ci]+rp*pos_noise[j]; na[j] = aux[ci]+rv*aux_noise[j]
    return np2, na
""",
        "spaced deterministic resampler",
    )
    value = replace_once(
        value,
        """def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
""",
        """def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP, seed):
    np.random.seed(seed)
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
""",
        "spaced ancc kernel seed",
    )
    value = replace_once(
        value,
        """        if 1./ne < RESAMP*N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1./N
""",
        """        if 1./ne < RESAMP*N:
            u0 = np.random.uniform(0., 1./N)
            pos_noise = np.empty(N); aux_noise = np.empty(N)
            for k in range(N):
                pos_noise[k] = np.random.randn(); aux_noise[k] = np.random.randn()
            pos, rate = _resamp(pos, rate, w, N, RP, RR, u0, pos_noise, aux_noise)
            for j in range(N): w[j] = 1./N
""",
        "spaced ancc resample draws",
    )
    value = replace_once(
        value,
        """def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N,
         MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos = np.empty(N); vel = np.empty(N); w = np.ones(N)/N
""",
        """def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N,
         MOM, VN, PN, GR_WT, RP, RV, RESAMP, seed):
    np.random.seed(seed)
    pos = np.empty(N); vel = np.empty(N); w = np.ones(N)/N
""",
        "spaced z kernel seed",
    )
    value = replace_once(
        value,
        """        if 1./ne < RESAMP*N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N): w[j] = 1./N
""",
        """        if 1./ne < RESAMP*N:
            u0 = np.random.uniform(0., 1./N)
            pos_noise = np.empty(N); aux_noise = np.empty(N)
            for k in range(N):
                pos_noise[k] = np.random.randn(); aux_noise[k] = np.random.randn()
            pos, vel = _resamp(pos, vel, w, N, RP, RV, u0, pos_noise, aux_noise)
            for j in range(N): w[j] = 1./N
""",
        "spaced z resample draws",
    )
    value = replace_once(
        value,
        "def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):",
        "def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N, seed=0):",
        "spaced ancc wrapper seed",
    )
    value = replace_once(
        value,
        "ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP)",
        "ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP, seed)",
        "spaced ancc wrapper pass-through",
    )
    value = replace_once(
        value,
        "def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):",
        "def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N, seed=0):",
        "spaced z wrapper seed",
    )
    value = replace_once(
        value,
        "PF_MOM, PF_VN, PF_PN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP)",
        "PF_MOM, PF_VN, PF_PN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP, seed)",
        "spaced z wrapper pass-through",
    )
    return value


def transform_learned_warmup(value: str) -> str:
    value = replace_once(
        value,
        "_pf_ancc(_m, _z, _g, _gg, 45., .1, 20., 50., 0., 8, .998, .002, .005, .3, .1, .001, .5)",
        "_pf_ancc(_m, _z, _g, _gg, 45., .1, 20., 50., 0., 8, "
        ".998, .002, .005, .3, .1, .001, .5, 303)",
        "learned ancc warm-up seed",
    )
    value = replace_once(
        value,
        "_pf_z(_m, _z, _g, _g, _gg, _gg, 45., .1, 20., 50., 0., -1., 0., "
        ".1, 8, .993, .005, .01, .3, .2, .003, .5)",
        "_pf_z(_m, _z, _g, _g, _gg, _gg, 45., .1, 20., 50., 0., -1., 0., "
        ".1, 8, .993, .005, .01, .3, .2, .003, .5, 404)",
        "learned z warm-up seed",
    )
    value = replace_once(
        value,
        "_beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15., 100.)",
        "_beam_jit(np.linspace(-1., 1., 30), np.linspace(-2., 2., 50), 25, 8, 15., 100.)",
        "learned deterministic beam warm-up",
    )
    value = replace_once(
        value,
        "pf, _ = run_pf_ancc(hw, tw_tvt, tw_gr); out, _, _ = lik_pf(",
        "pf, _ = run_pf_ancc("
        "hw, tw_tvt, tw_gr, seed=_stable_pf_seed(wid, 'diagnostic_ancc')"
        "); out, _, _ = lik_pf(",
        "diagnostic ancc seed",
    )
    return value


def transform_learned_build(value: str) -> str:
    return replace_once(
        value,
        """    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0: return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
""",
        """    seed_branch = 'learned_train' if is_train else 'learned_test'
    pf_a, std_a = run_pf_ancc(
        hw, tw_tvt, tw_gr, seed=_stable_pf_seed(wid, seed_branch + '_ancc')
    )
    if len(pf_a) == 0: return None
    pf_z, std_z = run_pf_z(
        hw, tw_tvt, tw_gr, seed=_stable_pf_seed(wid, seed_branch + '_z')
    )
""",
        "learned per-well scoring seeds",
    )


def main() -> None:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    cells = copy.deepcopy(parent["cells"])

    # Identity/provenance changes do not alter scoring math.
    title = cells[0]
    title["source"] = source(title).replace(
        "# ROGII PF Clean128 v5",
        "# ROGII PF Clean128 v5d",
        1,
    ) + (
        "\nV5d differs from v5 only by deterministic seeding of previously "
        "unseeded single-PF kernels.\n"
    )
    flow = cells[1]
    flow["source"] = source(flow) + (
        "\nAll stochastic scoring kernels are seeded from the stable "
        "`well ID + branch` namespace, independent of joblib thread order.\n"
    )

    control = marked_cell(cells, "SUBMISSION_PROFILE = 'clean128_v5'")
    control["source"] = source(control).replace("clean128_v5", "clean128_v5d")
    control_index = cells.index(control)
    cells.insert(
        control_index + 1,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "clean128_stage": "deterministic_seed_control",
                "clean128_v5d_change": "stable_seed_helper",
            },
            "outputs": [],
            "source": SEED_HELPER_SOURCE,
        },
    )

    compact = source_cell(cells, 15)
    compact["source"] = transform_compact_pf(source(compact))
    compact.setdefault("metadata", {})["clean128_v5d_change"] = "seed_sp45_single_pf"

    spaced = source_cell(cells, 37)
    spaced["source"] = transform_spaced_pf(source(spaced))
    spaced.setdefault("metadata", {})["clean128_v5d_change"] = "seed_learned_single_pf"

    warmup = source_cell(cells, 38)
    warmup["source"] = transform_learned_warmup(source(warmup))
    warmup.setdefault("metadata", {})["clean128_v5d_change"] = "deterministic_warmup"

    learned_build = source_cell(cells, 41)
    learned_build["source"] = transform_learned_build(source(learned_build))
    learned_build.setdefault("metadata", {})["clean128_v5d_change"] = "per_well_learned_seeds"

    scoring_audit = marked_cell(cells, "# Clean128 scoring-path audit")
    scoring_audit_source = source(scoring_audit).replace(
        "clean128_v5_scoring_path_audit.json",
        "clean128_v5d_scoring_path_audit.json",
        1,
    )
    scoring_audit_source = replace_once(
        scoring_audit_source,
        "    'query_well_self_exclusion': True,\n",
        "    'query_well_self_exclusion': True,\n"
        "    'deterministic_pf_seeding': bool(PF_DETERMINISTIC_SEEDING),\n"
        "    'pf_seed_namespace': PF_SEED_NAMESPACE,\n",
        "scoring audit deterministic fields",
    )
    scoring_audit["source"] = scoring_audit_source

    contract = marked_cell(cells, "# Final hidden-set contract transaction.")
    contract["source"] = source(contract).replace(
        "'derived_kernel': 'heralces/rogii-pf-clean128-v5'",
        "'derived_kernel': 'heralces/rogii-pf-clean128-v5d'",
        1,
    )
    final_audit = marked_cell(cells, "# Final immutable scoring audit")
    final_audit["source"] = (
        source(final_audit)
        .replace(
            "clean128_v5_final_audit.json",
            "clean128_v5d_final_audit.json",
            1,
        )
        .replace(
            "Clean128 v5 final audit:",
            "Clean128 v5d final audit:",
            1,
        )
    )

    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    metadata = copy.deepcopy(parent.get("metadata", {}))
    v5_metadata = copy.deepcopy(metadata.pop("clean128_v5"))
    v5_metadata.update(
        {
            "variant_parent": "heralces/rogii-pf-clean128-v5",
            "behavioral_delta": "deterministic single-PF seeding only",
            "seed_derivation": "sha256(namespace, well_id, branch) -> signed31",
            "seeded_single_pf_branches": [
                "sp45_train_ancc",
                "sp45_train_z",
                "sp45_test_ancc",
                "sp45_test_z",
                "learned_train_ancc",
                "learned_train_z",
                "learned_test_ancc",
                "learned_test_z",
            ],
            "resampler_random_draws_owned_by_seeded_parent": True,
        }
    )
    metadata["clean128_v5d"] = v5_metadata

    output = {
        "cells": cells,
        "metadata": metadata,
        "nbformat": parent.get("nbformat", 4),
        "nbformat_minor": parent.get("nbformat_minor", 5),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT} with {len(cells)} cells")


if __name__ == "__main__":
    main()
