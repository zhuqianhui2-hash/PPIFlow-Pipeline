#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import math
import glob
import shutil
from pathlib import Path

import torch
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict


FLOWPACKER_REPO = os.environ.get("FLOWPACKER_REPO")
if FLOWPACKER_REPO:
    sys.path.insert(0, FLOWPACKER_REPO)

from utils.loader import load_seed, load_device, load_ema, load_checkpoint, load_config
from utils.logger import Logger, set_log
from utils.train_utils import count_parameters
from dataset_cluster import get_dataloader
from utils.structure_utils import create_structure_from_crds
from utils.sidechain_utils import Idealizer
from models.cnf import CNF
from models.confidence import Confidence
from models.equiformer_v2.equiformer_v2 import EquiformerV2
from utils.metrics import metrics_per_chi, atom_rmsd
from utils.constants import chi_mask as chi_mask_true
from utils.constants import atom14_mask as atom_mask_true


def _split_multichain_sequence(seq: str) -> list[str]:
    # ProteinMPNN/AbMPNN multi-chain outputs commonly use "/" separators.
    if "/" in seq:
        return [part.strip() for part in seq.split("/") if part.strip()]
    return [seq.strip()]


def _parse_chain_list(raw: str | None) -> list[str]:
    if not raw:
        return ["A"]
    out: list[str] = []
    for token in str(raw).split(","):
        chain = token.strip()
        if not chain:
            continue
        if chain not in out:
            out.append(chain)
    return out or ["A"]


def _assign_sequences_to_chains(
    raw_sequence: str,
    chain_order: list[str],
    chain_lengths: dict[str, int],
) -> dict[str, str] | None:
    """
    Map an input sequence string to requested chains.

    Supports:
    - single segment (single-chain)
    - delimiter-separated segments (e.g. HEAVY/LIGHT)
    - concatenated heavy+light string (split by chain lengths)
    """
    # Chain-labeled segments (e.g. "A:SEQ/C:SEQ") provide unambiguous mapping.
    labeled_parts = [part.strip() for part in str(raw_sequence).split("/") if part.strip()]
    if labeled_parts and all(":" in part for part in labeled_parts):
        labeled: dict[str, str] = {}
        for part in labeled_parts:
            chain, seq = part.split(":", 1)
            chain = chain.strip()
            seq = seq.strip()
            if not chain or not seq:
                return None
            labeled[chain] = seq
        out_labeled: dict[str, str] = {}
        for chain in chain_order:
            seq = labeled.get(chain)
            target_len = int(chain_lengths.get(chain) or 0)
            if not seq or target_len <= 0 or len(seq) != target_len:
                return None
            out_labeled[chain] = seq
        return out_labeled

    segments = [seg for seg in _split_multichain_sequence(raw_sequence) if seg]
    if not segments:
        return None

    # Fast path: single active chain.
    if len(chain_order) == 1:
        chain = chain_order[0]
        target_len = int(chain_lengths.get(chain) or 0)
        if target_len <= 0:
            return None
        if len(segments) == 1 and len(segments[0]) == target_len:
            return {chain: segments[0]}
        matches = [seg for seg in segments if len(seg) == target_len]
        if len(matches) == 1:
            return {chain: matches[0]}
        return None

    # Ordered segments, one per chain.
    if len(segments) == len(chain_order):
        ordered_ok = True
        for idx, chain in enumerate(chain_order):
            if len(segments[idx]) != int(chain_lengths.get(chain) or 0):
                ordered_ok = False
                break
        if ordered_ok:
            return {chain: segments[idx] for idx, chain in enumerate(chain_order)}

    # Single concatenated segment for multiple chains.
    if len(segments) == 1:
        merged = segments[0]
        expected = sum(int(chain_lengths.get(chain) or 0) for chain in chain_order)
        if len(merged) == expected:
            out: dict[str, str] = {}
            offset = 0
            for chain in chain_order:
                L = int(chain_lengths.get(chain) or 0)
                out[chain] = merged[offset : offset + L]
                offset += L
            return out

    # Fallback: unique length-based assignment.
    out: dict[str, str] = {}
    used: set[int] = set()
    for chain in chain_order:
        target_len = int(chain_lengths.get(chain) or 0)
        candidate_idxs = [i for i, seg in enumerate(segments) if i not in used and len(seg) == target_len]
        if len(candidate_idxs) != 1:
            return None
        idx = candidate_idxs[0]
        out[chain] = segments[idx]
        used.add(idx)
    return out


def adding_aatype(csv_path, before_pdb_dir, pdb_output_dir, binder_chain="A"):
    one_to_three = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }

    import pandas as pd

    df = pd.read_csv(csv_path)
    os.makedirs(pdb_output_dir, exist_ok=True)
    requested_chains = _parse_chain_list(binder_chain)
    require_all_requested = len(requested_chains) > 1

    existing_pdbs = set(os.path.basename(p) for p in glob.glob(f"{before_pdb_dir}/*.pdb"))
    stats = {
        "rows": int(len(df)),
        "written": 0,
        "missing_template_pdb": 0,
        "empty_sequence": 0,
        "missing_binder_chain": 0,  # required chain A missing
        "missing_required_chain": 0,
        "length_mismatch": 0,
        "ambiguous_chain_assignment": 0,
    }
    count = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pdb_name = os.path.basename(str(row["link_name"]))
        if pdb_name not in existing_pdbs:
            stats["missing_template_pdb"] += 1
            continue

        full_pdb_path = os.path.join(before_pdb_dir, pdb_name)
        if not os.path.exists(full_pdb_path):
            stats["missing_template_pdb"] += 1
            continue

        with open(full_pdb_path, "r") as f:
            pdb_lines = f.readlines()

        seq_val = row.get("seq")
        if pd.isna(seq_val):
            stats["empty_sequence"] += 1
            continue
        sequence = str(seq_val).strip()
        if not sequence or sequence.lower() == "nan":
            stats["empty_sequence"] += 1
            continue

        res_indices_by_chain: dict[str, list[str]] = {chain: [] for chain in requested_chains}
        last_resi_by_chain: dict[str, str | None] = {chain: None for chain in requested_chains}
        for line in pdb_lines:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21]
            if chain_id not in res_indices_by_chain:
                continue
            resi = line[22:26]
            if resi != last_resi_by_chain[chain_id]:
                res_indices_by_chain[chain_id].append(resi)
                last_resi_by_chain[chain_id] = resi

        # Always require chain A when requested.
        if "A" in requested_chains and not res_indices_by_chain.get("A"):
            stats["missing_binder_chain"] += 1
            continue

        if require_all_requested:
            missing = [chain for chain in requested_chains if not res_indices_by_chain.get(chain)]
            if missing:
                stats["missing_required_chain"] += 1
                continue

        active_chains = [chain for chain in requested_chains if res_indices_by_chain.get(chain)]
        if not active_chains:
            stats["missing_binder_chain"] += 1
            continue

        chain_lengths = {chain: len(res_indices_by_chain[chain]) for chain in active_chains}
        chain_to_sequence = _assign_sequences_to_chains(sequence, active_chains, chain_lengths)
        if not chain_to_sequence:
            if len(active_chains) > 1 and len(set(chain_lengths.values())) < len(chain_lengths):
                stats["ambiguous_chain_assignment"] += 1
            stats["length_mismatch"] += 1
            continue

        resi_to_resname: dict[tuple[str, str], str] = {}
        for chain in active_chains:
            seq_for_chain = chain_to_sequence.get(chain, "")
            for resi, aa in zip(res_indices_by_chain[chain], seq_for_chain):
                resi_to_resname[(chain, resi)] = one_to_three.get(aa, "UNK")

        new_lines = []
        for line in pdb_lines:
            if line.startswith("ATOM"):
                chain_id = line[21]
                resi = line[22:26]
                key = (chain_id, resi)
                if key in resi_to_resname:
                    new_resname = resi_to_resname[key]
                    line = line[:17] + new_resname.ljust(3) + line[20:]
            new_lines.append(line)

        output_pdb = os.path.join(
            pdb_output_dir,
            os.path.basename(row["link_name"]).replace(".pdb", f"_{row['seq_idx']}.pdb"),
        )
        with open(output_pdb, "w") as f:
            f.writelines(new_lines)

        count += 1
        stats["written"] += 1

    print(f"Wrote {count} sequence-tagged PDBs to {pdb_output_dir}")
    print(
        "FlowPacker precheck:"
        f" rows={stats['rows']}"
        f" written={stats['written']}"
        f" missing_template_pdb={stats['missing_template_pdb']}"
        f" empty_sequence={stats['empty_sequence']}"
        f" missing_binder_chain={stats['missing_binder_chain']}"
        f" missing_required_chain={stats['missing_required_chain']}"
        f" length_mismatch={stats['length_mismatch']}"
        f" ambiguous_chain_assignment={stats['ambiguous_chain_assignment']}",
        flush=True,
    )
    return stats


class Sampler(object):
    def __init__(self, config, use_gt_masks=False, ddp=False):
        super(Sampler, self).__init__()
        self.config = config
        self.use_gt_masks = use_gt_masks
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader, _, _ = get_dataloader(self.config, ddp=ddp, sample=True)
        self.idealizer = Idealizer(use_native_bb_coords=True)

    def sample(self, ts, name="test", save_traj=False, inpaint=""):
        self.config.exp_name = ts
        self.ckpt = f"{ts}"

        ckpt_dict = torch.load(self.config.ckpt)
        train_cfg = ckpt_dict["config"]
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(train_cfg)
        self.model = CNF(
            EquiformerV2(**train_cfg.model),
            train_cfg,
            coeff=self.config.sample.coeff,
            stepsize=self.config.sample.num_steps,
            mode=self.config.mode,
        ).cuda()
        print(f"Number of parameters: {count_parameters(self.model)}")
        self.ema = load_ema(self.model, decay=train_cfg.train.ema)
        self.model, self.ema = load_checkpoint(self.model, self.ema, ckpt_dict)
        self.model.eval()
        self.ema.copy_to(self.model.parameters())

        if self.config.conf_ckpt is not None:
            conf_ckpt = torch.load(self.config.conf_ckpt)
            self.conf_model = Confidence(EquiformerV2(**conf_ckpt["config"].model), conf_ckpt["config"]).cuda()
            if "module." in list(conf_ckpt["state_dict"].keys())[0]:
                state_dict = {k[7:]: v for k, v in conf_ckpt["state_dict"].items()}
            self.conf_model.load_state_dict(state_dict)

        logger = Logger(str(os.path.join(self.log_dir, f"{self.ckpt}.log")), mode="a")
        logger.log(f"{self.ckpt}", verbose=False)

        save_path = Path(f"{args.save_dir}")
        save_path.mkdir(exist_ok=True, parents=True)
        sample_path = save_path
        sample_path.mkdir(exist_ok=True, parents=True)

        output_dict = {}
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                batch = batch.to(f"cuda:{self.device[0]}")
                aa_str, aa_onehot, aa_num, coords, mask, atom_mask, batch_id, pdb_codes = (
                    batch.aa_str,
                    batch.aa_onehot,
                    batch.aa_num,
                    batch.pos,
                    batch.aa_mask,
                    batch.atom_mask,
                    batch.batch,
                    batch.id,
                )
                chi, chi_alt, chi_mask = batch.chi, batch.chi_alt, batch.chi_mask
                bb_coords = coords[:, :4]

                if self.use_gt_masks:
                    chi_mask = chi_mask_true.to(aa_num)
                    chi_mask = chi_mask[aa_num]
                    batch.chi_mask = chi_mask

                    atom_mask = atom_mask_true.to(atom_mask)
                    atom_mask = atom_mask[aa_num]
                    batch.atom_mask = atom_mask

                chi = (chi + math.pi) * chi_mask
                chi_alt = (chi_alt + math.pi) * chi_mask

                batch_size = batch_id.max().item() + 1
                output_dict = {**output_dict, **{i: {} for i in pdb_codes}}

                for sample_idx in range(self.config.sample.n_samples):
                    exists = True
                    for i in range(batch_size):
                        pdb_path = sample_path.joinpath(f"run_{sample_idx + 1}", f"{pdb_codes[i]}.pdb")
                        if not pdb_path.exists():
                            exists = False
                    if exists:
                        continue

                    pred_sc = self.model.decode(batch, return_traj=save_traj, inpaint=inpaint)
                    pred_sc = (pred_sc - math.pi) * chi_mask

                    if save_traj:
                        pred_sc_traj = pred_sc.clone()
                        pred_sc = pred_sc[-1]

                    all_atom_coords = self.idealizer(aa_num, bb_coords, pred_sc) * atom_mask.unsqueeze(-1)
                    gt_idealized = self.idealizer(aa_num, bb_coords, chi - math.pi) * atom_mask.unsqueeze(-1)

                    for i in range(batch_size):
                        metrics = {}
                        chi_batch = chi[batch_id == i]
                        chi_alt_batch = chi_alt[batch_id == i]
                        chi_mask_batch = chi_mask[batch_id == i]
                        pred_batch = pred_sc[batch_id == i]
                        atom_mask_batch = atom_mask[batch_id == i]
                        crds_batch = coords[batch_id == i]
                        crds_batch_idealized = gt_idealized[batch_id == i]
                        pred_pos_batch = all_atom_coords[batch_id == i]
                        chain_id_batch = batch.chain_id[i]
                        res_id_batch = batch.res_id[i]
                        icode_batch = batch.icode[i]

                        cb_dist = torch.cdist(crds_batch[:, 4], crds_batch[:, 4])
                        cb_dist_w10 = ((cb_dist < 10) * cb_dist != 0).sum(-1)
                        core = cb_dist_w10 >= 20
                        surface = cb_dist_w10 <= 15
                        mae, acc = metrics_per_chi(pred_batch, chi_batch, chi_alt_batch, chi_mask_batch)
                        core_mae, core_acc = metrics_per_chi(
                            pred_batch[core], chi_batch[core], chi_alt_batch[core], chi_mask_batch[core]
                        )
                        surface_mae, surface_acc = metrics_per_chi(
                            pred_batch[surface],
                            chi_batch[surface],
                            chi_alt_batch[surface],
                            chi_mask_batch[surface],
                        )
                        rmsd = atom_rmsd(pred_pos_batch[:, 4:], crds_batch[:, 4:], atom_mask_batch[:, 4:])
                        rmsd_idealized = atom_rmsd(
                            pred_pos_batch[:, 4:], crds_batch_idealized[:, 4:], atom_mask_batch[:, 4:]
                        )
                        core_rmsd = atom_rmsd(
                            pred_pos_batch[core][:, 4:], crds_batch[core][:, 4:], atom_mask_batch[core][:, 4:]
                        )
                        surface_rmsd = atom_rmsd(
                            pred_pos_batch[surface][:, 4:],
                            crds_batch[surface][:, 4:],
                            atom_mask_batch[surface][:, 4:],
                        )
                        clash = 0

                        metrics["angle_mae"] = mae
                        metrics["angle_acc"] = acc
                        metrics["core_mae"] = core_mae
                        metrics["core_acc"] = core_acc
                        metrics["surf_mae"] = surface_mae
                        metrics["surf_acc"] = surface_acc
                        metrics["atom_rmsd"] = rmsd
                        metrics["atom_rmsd_ideal"] = rmsd_idealized
                        metrics["core_rmsd"] = core_rmsd
                        metrics["surface_rmsd"] = surface_rmsd
                        metrics["clash"] = clash

                        output_dict[pdb_codes[i]][f"run_{sample_idx + 1}"] = metrics

                        pdb_path = sample_path.joinpath(f"run_{sample_idx + 1}", f"{pdb_codes[i]}.pdb")
                        pdb_path.parent.mkdir(exist_ok=True, parents=True)

                        if save_traj:
                            aa_batch = aa_num[batch_id == i]
                            crds_traj = []
                            for traj in pred_sc_traj:
                                traj_batch = traj[batch_id == i]
                                crds = (
                                    self.idealizer(aa_batch, pred_pos_batch[:, :4], traj_batch)
                                    * atom_mask_batch.unsqueeze(-1)
                                )
                                crds_traj.append(crds)
                            crds_traj = torch.stack(crds_traj)
                            create_structure_from_crds(
                                aa_str[i],
                                crds_traj.cpu(),
                                atom_mask_batch.cpu(),
                                chain_id_batch,
                                resseq=res_id_batch,
                                icode=icode_batch,
                                outPath=str(pdb_path),
                                save_traj=True,
                            )
                        else:
                            create_structure_from_crds(
                                aa_str[i],
                                pred_pos_batch.cpu(),
                                atom_mask_batch.cpu(),
                                chain_id_batch,
                                resseq=res_id_batch,
                                icode=icode_batch,
                                outPath=str(pdb_path),
                                save_traj=False,
                            )

                        if self.config.conf_ckpt is not None:
                            best_path = sample_path.joinpath("best_run", f"{pdb_codes[i]}.pdb")
                            best_path.parent.mkdir(exist_ok=True, parents=True)

                            pred_rmsd, gt_rmsd = self.conf_model.get_pred(pred_sc, batch)
                            pred_rmsd = pred_rmsd.mean().item()
                            gt_rmsd = gt_rmsd.mean().item()
                            if best_pred_rmsd > pred_rmsd:
                                best_pred_idx = sample_idx
                                best_pred_rmsd = pred_rmsd
                                shutil.copy(
                                    sample_path.joinpath(f"run_{best_pred_idx + 1}", f"{pdb_codes[i]}.pdb"), best_path
                                )
                                if best_gt_rmsd > gt_rmsd:
                                    best_gt_idx = sample_idx
                                    best_gt_rmsd = gt_rmsd

                                output_dict[pdb_codes[i]]["best_pred_idx"] = best_pred_idx + 1
                                output_dict[pdb_codes[i]]["best_gt_idx"] = best_gt_idx + 1
                                output_dict[pdb_codes[i]]["best_pred_rmsd"] = best_pred_rmsd
                                output_dict[pdb_codes[i]]["best_gt_rmsd"] = best_gt_rmsd

            torch.save(output_dict, sample_path.joinpath("output_dict.pth"))

        return self.ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--save_traj", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_gt_masks", type=bool, default=False)
    parser.add_argument("--inpaint", type=str, default="")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--binder_chain", type=str, default=os.environ.get("FLOWPACKER_BINDER_CHAIN", "A"))
    args = parser.parse_args()

    start_time = time.time()
    print(f"results will save in {args.save_dir}")

    if os.path.isfile(args.config):
        config = edict(yaml.safe_load(open(args.config, "r")))
        config.seed = args.seed
    else:
        config = load_config(args.config, seed=args.seed, inference=True)
    before_pdb_dir = config.data.test_path
    after_pdb_dir_batch = os.path.join(
        os.path.dirname(args.save_dir), "after_pdbs_batch", f"{os.path.basename(before_pdb_dir)}"
    )

    config.data.test_path = after_pdb_dir_batch
    stats = adding_aatype(args.csv_file, before_pdb_dir, after_pdb_dir_batch, args.binder_chain)
    if int(stats.get("written", 0)) <= 0:
        raise RuntimeError(
            "FlowPacker preprocessing produced 0 sequence-tagged PDBs. "
            f"Check binder_chain={args.binder_chain}, link_name vs input PDB names, and sequence length/format "
            "(multi-chain sequences may need explicit heavy/light separation or chain-labeled format like A:.../C:...)."
        )

    after_pdbs = os.path.join(os.path.dirname(args.save_dir), "after_pdbs")
    os.makedirs(after_pdbs, exist_ok=True)

    sampler = Sampler(config, args.use_gt_masks)
    ts = time.strftime("%b%d-%H:%M:%S", time.gmtime())
    # Run sampler in a batch-specific working dir so we can safely collect outputs.
    batch_name = Path(args.config).stem
    work_dir = Path(args.save_dir).resolve() / "_work" / batch_name
    work_dir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        sampler.sample(ts, name="run_1", save_traj=args.save_traj, inpaint=args.inpaint)
    finally:
        os.chdir(cwd)
    print(f"Inference took {time.time() - start_time:.2f} seconds")

    # Copy predicted PDBs into save_dir/run_1 for downstream steps.
    src_dir = work_dir / "samples" / "run_1" / "run_1"
    dst_dir = Path(args.save_dir).resolve() / "run_1"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pdb_file in src_dir.glob("*.pdb"):
        shutil.copy2(pdb_file, dst_dir / pdb_file.name)

    all_after_pdbs = glob.glob(f"{after_pdb_dir_batch}/*pdb")
    for src_file in all_after_pdbs:
        dst_file = os.path.join(after_pdbs, os.path.basename(src_file))
        shutil.copy2(src_file, dst_file)
