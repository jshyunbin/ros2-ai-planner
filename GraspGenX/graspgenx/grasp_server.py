# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import time
import numpy as np
import omegaconf
import torch

from graspgenx.dataset.dataset import collate
from graspgenx.models.grasp_gen import GraspGen
from graspgenx.utils.point_cloud import knn_points, point_cloud_outlier_removal
from graspgenx.x_grippers import resolve_gripper_info
from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)


class GraspGenXSampler:
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        gripper_name: str,
        assets_dir: str = None,
    ):
        """
        Args:
            cfg: Hydra config object
            gripper_name: Name of the gripper (must exist in assets)
            assets_dir: Root directory containing x_grippers/ and proc_grippers/
                        subdirectories. Defaults to /code/assets (docker path).
        """

        self.cfg = cfg
        self.gripper_name = gripper_name

        if assets_dir is None:
            assets_dir = "/code/assets"

        self.gripper = resolve_gripper_info(gripper_name, assets_dir)

        model = GraspGen.from_config(cfg.diffusion, cfg.discriminator)
        if not os.path.exists(cfg.eval.gen_checkpoint):
            raise FileNotFoundError(
                f"Checkpoint {cfg.eval.gen_checkpoint} does not exist"
            )
        if not os.path.exists(cfg.eval.dis_checkpoint):
            raise FileNotFoundError(
                f"Checkpoint {cfg.eval.dis_checkpoint} does not exist"
            )

        model.load_state_dict(cfg.eval.gen_checkpoint, cfg.eval.dis_checkpoint)
        model.eval()

        self.model = model.cuda().eval()

    @staticmethod
    def run_inference(
        object_pc: np.ndarray | torch.Tensor,
        grasp_sampler: "GraspGenXSampler",
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = -1,
        min_grasps: int = 40,
        max_tries: int = 6,
        remove_outliers: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run grasp generation inference on a point cloud.

        Args:
            object_pc: Point cloud to generate grasps for
            grasp_sampler: Initialized GraspGenXSampler instance
            grasp_threshold: Threshold for valid grasps. If -1.0, then the top topk_grasps grasps will be ranked and returned
            num_grasps: Number of grasps to generate
            topk_grasps: Maximum number of grasps to return
            min_grasps: Minimum number of grasps required. If fewer grasps are found, inference will be retried
            max_tries: Maximum number of inference attempts to make before returning results

        Returns:
            grasps: Generated grasp poses
            grasp_conf: Confidence scores for the grasps
        """
        if type(object_pc) == np.ndarray:
            object_pc = torch.from_numpy(object_pc).cuda().float()

        if grasp_threshold == -1.0 and topk_num_grasps == -1:
            topk_num_grasps = 100

        all_grasps = []
        all_conf = []
        num_tries = 0

        while sum(len(g) for g in all_grasps) < min_grasps and num_tries < max_tries:
            num_tries += 1
            t0 = time.time()
            output = grasp_sampler.sample(
                object_pc,
                threshold=grasp_threshold,
                num_grasps=num_grasps,
                remove_outliers=remove_outliers,
            )
            grasp_conf = output[1]
            grasps = output[0]

            # Sort and prune grasps within this iteration
            if topk_num_grasps != -1 and len(grasps) > 0:
                grasp_conf, grasps = zip(
                    *sorted(zip(grasp_conf, grasps), key=lambda x: x[0], reverse=True)
                )
                grasps = torch.stack(grasps)
                grasp_conf = torch.stack(grasp_conf)
                grasps = grasps[:topk_num_grasps]
                grasp_conf = grasp_conf[:topk_num_grasps]

            all_grasps.append(grasps)
            all_conf.append(grasp_conf)

            logger.info(
                f"Found {len(grasps)} grasps in iteration {len(all_grasps)}, total grasps: {sum(len(g) for g in all_grasps)}"
            )
            t1 = time.time()
            logger.info(f"Time taken for inference: {t1 - t0} seconds")

        if len(all_grasps) == 0:
            return torch.tensor([]), torch.tensor([])

        # Concatenate all grasps and confidences
        grasps = torch.cat(all_grasps, dim=0)
        grasp_conf = torch.cat(all_conf, dim=0)
        grasps[:, 3, 3] = 1  # TODO: Fix this in grasp_gen.py later.

        return grasps, grasp_conf

    @staticmethod
    def run_inference_batch(
        object_pcs: list,
        grasp_sampler: "GraspGenXSampler",
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = -1,
        remove_outliers: bool = True,
    ) -> list:
        """Batched run_inference: one diffusion + discriminator forward pass
        over N object PCs. Equivalent to ``[run_inference(pc, ...) for pc in
        object_pcs]`` but folds the reverse-diffusion loop into a single
        batched call.

        Each input PC is resampled (with replacement) to
        ``grasp_sampler.cfg.data.num_points`` so that ``collate`` can stack
        them — this matches the training distribution. Per-object centers
        are tracked and added back to the predicted grasps.

        Args:
            object_pcs: list of N point clouds, each (Mi, 3) np.ndarray or
                        torch.Tensor in the same world frame.
            grasp_threshold: per-object discriminator threshold; -1.0 keeps
                             everything then prunes by topk.
            num_grasps: diffusion samples per object (= the model's
                        ``num_grasps_per_object``).
            topk_num_grasps: cap per object after thresholding; -1 means no
                             top-k cap unless ``grasp_threshold == -1.0``
                             (in which case defaults to 100, matching
                             ``run_inference``).
            remove_outliers: run outlier removal per object before
                             resampling.

        Returns:
            List of (grasps, grasp_conf) tuples, one per input PC, in
            input order. Shapes: grasps (Ki, 4, 4), grasp_conf (Ki,) on the
            sampler's device.
        """
        if len(object_pcs) == 0:
            return []
        if grasp_threshold == -1.0 and topk_num_grasps == -1:
            topk_num_grasps = 100

        device = next(grasp_sampler.model.parameters()).device
        target_n = int(grasp_sampler.cfg.data.num_points)

        centers: list = []
        data_items: list = []
        for pc in object_pcs:
            if isinstance(pc, np.ndarray):
                pc_t = torch.from_numpy(pc).float()
            else:
                pc_t = pc.float()
            if remove_outliers:
                pc_t, _ = point_cloud_outlier_removal(pc_t)
            # Resample with replacement to the training-time budget so all
            # batch items have the same shape (collate uses torch.stack).
            n = pc_t.shape[0]
            if n != target_n:
                idx = torch.randint(0, n, (target_n,))
                pc_t = pc_t[idx]
            pc_t = pc_t.to(device)
            center = pc_t.mean(dim=0)
            centers.append(center)
            centered = pc_t - center[None]
            color = torch.zeros_like(centered)

            data = {}
            data["task"] = "pick"
            data["inputs"] = torch.cat([centered, color[:, :3]], dim=-1).float()
            data["points"] = centered
            data = grasp_sampler.load_gripper_input(data)
            data_items.append(data)

        data_batch = collate(data_items)
        data_batch["grasp_key"] = "grasps_pred"

        # Set the diffusion sampler to produce num_grasps per object; the
        # model's batched forward already returns (N, num_grasps, 4, 4).
        grasp_sampler.model.grasp_generator.num_grasps_per_object = num_grasps

        with torch.inference_mode():
            model_outputs, _, _ = grasp_sampler.model.infer(data_batch)

        grasps_pred = model_outputs["grasps_pred"]  # (N, K, 4, 4)
        grasp_conf = model_outputs["grasp_confidence"][..., 0]  # (N, K)

        outputs: list = []
        for i, center in enumerate(centers):
            g_i = grasps_pred[i].to(device)
            c_i = grasp_conf[i].to(device)
            if grasp_threshold > 0.0:
                keep = c_i >= grasp_threshold
                g_i = g_i[keep]
                c_i = c_i[keep]
            if topk_num_grasps != -1 and len(g_i) > topk_num_grasps:
                order = torch.argsort(c_i, descending=True)[:topk_num_grasps]
                g_i = g_i[order]
                c_i = c_i[order]
            # Restore world frame.
            if len(g_i) > 0:
                g_i = g_i.clone()
                g_i[:, :3, 3] = g_i[:, :3, 3] + center.to(g_i.device)
                g_i[:, 3, 3] = 1.0
            outputs.append((g_i, c_i))
        return outputs

    @torch.inference_mode()
    def sample(
        self,
        obj_pcd: np.ndarray,
        threshold: float = -1.0,
        num_grasps: int = 200,
        remove_outliers: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obj_pcd: np.array of shape (N, 3)
            obj_pts_color (Optional): np.array of shape (N, 4)

        Returns:
            grasps: torch.tensor of shape (M, 6)
            grasp_conf: torch.tensor of shape (M,)
            grasp_contacts: torch.tensor of shape (M, 3)
        """

        if remove_outliers:
            obj_pcd, _ = point_cloud_outlier_removal(obj_pcd)

        obj_pcd_center = obj_pcd.mean(axis=0)
        obj_pts_color = torch.zeros_like(obj_pcd)
        obj_mean_points = obj_pcd - obj_pcd_center[None]

        data = {}
        data["task"] = "pick"
        data["inputs"] = torch.cat(
            [obj_mean_points, obj_pts_color[:, :3].squeeze(1)], dim=-1
        ).float()
        data["points"] = obj_mean_points
        data = self.load_gripper_input(data)

        data_batch = collate([data])
        grasp_key = "grasps"
        with torch.inference_mode():
            grasp_key = "grasps_pred"
            self.model.grasp_generator.num_grasps_per_object = num_grasps
            model_outputs, _, _ = self.model.infer(data_batch)

        if len(model_outputs[grasp_key][0]) == 0:
            return [], [], []

        grasps = model_outputs[grasp_key][0]

        grasp_conf = model_outputs["grasp_confidence"][0][:, 0]
        logger.info(
            f"Confidences min: {grasp_conf.min():.5f}, max: {grasp_conf.max():.5f}"
        )
        mask_best_grasps = grasp_conf >= threshold
        logger.info(
            f"Thresholding grasps @ {threshold}. Only {mask_best_grasps.sum()}/{mask_best_grasps.shape[0]} grasps remaining"
        )

        grasps = grasps[mask_best_grasps]
        grasp_conf = grasp_conf[mask_best_grasps]

        grasps[:, :3, 3] += obj_pcd_center
        return grasps, grasp_conf, None

    @torch.inference_mode()
    def load_gripper_input(self, outputs):

        outputs["z_offset"] = torch.tensor(
            (self.gripper.depth,), dtype=torch.float32
        ).cuda()
        outputs["sweep_volume"] = torch.from_numpy(
            self.gripper.sweep_volume.astype(np.float32)
        ).cuda()
        outputs["gripper_type"] = self.gripper.gripper_type

        sweep_volume_open = self.gripper.sweep_volume.astype(np.float32)
        sweep_volume_mid = self.gripper.sweep_volume_mid.astype(np.float32)
        outputs["sweep_volume_open_and_mid"] = torch.from_numpy(
            np.concatenate([sweep_volume_open, sweep_volume_mid], axis=0)
        ).cuda()

        gripper_open_ptc = self.gripper.open_pointcloud.copy()
        gripper_close_ptc = self.gripper.close_pointcloud.copy()

        mask = np.random.randint(
            0, gripper_open_ptc.shape[0], (self.cfg.data.num_points,)
        )
        gripper_open_ptc = torch.from_numpy(
            gripper_close_ptc[mask].astype(np.float32)
        ).cuda()

        mask = np.random.randint(
            0, gripper_close_ptc.shape[0], (self.cfg.data.num_points,)
        )
        gripper_close_ptc = torch.from_numpy(
            gripper_close_ptc[mask].astype(np.float32)
        ).cuda()

        outputs["gripper_open_ptc"] = gripper_open_ptc
        outputs["gripper_close_ptc"] = gripper_close_ptc

        vol_tsdf = np.stack(
            [self.gripper.vol_tsdf[f"open_tsdf"], self.gripper.vol_tsdf[f"close_tsdf"]],
            axis=0,
        )
        outputs["gripper_vol_tsdf"] = torch.from_numpy(
            vol_tsdf.astype(np.float32)
        ).cuda()

        outputs["gripper_pointnet_repr"] = torch.from_numpy(
            np.concatenate(
                [
                    self.gripper.pointnet_vae["open"],
                    self.gripper.pointnet_vae["half"],
                    self.gripper.pointnet_vae["close"],
                ],
                axis=0,
            ).astype(np.float32)
        ).cuda()

        return outputs

    def get_gripper_info(self):
        return self.gripper
