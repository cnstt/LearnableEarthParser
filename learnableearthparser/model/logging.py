import torch
import numpy as np
from ..utils import chamfer_distance


class LoggingModel:

    def do_greedy_step(self):
        return (self.current_epoch % int(self.hparams.log_every_n_epochs)) == 0

    @torch.no_grad()
    def greedy_step(self, batch, out, batch_idx, tag, batch_size):

        if tag == "train":
            n = (out["choice"] != -1).sum()
            self.log(f'Prediction/N_protos', n / batch_size, on_step=False, on_epoch=True, batch_size = batch_size)

            if hasattr(self, "_protosfeat"):
                for i, val in enumerate(self.get_protosfeat().flatten()):
                    self.log(f'_protosfeat/{i}', val.item(), on_step=False, on_epoch=True, batch_size = batch_size)

            if hasattr(self, "_protosscale") and self.hparams.protos.learn_specific_scale:
                for k in range(self.hparams.K):
                    for xzy, val in zip(["x", "y", "z"], self.get_protosscale()[k].flatten()):
                        self.log(f'_protosscale/{xzy}_{k}', val.item(), on_step=False, on_epoch=True, batch_size = batch_size)


        if batch_idx == 0 and self.do_greedy_step():
            with torch.no_grad():
                if tag == "train":
                    self.greedy_model()
                    self.greedy_histograms(batch, out)
                if out["recs"] is not None:
                    self.greedy_pcs(batch, out, tag)

    @torch.no_grad()
    def greedy_model(self):
        # Plot Protos
        points, faces = None, None

        config_dict = {
            'material': {
                'cls': 'PointsMaterial',
                'size': 0.3 # Default 0.05
            }
        }

        protos = self.get_protos(points)

        for ixyz, xyz in enumerate(["x", "y", "z"]):
            self.logger.experiment.add_histogram(
                    f"protos/{xyz}",
                    protos[..., ixyz].detach().cpu().flatten(),
                    global_step=self.current_epoch
                )       

        mini, maxi = protos.min(1)[0].unsqueeze(1), protos.max(1)[0].unsqueeze(1)
        
        colors = (255 * (protos - mini) / (maxi - mini + 10e-8)).int()

        self.logger.experiment.add_mesh(
            f"protos_pointcloud", protos,
            colors=colors, faces=faces,
            config_dict=config_dict, global_step=self.current_epoch
        )

    
    @torch.no_grad()    
    def greedy_histograms(self, batch, out):

        if "kappa_presoftmax" in out.keys():
            self.logger.experiment.add_histogram(
                f"kappa/presoftmax",
                out["kappa_presoftmax"].detach().cpu().flatten(),
                global_step=self.current_epoch
            )
        if "kappa_postsoftmax" in out.keys():
            self.logger.experiment.add_histogram(
                f"kappa/postsoftmax",
                out["kappa_postsoftmax"].detach().cpu().flatten(),
                global_step=self.current_epoch
            )

        if hasattr(self, "_protos"):
            for exp in ["exp_avg_sq", "exp_avg"]:
                if exp in self.optimizers().state[self._protos].keys():
                    self.logger.experiment.add_histogram(
                        f"protos_optim/{exp}",
                        self.optimizers().state[self._protos][exp].detach().cpu().flatten(),
                        global_step=self.current_epoch
                    )

        for i, (feat, featname) in enumerate(zip(batch.features.T, self.trainer.datamodule.get_feature_names())):  
            self.logger.experiment.add_histogram(
                    f"features/{featname}",
                    feat.detach().cpu().flatten(),
                    global_step=self.current_epoch
                )

        for ixyz, xyz in enumerate(["x", "y", "z"]):
            self.logger.experiment.add_histogram(
                    f"positions/{xyz}",
                    batch.pos[:, ixyz].detach().cpu().flatten(),
                    global_step=self.current_epoch
                )

        for o in self.activated_transformations.keys():
            if o in out.keys() and out[o] is not None:
                if "translation" in o:
                    www = out[o].squeeze()
                    mini = www.min(0)[0]
                    maxi = www.max(0)[0]

                    for ixyz, xyz in enumerate(["x", "y", "z"]):
                        self.logger.experiment.add_histogram(
                            f"{o}/spatial_extend/{xyz}",
                            (maxi - mini)[..., ixyz].detach().cpu().flatten(),
                            global_step=self.current_epoch
                        )

                        if hasattr(self.decoders[o][-1], "bias") and self.decoders[o][-1].bias is not None:
                            self.logger.experiment.add_histogram(
                                f"{o}/ll_bias/{xyz}",
                                self.decoders[o][-1].bias[ixyz::3].detach().cpu().flatten(),
                                global_step=self.current_epoch
                            )

                        self.logger.experiment.add_histogram(
                            f"{o}/ll_weight/{xyz}",
                            self.decoders[o][-1].weight[ixyz::3].detach().cpu().flatten(),
                            global_step=self.current_epoch
                        )

                    
                    
                if out[o][0, 0, 0].numel() == 3:
                    for ixyz, xyz in enumerate(["x", "y", "z"]):
                        self.logger.experiment.add_histogram(
                            f"{o}/{xyz}",
                            out[o][..., ixyz].detach().cpu().flatten(),
                            global_step=self.current_epoch
                        )
                else:
                    self.logger.experiment.add_histogram(
                        f"{o}",
                        out[o].detach().cpu().flatten(),
                        global_step=self.current_epoch
                    )

    @torch.no_grad()        
    def greedy_pcs(self, batch, out, tag):

        self.log(f'NpointsX/{tag}', batch.pos_lenght.float().mean().item(), on_step=False,
                on_epoch=True, batch_size=1)



        pos, rec = batch.pos[batch.batch == 0], out["recs"][0]

        # This is already performing a subsampling (2**13 = 8192 points maximum)
        NMAX_pos = 2**14 if pos.size(0) > 2**14 else pos.size(0) # default 2**13
        NMAX_rec = 2**14 if rec.size(0) > 2**14 else rec.size(0)
        NMAX = max(NMAX_pos, NMAX_rec)
        if pos.size(0) > NMAX:
            pos = pos[np.random.choice(pos.size(0), NMAX, replace=True)]
        elif pos.size(0) < NMAX:
            pos = torch.nn.functional.pad(pos, (0,0,0,NMAX-pos.size(0)), mode='constant', value=0)
        if rec.size(0) > NMAX:
            rec_select = np.random.choice(rec.size(0), NMAX, replace=True)
            rec = rec[rec_select]
        elif rec.size(0) < NMAX:
            rec_select = np.arange(rec.size(0))
            rec = torch.nn.functional.pad(rec, (0,0,0,NMAX-rec.size(0)), mode='constant', value=0)
        else:
            rec_select = np.arange(rec.size(0))
        
        ############
        # WIP mesh implementation
        if self.hparams.protos.name == "mesh":
            rec = torch.zeros_like(pos)
            rec_ori = out["recs"][0]
            rec[:rec_ori.size(0)]=rec_ori
            mesh_protos = self.get_protos_mesh().to(rec.device)
            choice = out["choice"][0][out["choice"][0] != -1]
            mesh_rec = mesh_protos[choice]
            verts_offset = torch.arange(mesh_rec.size(0), device=rec.device)*self.hparams.protos.points
            verts_offset = verts_offset.unsqueeze(1).unsqueeze(2)
            mesh_rec += verts_offset
            mesh_rec = mesh_rec.view(-1,3)
            # Offset to account for in pc_out
            mesh_rec += pos.size(0)
            mesh_out = mesh_rec.unsqueeze(0)
        ############
        
        pc = torch.cat([pos.unsqueeze(0), rec.unsqueeze(0)], 0)
        # subtract mean
        sub_mean = pc.view(-1, 3).mean(0)
        pc -= sub_mean

        mini, maxi = pc.min(1)[0].unsqueeze(1), pc.max(1)[0].unsqueeze(1)
        colors = (255 * (pc - mini) / (maxi - mini + 10e-8)).int()

        config_dict = {
            'material': {
                'cls': 'PointsMaterial',
                'size': 0.3 * (1 + 3*("lidar" in self.hparams.data.name)) # Default 0.05
            },
        }

        colors[1] *= 0
        """
        ##################################
        # Chamfer distance monitoring of the reconstruction
        posV = pos.view(self.hparams.S, -1, 3)
        recV = rec.view(self.hparams.S, -1, 3)
        _, cham_y, _, _ = chamfer_distance(posV, recV)
        cham_y_det = cham_y.detach().cpu()
        for i in range(cham_y_det.shape[0]):
            self.logger.experiment.add_histogram(
                f"recs_Cham_y/logCham_y",
                torch.log(cham_y_det[i]),
                global_step=self.current_epoch
            )
        ##################################
        """
        
        # Add special logging for lidar and masking (self occultation)
        if hasattr(self.hparams, "masking"):
            mask = out["masks"][0]
            # choice selects the slots and protos associated
            choice = out["choice"][0] # shape nb_slots
            rec_mask = torch.zeros_like(out["recs"][0][...,0])
            rec_mask = rec_mask.view(len(out["recs_k"][0]), -1)
            i = 0
            for l,k in enumerate(choice):
                if k != -1:
                    rec_mask[i] = mask[l,k]
                    i+=1
            rec_mask_selected = rec_mask.flatten()[rec_select]
            center = batch.lidar_center[0]
            
            # Change color of masked points
            mask_rgb = torch.tensor([255, 0, 0], dtype=torch.int32, device=colors.device)
            # Account for padding
            rec_mask_selected = torch.nn.functional.pad(rec_mask_selected,
                                                        (0,colors[1].size(0)-rec_mask_selected.size(0)),
                                                        mode='constant',
                                                        value=1)
            colors[1][rec_mask_selected!=1] = mask_rgb
            
            pc_out = pc.view(1, -1, 3)
            colors_out = colors.view(1, -1, 3)
            init_len=pc_out.shape[1]
            # Create 3D grid to materialize the lidar
            x_range = torch.arange(-7, 8, 1, device=pc.device)
            x, y, z = torch.meshgrid(x_range, x_range, x_range, indexing='ij')
            grid = torch.stack((x, y, z), dim=-1).view(1,-1,3)
            size = 0.1
            grid = grid*size
            center_log = center.detach().clone()
            # Apply the same offset as for pc (sub_mean)
            #center_log[...,:2] -= sub_mean[:2]
            center_log -= sub_mean
            grid += center_log
            grid_colors = torch.tensor([255, 165, 0], device=pc.device).repeat(1, len(x_range)**3, 1)
            pc_out = torch.cat((pc_out, grid), dim=1)
            colors_out = torch.cat((colors_out, grid_colors), dim=1)
            
            self.logger.experiment.add_mesh(
                f"pred_{tag}", pc_out, colors=colors_out,
                faces=None, config_dict=config_dict, global_step=self.current_epoch
            )
            if self.hparams.protos.name == "mesh":
                self.logger.experiment.add_mesh(
                    f"pred_mesh_{tag}", pc_out, colors=colors_out,
                    faces=mesh_out, config_dict=config_dict, global_step=self.current_epoch
                )
        else:
            self.logger.experiment.add_mesh(
                f"pred_{tag}", pc.view(1, -1, 3), colors=colors.view(1, -1, 3),
                faces=None, config_dict=config_dict, global_step=self.current_epoch
            )