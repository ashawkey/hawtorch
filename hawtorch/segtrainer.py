from . trainer import *
from . vision import plot_images

import skimage.transform as skt


class SegTrainer3D(Trainer):

    def resize(self, inputs, new_shape, interpolation=1):
        B, C, H, W, D = inputs.shape
        inputs = inputs.reshape(-1, H, W, D).transpose(1,2,3,0)
        inputs = skt.resize(inputs, new_shape, order=interpolation, mode='constant', cval=0, clip=True, anti_aliasing=False)
        inputs = inputs.transpose(3,0,1,2).reshape(B, C, new_shape[0], new_shape[1], new_shape[2])
        return inputs

    def eval_step(self, data):
        """
        inputs: [B, C, H, W]
        truths: [B, H, W]
        """
        inputs, masks = data["image"].numpy(), data["mask"].numpy()
        
        if self.conf.eval_type == 'whole':
            B, C, H, W, D = inputs.shape
            resized_inputs = self.resize(inputs, self.conf.patch_size, 1)

            resized_inputs = torch.from_numpy(resized_inputs).float().to(self.device)
            preds = self.model(resized_inputs) 
            preds = preds.detach().cpu().numpy()
            
            preds = self.resize(preds, (H, W, D), 0) 
            preds = preds.argmax(1)

        elif self.conf.eval_type == 'sliding_window':
            B, C, H, W, D = inputs.shape
            # TODO: pad to at least conf.patch_size and return slicer
            preds = np.zeros((B, self.conf.num_classes, H, W, D))
            preds_cnt = np.zeros((1, 1, H, W, D))
            # slide window
            ph, pw, pd = self.conf.patch_size
            sh, sw, sd = self.conf.patch_stride
            for h in range(0, H, sh):
                for w in range(0, W, sw):
                    for d in range(0, D, sd):
                        hh = min(h + ph, H)
                        ww = min(w + pw, W)
                        dd = min(d + pd, D)
                        h = hh - ph
                        w = ww - pw
                        d = dd - pd

                        patch = inputs[:, :, h:hh, w:ww, d:dd] # [B, C, ph, pw, pd]
                        patch = torch.from_numpy(patch).float().to(self.device)
                        pred = self.model(patch)
                        pred = pred.detach().cpu().numpy()

                        preds[:, :, h:hh, w:ww, d:dd] += pred
                        preds_cnt[0, 0, h:hh, w:ww, d:dd] += 1

            preds /= preds_cnt
            preds = preds.argmax(1) # [B, H, W, D]


        else:
            raise NotImplementedError

        
        return preds, masks
        

    def evaluate(self, 
                 eval_set=None,
                 save_snap=True,
                 save_image=False,
                 save_image_folder=None,
                 show_image=False,
                 ):

        """
        final evaluate at the best epoch.
        """
        eval_set = self.eval_set if eval_set is None else eval_set
        self.log.info(f"Evaluate at the best epoch on {eval_set} set...")

        # load model
        model_name = type(self.model).__name__
        ckpt_path = os.path.join(self.workspace_path, 'checkpoints')
        best_path = f"{ckpt_path}/{model_name}_best.pth.tar"
        if not os.path.exists(best_path):
            self.log.error(f"Best checkpoint not found at {best_path}, load by default.")
            self.load_checkpoint()
        else:
            self.load_checkpoint(best_path)

        # turn off logging to tensorboardX
        self.use_tensorboardX = False
        self.evaluate_one_epoch(eval_set, save_snap, save_image, save_image_folder, show_image)

    def evaluate_one_epoch(self, 
                           eval_set,
                           save_snap=False,
                           save_image=False,
                           save_image_folder=None,
                           show_image=False,
                           ):
        self.log.log(f"++> Evaluate at epoch {self.epoch} ...")

        for metric in self.metrics:
            metric.clear()

        self.model.eval()

        pbar = self.dataloaders[eval_set]
        if self.use_tqdm:
            pbar = tqdm.tqdm(pbar)

        epoch_start_time = self.get_time()

        if save_image:
            if save_image_folder is None:
                save_image_folder = 'evaluation_' + self.time_stamp
            os.makedirs(os.path.join(self.workspace_path, save_image_folder), exist_ok=True)

        with torch.no_grad():
            self.local_step = 0
            start_time = self.get_time()
            
            for data in pbar:    
                self.local_step += 1
                                
                if self.max_eval_step is not None and self.local_step > self.max_eval_step:
                    break

                preds, truths = self.eval_step(data)

                for metric in self.metrics:
                    metric.update(preds, truths)
                
                if show_image:
                    batch_size = preds.shape[0]
                    f = plt.figure()
                    ax0 = f.add_subplot(121)
                    ax1 = f.add_subplot(122)
                    for batch in range(batch_size):
                        ax0.imshow(preds[batch])
                        ax1.imshow(truths[batch])
                        plt.show()
                
                if save_image:
                    batch_size = preds.shape[0]
                    for batch in range(batch_size):
                        if 'name' in data:
                            name = data['name'][batch] + '.npy'
                        else:
                            name = str(self.local_step) + '_' + str(batch) + '.npy'
                        
                        np.save(os.path.join(self.workspace_path, save_image_folder, name), preds[batch])
                        self.log.info(f"Saved image {name} at {save_image_folder}.")
                    
            
            total_time = self.get_time() - start_time
            self.log.log1(f"total_time={total_time:.2f}")
            
            self.stats["EvalResults"].append(self.metrics[0].measure())

            if save_snap and self.use_tensorboardX:
                # only save first batch first layer
                self.writer.add_image("evaluate/image", data["image"][0, :, :, 0], self.epoch)
                self.writer.add_image("evaluate/pred", np.expand_dims(preds[0, :, :, 0], 0), self.epoch)
                self.writer.add_image("evaluate/mask", np.expand_dims(truths[0, :, :, 0], 0), self.epoch)

            for metric in self.metrics:
                self.log.log1(metric.report())
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        epoch_end_time = self.get_time()
        self.log.log(f"++> Evaluate Finished. time={epoch_end_time-epoch_start_time:.4f}")