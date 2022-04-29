from pathlib import Path

import wandb
from pytorch_lightning.loggers.wandb import WandbLogger, _WANDB_GREATER_EQUAL_0_10_22

def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)
    
class CustomWandbLogger(WandbLogger):

    def _scan_and_log_checkpoints(self, checkpoint_callback: "ReferenceType[ModelCheckpoint]") -> None:
        # get checkpoints to be saved with associated score
        checkpoints = {
            checkpoint_callback.last_model_path: checkpoint_callback.current_score,
            checkpoint_callback.best_model_path: checkpoint_callback.best_model_score,
            **checkpoint_callback.best_k_models,
        }
        checkpoints = sorted((Path(p).stat().st_mtime, p, s) for p, s in checkpoints.items() if Path(p).is_file())
        checkpoints = [
            c for c in checkpoints if c[1] not in self._logged_model_time.keys() or self._logged_model_time[c[1]] < c[0]
        ]

        # log iteratively all new checkpoints
        for t, p, s in checkpoints:
            metadata = (
                {
                    "score": s,
                    "original_filename": Path(p).name,
                    "ModelCheckpoint":
                        {
                            k: getattr(checkpoint_callback, k)
                            for k in [
                                "monitor",
                                "mode",
                                "save_last",
                                "save_top_k",
                                "save_weights_only",
                                "_every_n_train_steps",
                                "_every_n_val_epochs",
                            ]
                            # ensure it does not break if `ModelCheckpoint` args change
                            if hasattr(checkpoint_callback, k)
                        },
                } if _WANDB_GREATER_EQUAL_0_10_22 else None
            )
            # -----------change-----------
            artifact_type = 'model'
            if 'group' in self._wandb_init.keys():
                artifact_type = self._wandb_init['group']
            # ----------------------------
            artifact = wandb.Artifact(name=f"model-{self.experiment.id}", type=artifact_type, metadata=metadata)
            artifact.add_file(p, name="model.ckpt")
            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]
            self.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t