import os
from pathlib import Path
from omegaconf import OmegaConf
from nemo.collections.nlp.models.dialogue.dialogue_zero_shot_slot_filling_model import DialogueZeroShotSlotFillingModel


def load_model():
    self_path = Path(__file__).resolve()
    nemo_path = self_path.parent.parent.parent.resolve()
    print(nemo_path)
    cfg = OmegaConf.load(Path(nemo_path)/"examples/nlp/dialogue/conf/dialogue_config.yaml")
    cfg.model.dataset.data_dir = os.path.expanduser('~/datasets/assistant/with_entity')
    cfg.model.dataset.dialogues_example_dir = os.path.expanduser('~/datasets/assistant/with_entity_prediction')
    cfg.model.dataset.task = 'zero_shot_slot_filling'
    cfg.model.language_model.pretrained_model_name = 'bert-base-uncased'
    cfg.model.nemo_path = os.path.expanduser('~/models/nemo_experiments/assistant/assistant_0.5_0.5_epoch_10_lr_0.00001-1.nemo')
    cfg.model.bio_slot_loss_weight = 0.5
    cfg.model.optim.lr = 0.00001
    cfg.exp_manager.create_wandb_logger = False
    return DialogueZeroShotSlotFillingModel.restore_from(cfg.model.nemo_path, override_config_path=cfg.model)