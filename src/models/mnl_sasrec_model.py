from src.models.mnl_model import MNL_Model


class MNL_SASRec_Model(MNL_Model):
    _force_user_backbone = "sasrec"
