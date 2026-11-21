from src.models.mnl_multi_emb_model import MNL_Multi_Model


class MNL_Multi_SASRec_Model(MNL_Multi_Model):
    _force_user_backbone = "sasrec"
