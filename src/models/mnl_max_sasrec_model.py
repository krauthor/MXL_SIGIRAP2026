from src.models.mnl_max_emb_model import MNL_Max_Model


class MNL_Max_SASRec_Model(MNL_Max_Model):
    _force_user_backbone = "sasrec"
