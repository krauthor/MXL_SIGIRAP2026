from src.models.var_inf_corr_model import MXL_Corr_Model


class MXL_Corr_SASRec_Model(MXL_Corr_Model):
    _force_user_backbone = "sasrec"
