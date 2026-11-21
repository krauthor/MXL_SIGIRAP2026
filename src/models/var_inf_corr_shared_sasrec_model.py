from src.models.var_inf_corr_shared_model import MXL_Corr_Shared_Model


class MXL_Corr_Shared_SASRec_Model(MXL_Corr_Shared_Model):
    _force_user_backbone = "sasrec"
