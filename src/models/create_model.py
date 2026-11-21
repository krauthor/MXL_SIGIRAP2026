import inspect

from src.models.var_inf_model import Model as MXL_Model
from src.models.var_inf_corr_model import MXL_Corr_Model
from src.models.var_inf_corr_shared_model import MXL_Corr_Shared_Model
from src.models.var_inf_sasrec_model import MXL_SASRec_Model
from src.models.var_inf_corr_sasrec_model import MXL_Corr_SASRec_Model
from src.models.var_inf_corr_shared_sasrec_model import MXL_Corr_Shared_SASRec_Model
from src.models.mnl_model import MNL_Model
from src.models.mnl_multi_emb_model import MNL_Multi_Model
from src.models.mnl_max_emb_model import MNL_Max_Model
from src.models.mnl_sasrec_model import MNL_SASRec_Model
from src.models.mnl_multi_sasrec_model import MNL_Multi_SASRec_Model
from src.models.mnl_max_sasrec_model import MNL_Max_SASRec_Model
from src.models.intercept_model import Intercept_Model
from src.models.baseline_models import MostPop_Model, Random_Model


MODEL_REGISTRY = {
    "MXL": MXL_Model,
    "MXL_Corr": MXL_Corr_Model,
    "MXL_Corr_Shared": MXL_Corr_Shared_Model,
    "MXL_SASRec": MXL_SASRec_Model,
    "MXL_Corr_SASRec": MXL_Corr_SASRec_Model,
    "MXL_Corr_Shared_SASRec": MXL_Corr_Shared_SASRec_Model,
    "MNL": MNL_Model,
    "MNL_Multi": MNL_Multi_Model,
    "MNL_Max": MNL_Max_Model,
    "MNL_SASRec": MNL_SASRec_Model,
    "MNL_Multi_SASRec": MNL_Multi_SASRec_Model,
    "MNL_Max_SASRec": MNL_Max_SASRec_Model,
    "Intercept": Intercept_Model,
    "MostPop": MostPop_Model,
    "Random": Random_Model,
}

_LEGACY_HYPERPARAM_KEYS = {
    "sasrec_num_heads": "seq_num_heads",
    "sasrec_num_blocks": "seq_num_blocks",
    "sasrec_dropout": "seq_dropout",
}


def uses_sequence_backbone(model_name):
    cls = MODEL_REGISTRY.get(model_name)
    return getattr(cls, "_force_user_backbone", None) is not None


def create_model(model_name, hyperparams, num_users, num_items, padding_id=0, no_choice_id=1, never_chosen_id=2):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not recognized.")
    model_class = MODEL_REGISTRY[model_name]

    for legacy_key, new_key in _LEGACY_HYPERPARAM_KEYS.items():
        if legacy_key in hyperparams and new_key not in hyperparams:
            hyperparams[new_key] = hyperparams[legacy_key]

    filtered_hyperparams = {
        k: v
        for k, v in hyperparams.items()
        if k in [p.name for p in inspect.signature(model_class.__init__).parameters.values()]
    }

    return model_class(
            num_users=num_users,
            num_items=num_items,
            padding_id=padding_id,
            no_choice_id=no_choice_id,
            never_chosen_id=never_chosen_id,
            **filtered_hyperparams,
        )
