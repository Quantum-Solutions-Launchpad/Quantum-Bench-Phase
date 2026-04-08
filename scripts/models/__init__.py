from models import haldane, hubbard, haldane_hubbard

MODELS = {
    haldane.NAME: haldane,
    hubbard.NAME: hubbard,
    haldane_hubbard.NAME: haldane_hubbard,
}

def get_model(name):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS.keys())}")
    return MODELS[name]
