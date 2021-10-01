
def optimize(model, optimizers, args, **kwargs):
    for e in model.models:
        e.optimize()
