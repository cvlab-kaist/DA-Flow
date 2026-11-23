import torch

from .network import DAFlow

MAX_FLOW = 400

_GRU_LAYERS = [f"update_block.gru.conv{g}{i}" for g in "zrq" for i in (1, 2)]


def _strip_module_prefix(state_dict):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def build_daflow(checkpoint, from_raft=False, mixed_precision=False, dpt_channels=256):
    """Build the DA-Flow network.

    Args:
        checkpoint: path to a DA-Flow flow checkpoint (``raft_weights.pt``), or to
            a RAFT checkpoint (e.g. ``raft-things.pth``) when ``from_raft=True``.
        from_raft: initialize the CNN encoders and update block from RAFT for
            training. The GRU input weights for the extra diffusion context
            channels are zero-initialized; DPT heads are randomly initialized.
    """
    model = DAFlow(mixed_precision=mixed_precision, dpt_channels=dpt_channels)
    state_dict = _strip_module_prefix(torch.load(checkpoint, map_location="cpu"))

    if from_raft:
        for key in list(state_dict.keys()):
            if key.endswith("weight") and any(layer in key for layer in _GRU_LAYERS):
                weight = state_dict[key]
                zeros = torch.zeros_like(weight)[:, :128]  # 128 DPT context channels
                state_dict[key] = torch.cat([zeros, weight], dim=1)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert not unexpected, f"Unexpected keys when loading RAFT weights: {unexpected}"
    else:
        model.load_state_dict(state_dict, strict=True)
    return model


def use_batch_statistics(model):
    """Make BatchNorm layers normalize with per-batch statistics at inference.

    DA-Flow is trained and was evaluated in the paper with BatchNorm in training mode
    (batch statistics, here: batches of ``flow_batch_size`` frame pairs). This keeps that
    behavior while the rest of the model is in eval mode, without updating running stats.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """RAFT sequence loss with exponentially increasing weights."""
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    mag = torch.sum(flow_gt ** 2, dim=1).sqrt().unsqueeze(1)
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    metrics = {
        "epe": epe.mean().item(),
        "acc_1px": (epe < 1).float().mean().item(),
        "acc_3px": (epe < 3).float().mean().item(),
        "acc_5px": (epe < 5).float().mean().item(),
    }
    return flow_loss, metrics
