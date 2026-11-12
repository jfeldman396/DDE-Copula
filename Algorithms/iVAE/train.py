"""Portable version of the saved 10/1 iVAE protocol; training is opt-in."""
from pathlib import Path
import argparse
import hashlib
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true', help='Explicitly run the published training budget.')
    parser.add_argument('--data', type=Path, help='External original survey CSV; never bundled here.')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root/'Scripts/Publication/config/ivae.json').read_text())
    if not args.train:
        print(json.dumps({'training_started': False, 'protocol': config}, indent=2))
        return
    if args.data is None or args.output_dir is None:
        parser.error('--train requires --data and --output-dir')
    out = args.output_dir.resolve()
    if out == root or root in out.parents:
        parser.error('Training outputs must be outside the immutable handoff package')
    if out.exists() and any(out.iterdir()):
        parser.error('Use a new or empty output directory')
    expected = json.loads((root/'Results/Publication/data/ivae/RUN_MANIFEST.json').read_text())['data_sha256']
    actual = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('Input hash differs from the archived survey input; this command preserves the paper protocol.')
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(Path(__file__).resolve().parent/'vendor'))
    from models import nets
    from models.wrappers import IVAE_wrapper

    # Exact Gaussian broadcasting patches used by the saved notebook run.
    def normal_sample(self, mu, v):
        eps = torch.randn_like(mu)
        if v.dim() == 1:
            v = v.unsqueeze(1)
        if v.size() != mu.size():
            v = v.expand_as(mu)
        return eps.mul(v.sqrt()).add(mu)

    def normal_log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        if mu.dim() < x.dim():
            mu = mu.view(*mu.shape, *([1] * (x.dim() - mu.dim())))
        if v.dim() < x.dim():
            v = v.view(*v.shape, *([1] * (x.dim() - v.dim())))
        if v.dim() == 1 and mu.dim() == 2:
            v = v.unsqueeze(1)
        if mu.size() != x.size():
            mu = mu.expand_as(x)
        if v.size() != mu.size():
            v = v.expand_as(mu)
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        return lpdf.sum(dim=-1) if reduce else lpdf

    nets.Normal.sample = normal_sample
    nets.Normal.log_pdf = normal_log_pdf
    frame = pd.read_csv(args.data)
    columns = np.setdiff1d(np.arange(2, 102), [45, 52])
    X = frame.iloc[:, columns].to_numpy(dtype=np.float32)
    means = np.nanmean(X, axis=0)
    missing = np.where(np.isnan(X))
    X[missing] = np.take(means, missing[1])
    U = frame[['ideo5']].to_numpy(dtype=np.float32)
    assert X.shape == (1500, 98) and U.shape == (1500, 1)
    assert np.isfinite(X).all() and np.isfinite(U).all()
    _, model, _, logger = IVAE_wrapper(
        X=X, U=U, S=None, batch_size=256, max_iter=2000, seed=0,
        n_layers=2, hidden_dim=[10], lr=1e-2, cuda=False,
        activation='lrelu', slope=.1, discrete=False, inference_dim=1,
        encoder_n_layers=2, encoder_hidden_dim=[10],
        decoder_n_layers=2, decoder_hidden_dim=[10],
        prior_n_layers=1, prior_hidden_dim=[], anneal=False, scheduler_tol=3)
    out.mkdir(parents=True, exist_ok=True)
    for label, net in [('encoder', model.g), ('decoder', model.f)]:
        layers = [layer for layer in net.fc if isinstance(layer, nn.Linear)]
        assert len(layers) == 2
        for i, layer in enumerate(layers, 1):
            pd.DataFrame(layer.weight.detach().cpu().numpy()).to_csv(out/f'{label}_W{i}.csv', index=False)
            pd.DataFrame({f'{label}_b{i}': layer.bias.detach().cpu().numpy()}).to_csv(out/f'{label}_b{i}.csv', index=False)
    loss = np.asarray(logger.log_dict['elbo'])
    np.save(out/'training_elbo.npy', loss)
    (out/'RUN.json').write_text(json.dumps(dict(config, data_sha256=actual, torch=torch.__version__, numpy=np.__version__, finite_loss=bool(np.isfinite(loss).all())), indent=2)+'\n')


if __name__ == '__main__':
    main()
