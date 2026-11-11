"""Small public adapter matching the group workflow, without evaluator imports."""
import json
from pathlib import Path

import torch
from independent_tpp.protocol import sha256, write_json
from .experiment import load_model, train
from .model import anchor_sample


class ModelAdapter:
    def __init__(self, data_root, run_directory, fold, device='cpu'):
        self.data_root = Path(data_root)
        self.run = Path(run_directory)
        self.fold, self.device = fold, device
        self.model = None

    def fit(self, train_manifest, val_manifest, config, seed):
        # Refuse arbitrary or test manifests; train() always uses the locked loader.
        for split,path in [('train',train_manifest),('val',val_manifest)]:
            expected = self.data_root/self.fold/f'{split}_manifest.csv'
            if sha256(path) != sha256(expected):
                raise ValueError(f'{split} manifest is not the configured fold manifest')
        train(self.data_root,self.run,self.fold,seed,config,self.device)
        self.select_checkpoint(val_manifest)
        return self

    def select_checkpoint(self, val_manifest):
        config = json.loads((self.run/'config.json').read_text())
        if sha256(val_manifest) != config['val_identity']['manifest_hash']:
            raise ValueError('Checkpoint selection is bound to the validation manifest')
        self.model,_,checkpoint = load_model(self.run,self.device)
        return checkpoint

    @torch.no_grad()
    def predict(self, observation_batch, num_samples, seed):
        if self.model is None:
            self.model,_,_ = load_model(self.run,self.device)
        positions = observation_batch['positions'].to(self.device)
        mask = observation_batch['mask'].to(self.device)
        origins = observation_batch['origin'].to(self.device)
        seeds = [int(seed)+int(i)*1009 for i in observation_batch['indices']]
        samples = anchor_sample(self.model(positions,mask),seeds,num_samples)
        return (samples+origins[:,None,None]).cpu()

    def save_provenance(self, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True,exist_ok=True)
        config = json.loads((self.run/'config.json').read_text())
        selection = json.loads((self.run/'checkpoint_selection.json').read_text())
        checkpoint = self.run/selection['selected']['checkpoint']
        write_json(output/'model_provenance.json',dict(model='DSPIGCN',config=config,
                   selection=selection,checkpoint_sha256=sha256(checkpoint)))
