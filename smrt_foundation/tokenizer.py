# tokenizer.py

import yaml

class BaseTokenizer:
    """Static base-pair level tokenizer that reads from config.yaml"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        token_map = config['data']['token_map']
        
        self.token_map = token_map
        self.inverse_token_map = {v: k for k, v in token_map.items()}
        
        self.vocab_size = len(token_map)
        
        self.mask_token_id = token_map['[MASK]']
        self.pad_token_id = token_map['[PAD]']
        self.n_token_id = token_map['N']
        
        self.non_maskable_ids = {
            self.n_token_id,
            self.pad_token_id,
            self.mask_token_id
        }

    def token_to_id(self, token: str) -> int:
        return self.token_map.get(token)

    def id_to_token(self, token_id: int) -> str:
        return self.inverse_token_map.get(token_id)