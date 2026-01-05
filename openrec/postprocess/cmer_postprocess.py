import torch
from importlib import import_module
from transformers import PreTrainedTokenizerFast

class CMERLabelDecode(object):
    """
    Decodes model output Token IDs into text.
    Supports dynamic construction via config or direct object injection.
    """
    def __init__(self, 
                 args=None,          # Corresponds to config['args']
                 Tokenizer=None,     # Corresponds to config['Tokenizer']
                 tokenizer=None,     # Direct object injection
                 **kwargs):          # Captures remaining params (e.g., device, epoch_num)
        
        """
        Args:
            args: Dict containing params like 'remove_spaces'.
            Tokenizer: Dict containing tokenizer build info.
            tokenizer: Pre-built tokenizer object.
            **kwargs: Other configurations.
        """
        
        # 1. Handle remove_spaces
        self.remove_spaces = True 
        if args is not None and isinstance(args, dict):
            self.remove_spaces = args.get('remove_spaces', True)

        # 2. Build Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif Tokenizer is not None:
            # Build using Tokenizer dict from config
            self.tokenizer = self._build_tokenizer(Tokenizer)
        else:
            # Fallback: Check kwargs for legacy support or structure mismatch
            tok_cfg = kwargs.get('Tokenizer', {})
            if tok_cfg:
                 self.tokenizer = self._build_tokenizer(tok_cfg)
            else:
                raise ValueError("CMERLabelDecode requires 'Tokenizer' config or 'tokenizer' object.")

    def _build_tokenizer(self, config):
        """
        Dynamically builds Tokenizer based on config.
        """
        # Config expects module, class, and args
        module_path = config.get('module')
        class_name = config.get('class')
        args = config.get('args', {})

        # Fallback: If no module/class, assume direct PreTrainedTokenizerFast params
        if not module_path or not class_name:
             if 'tokenizer_path' in args:
                 return PreTrainedTokenizerFast(
                    tokenizer_file=args['tokenizer_path'],
                    padding_side=args.get('padding_side', 'right'),
                    truncation_side=args.get('truncation_side', 'right'),
                    pad_token=args.get('pad_token'),
                    bos_token=args.get('bos_token'),
                    eos_token=args.get('eos_token'),
                    unk_token=args.get('unk_token')
                 )
             raise ValueError("Tokenizer config must specify 'module' and 'class'")

        try:
            module = import_module(module_path)
            TokenizerClass = getattr(module, class_name)
            # Instantiate Tokenizer Builder
            tokenizer_builder = TokenizerClass(**args)
            
            # Call build method if exists
            if hasattr(tokenizer_builder, 'build'):
                return tokenizer_builder.build()
            return tokenizer_builder
        except Exception as e:
            raise ImportError(f"Failed to build tokenizer from {module_path}.{class_name}: {e}")

    def get_character_num(self):
        """
        Called by Trainer to determine classification layer output dimension (vocab_size).
        """
        if hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, '__len__'):
            return len(self.tokenizer)
        return 0 

    def __call__(self, preds, batch=None, *args, **kwargs):
        """
        Args:
            preds: Tensor (Batch, Seq_Len) or Dict
            batch: Raw batch data
        """
        # Handle tuple/dict inputs from Trainer
        if isinstance(preds, dict):
            if 'cmer_pred' in preds:
                token_ids = preds['cmer_pred']
            elif 'maps' in preds: 
                token_ids = preds['maps']
            else:
                token_ids = next(iter(preds.values())) 
        else:
            token_ids = preds

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu()

        # Batch decode using tokenizer
        decoded_texts = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

        final_texts = []
        for text in decoded_texts:
            if self.remove_spaces:
                text = text.replace(" ", "")
            final_texts.append(text.strip())

        return final_texts