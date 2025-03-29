import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import os
import unicodedata
import time
import argparse
from typing import List, Tuple, Dict, Optional, Union

MAX_SEQ_LENGTH = 20
EMBED_DIM = 512
NUM_HEADS = 16
NUM_ENCODER_LAYERS = 8
NUM_DECODER_LAYERS = 8
FFN_DIM = 2048
DROPOUT = 0.1
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SpellingTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=EMBED_DIM, nhead=NUM_HEADS, 
                 num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS, 
                 dim_feedforward=FFN_DIM, dropout=DROPOUT):
        super(SpellingTransformer, self).__init__()
        
        self.d_model = d_model
        
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        self.transformer = Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        
        src = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        output = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        output = self.output_layer(output)
        
        return output.transpose(0, 1)

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)

class AzerbaijaniSpellCorrector:
    """
    A class for correcting spelling errors in Azerbaijani text using a Transformer model.
    
    Attributes:
        model (SpellingTransformer): The Transformer model for spell correction
        char_to_idx (Dict[str, int]): Mapping from characters to indices
        idx_to_char (Dict[int, str]): Mapping from indices to characters
        device (torch.device): Device to run the model on
    """
    
    def __init__(self, checkpoint_path: str, device: Optional[torch.device] = None):
        """
        Initialize the spell corrector with a pre-trained model.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint
            device (Optional[torch.device]): Device to run the model on. Defaults to GPU if available, else CPU.
        
        Raises:
            FileNotFoundError: If the checkpoint file does not exist
            ValueError: If the checkpoint does not contain required data
        """
        self.device = device if device is not None else DEFAULT_DEVICE
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'char_to_idx' in checkpoint and 'idx_to_char' in checkpoint:
            self.char_to_idx = checkpoint['char_to_idx']
            self.idx_to_char = checkpoint['idx_to_char']
        else:
            raise ValueError("Checkpoint does not contain vocabulary information.")
        
        vocab_size = len(self.char_to_idx)
        self.model = SpellingTransformer(vocab_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Run a warmup inference to initialize the model
        self._warmup()
    
    def _warmup(self):
        """Run a simple inference to warm up the model."""
        with torch.no_grad():
            self.correct("test", return_time=False)
    
    def correct(self, word: str, return_time: bool = False) -> Union[str, Tuple[str, float]]:
        """
        Correct the spelling of a single Azerbaijani word.
        
        Args:
            word (str): The word to correct
            return_time (bool): Whether to return the execution time
            
        Returns:
            Union[str, Tuple[str, float]]: 
                If return_time is False: The corrected word
                If return_time is True: Tuple of (corrected_word, execution_time_ms)
        """
        self.model.eval()
        
        start_time = time.time()
        
        word = unicodedata.normalize('NFC', word)
        
        word_ids = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in word[:MAX_SEQ_LENGTH]]
        
        src = torch.tensor([self.char_to_idx['<BOS>']] + word_ids + [self.char_to_idx['<EOS>']], 
                          dtype=torch.long).unsqueeze(0).to(self.device)
        
        src_mask = generate_square_subsequent_mask(src.size(1), self.device)
        
        decoder_input = torch.ones(1, 1, dtype=torch.long, device=self.device) * self.char_to_idx['<BOS>']
        
        with torch.no_grad():
            for i in range(MAX_SEQ_LENGTH + 2 - 1):
                tgt_mask = generate_square_subsequent_mask(decoder_input.size(1), self.device)
                
                output = self.model(
                    src=src,
                    tgt=decoder_input,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask,
                    src_padding_mask=None,
                    tgt_padding_mask=None
                )
                
                next_token_logits = output[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=1).unsqueeze(1)
                
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
                
                if next_token.item() == self.char_to_idx['<EOS>']:
                    break
        
        corrected_word = ''.join(self.idx_to_char[idx] for idx in decoder_input[0, 1:-1].cpu().tolist())
        
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        
        if return_time:
            return corrected_word, execution_time_ms
        return corrected_word

    def correct_batch(self, words: List[str], return_time: bool = False) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Correct the spelling of a batch of Azerbaijani words.
        
        Args:
            words (List[str]): List of words to correct
            return_time (bool): Whether to return execution times
            
        Returns:
            Union[List[str], Tuple[List[str], List[float]]]:
                If return_time is False: List of corrected words
                If return_time is True: Tuple of (list_of_corrected_words, list_of_execution_times_ms)
        """
        corrected_words = []
        execution_times = []
        
        for word in words:
            result = self.correct(word, return_time=True)
            corrected_words.append(result[0])
            execution_times.append(result[1])
        
        if return_time:
            return corrected_words, execution_times
        return corrected_words

def interactive_mode(corrector):
    """Run an interactive spell correction session."""
    print("\n===== Spelling Correction Tool =====")
    print("Enter words to correct (type 'exit' or 'quit' to end):")
    
    total_time = 0
    word_count = 0
    max_time = 0
    min_time = float('inf')
    
    while True:
        user_input = input("\nEnter misspelled word: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        if not user_input:
            continue
        
        try:
            corrected, exec_time_ms = corrector.correct(user_input, return_time=True)
            
            total_time += exec_time_ms
            word_count += 1
            max_time = max(max_time, exec_time_ms)
            min_time = min(min_time, exec_time_ms)
            
            print(f"Original:  {user_input}")
            print(f"Corrected: {corrected}")
            print(f"Execution time: {exec_time_ms:.2f} ms")
            
            if user_input != corrected:
                print(f"Changes made: {user_input} → {corrected}")
            else:
                print("No changes needed.")
                
        except Exception as e:
            print(f"Error processing input: {str(e)}")
    
    if word_count > 0:
        print("\n===== Performance Statistics =====")
        print(f"Total words processed: {word_count}")
        print(f"Average execution time: {total_time / word_count:.2f} ms")
        print(f"Minimum execution time: {min_time:.2f} ms")
        print(f"Maximum execution time: {max_time:.2f} ms")
    
    print("Exiting spelling correction tool.")

def main():
    parser = argparse.ArgumentParser(description='Azerbaijani Spelling Correction Tool')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_model/best_model.pt',
                        help='Path to the model checkpoint')
    parser.add_argument('--words', nargs='+', help='Words to correct (separated by spaces)')
    parser.add_argument('--batch', action='store_true', help='Process multiple words in batch mode')
    args = parser.parse_args()
    
    try:
        corrector = AzerbaijaniSpellCorrector(checkpoint_path=args.checkpoint)
        
        print(f"Model loaded successfully (vocabulary size: {len(corrector.char_to_idx)})")
        print(f"Using device: {corrector.device}")
        
        if args.words:
            if args.batch:
                corrected_words, times = corrector.correct_batch(args.words, return_time=True)
                for original, corrected, exec_time in zip(args.words, corrected_words, times):
                    print(f"Original: {original}")
                    print(f"Corrected: {corrected}")
                    print(f"Execution time: {exec_time:.2f} ms")
                    if original != corrected:
                        print(f"Changes made: {original} → {corrected}")
                    else:
                        print("No changes needed.")
                    print()
            else:
                for word in args.words:
                    corrected, exec_time = corrector.correct(word, return_time=True)
                    print(f"Original: {word}")
                    print(f"Corrected: {corrected}")
                    print(f"Execution time: {exec_time:.2f} ms")
                    if word != corrected:
                        print(f"Changes made: {word} → {corrected}")
                    else:
                        print("No changes needed.")
                    print()
        else:
            interactive_mode(corrector)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()