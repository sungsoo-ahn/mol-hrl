import random
import tokenizers
from tqdm import tqdm
import numpy as np
    
from data.util import load_smiles_list
from data.score.score import _raw_plogp

from tokenizers import Tokenizer
from tokenizers import pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

if __name__ == "__main__":
    # create vocabulary
    smiles_list = []
    for split in ["train", "valid", "test"]:
        smiles_list += load_smiles_list("zinc", split)
    
    for split in ["train", "valid", "test"]:
        smiles_list += load_smiles_list("chembl24", split)
        
    for score_name in ["5ht1b", "5ht2b", "acm2", "cyp2d6"]:
        smiles_list += load_smiles_list(score_name, "default")

    """        
    tokenizer = Tokenizer(BPE())
    tokenizer.pad_token = "[PAD]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        "(\[|\]|Br?|Cl?|Si?|Se?|se?|@@?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])",
        "isolated",
    )
    trainer = BpeTrainer(vocab_size=400, special_tokens=["[PAD]", "[MASK]", "[BOS]", "[EOS]"])
    tokenizer.train_from_iterator(iter(smiles_list), trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")), ("[EOS]", tokenizer.token_to_id("[EOS]")),],
    )
    tokenizer.save(f"../resource/data/tokenizer.json")
    """
    """
    tokenizer = load_tokenizer()
    print(tokenizer.token_to_id("[PAD]"))
    for smiles in tqdm(smiles_list[:10000]):
        try:
            assert smiles == tokenizer.decode(tokenizer.encode(smiles).ids).replace(" ", "")
        except:
            print(smiles)
            print(tokenizer.decode(tokenizer.encode(smiles).ids).replace(" ", ""))
            assert False
    """

    # create dataset for plogp
    smiles_list = load_smiles_list("zinc", "train")
    random.shuffle(smiles_list)
    
    #
    train_smiles_list = smiles_list[:4000]        
    with open("../resource/data/plogp/train.txt", "w") as f:
        for smiles in train_smiles_list:
            f.write(smiles + "\n")
    
    train_score_list = [_raw_plogp(smiles) for smiles in train_smiles_list]
    print(np.mean(train_score_list))
    print(np.std(train_score_list))
    with open("../resource/data/plogp/train_score.txt", "w") as f:
        for score in train_score_list:
            f.write(str(score) + "\n")

    #
    valid_smiles_list = smiles_list[4000:5000]        
    with open("../resource/data/plogp/valid.txt", "w") as f:
        for smiles in valid_smiles_list:
            f.write(smiles + "\n")
    
    valid_score_list = [_raw_plogp(smiles) for smiles in valid_smiles_list]
    print(np.mean(valid_score_list))
    print(np.std(valid_score_list))
    with open("../resource/data/plogp/valid_score.txt", "w") as f:
        for score in valid_score_list:
            f.write(str(score) + "\n")