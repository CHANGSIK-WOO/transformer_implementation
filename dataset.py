import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len)->None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64) # [2]
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

        # tokenizer_src.token_to_id("[SOS]") : [SOS] 토큰이 vocab에서 어떤 정수 ID인지 찾아냄
        # [ ... ]로 감싼 이유: 텐서로 만들 때 1차원 벡터로 만들기 위해
        # torch.tensor(..., dtype=torch.int64) : 모델 입력으로 쓰기 위해 정수형 텐서로 변환
        # torch.Tensor(...) → 잘못된 생성 방식 (shape 기반)
        # torch.tensor(...) → 정확한 생성 방식 (값 기반)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index): # dataset[0] : 네가 정의한 __getitem__(self, index) 함수가 자동으로 호출돼서, index=0에 해당하는 데이터 샘플을 리턴
        src_target_pair = self.ds[index]
        # self.ds[0] →
        # {
        #     "translation": {
        #         "en": "I am a student.",
        #         "it": "Sono uno studente."
        #     }
        # }

        src_text = src_target_pair["translation"][self.src_lang] # "I am a student."
        tgt_text = src_target_pair["translation"][self.tgt_lang] # "Sono uno studente."

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # [4, 5, 6, 7]
        # self.tokenizer_src.encode(src_text).tokens --> ["I", "am", "a", "student"]
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # 2개는 [SOS], [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # 1개는 [EOS]

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # source 문장 → 모델 입력용 텐서 (encoder_input)로 만드는 과정
        encoder_input = torch.cat(
            [
                self.sos_token, # tensor([1])
                torch.tensor([enc_input_tokens], dtype = torch.int64), # real input sentence ID tensor([4, 5, 6, 7])
                self.eos_token, # tensor([2])
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64) # tensor([0, 0, 0])
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor([dec_input_tokens], dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64) # tensor([0, 0, 0])
            ]
        )

        label = torch.cat(
            [
                torch.tensor([dec_input_tokens], dtype = torch.int64),
                self.eos_token,  # tensor([2])
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)  # tensor([0, 0, 0])
            ]
        )

        assert encoder_input.shape[0] == self.seq_len
        assert decoder_input.shape[0] == self.seq_len
        assert label.shape[0] == self.seq_len

        return {"encoder_input" : encoder_input,
                "decoder_input" : decoder_input,
                "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
                "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size[0]), # (1, 1, seq_len) : will be broadcasting & (1, seq_len, seq_len)
                "label" : label,
                "src_text" : src_text,
                "tgt_text" : tgt_text} # all shape : (seq_len)
        # 만약 그냥 전달하면 순서에 의존해서 불러와야 하기에, 접근성 및 코드 가독성을 위해 딕셔너리로 return

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int) #주대각선 위는 0, 나머지는 1인 대상
    return mask == 0 # if 0 then True, and False if not 0.



