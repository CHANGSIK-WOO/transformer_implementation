#torch
import torch
import torch.nn as nn

#dataset
from dataset import BilingualDataset, causal_mask

#model
from model import build_transformer

#config
from config import get_config, get_weights_file_path

#tokenizers from Huggingface
from datasets import load_dataset #Hugging Face의 datasets 라이브러리에서 학습용 데이터셋을 불러오기 위해 사용
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

#Class Path
from pathlib import Path

#random split
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader # DataLoader: Dataset에서 샘플을 자동으로 배치로 묶어주는 도구

#tensorboard
from torch.utils.tensorboard import SummaryWriter

#tqdm
from tqdm import tqdm
import warnings


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang] #리스트가 아니라 "하나씩" 문장을 "제너레이터(generator)"로 반환. yield = 하나씩 값을 내보내고, 다음 루프 때 이어서 실행됨
        # result : "I am a student."

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang)) # tokenizer_file은 json 파일로 저장됨
    if not Path.exists(tokenizer_path): #build
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) #unk_token은 무조건 정의를 해줘야 함.
        tokenizer.pre_tokenizer = Whitespace() # 문장이 들어왔을 때 분리하는 pre_tokenizer (전처리기) 정의
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        # trainer : vocab을 어떻게 만들지 결정하는 학습 설정자(trainer), special_tokens : 무조건 추가되어야 하는 token
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        #get_all_sentences(ds, lang)는 데이터셋에서 문장들만 추출해서 리스트(iterator)로 반환하는 함수
        #그 문장들을 기반으로 vocab 학습
        tokenizer.save(str(tokenizer_path))

    else: #get
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

def get_dataset(config):
    # "train" split만 포함된 번역쌍 리스트
    ds_raw = load_dataset(path = "opus_books",
                          name = f'{0}-{1}'.format(config["lang_src"], config["lang_tgt"]),
                          split = "train")
    # example : ds_raw = load_dataset("opus_books", "en-fr")

    # build tokenizer.
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"]) #source 언어 ("en" 등)의 토크나이저, ds_raw 기준으로 학습됨
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"]) #arget 언어 ("it" 등)의 토크나이저, ds_raw 기준으로 학습됨

    # train split 중 90%만 학습, 10%는 validation에 사용 (검증)
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, lengths=[train_ds_size, val_ds_size]) #ds_raw에 대해 각각 해당 사이즈만큼 random_split 진행

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    max_len_src, max_len_tgt = 0, 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"max length of src sentence :{max_len_src}")
    print(f"max length of tgt sentence :{max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"])
    return model

def train_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device : {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # get_vocab_size() : Tokenizer 객체는 내부에 WordLevel 모델을 포함하고 있고,이 모델의 vocab 크기를 쉽게 알 수 있도록 get_vocab_size() 메서드를 제공.
    # vocab에 등록된 전체 토큰 개수 (스페셜 토큰 포함)

    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config["preload"]: # 저장된 모델 체크포인트를 불러와서 이어서 학습하겠다는 조건
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"preloading model : {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

        # state = {
        #     "epoch": 7,  # int, 마지막으로 학습한 epoch (이후 이어서 학습할 때 필요)
        #
        #     "model_state_dict": model.state_dict(),
        #     # dict[str, tensor], 모델의 파라미터 전체 (가중치 + 편향 등)
        #
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     # dict, 옵티마이저 내부 상태 (momentum, step, learning rate 등)
        #
        #     "global_step": 1823
        #     # int, 전체 학습 도중 진행된 미니배치 수 (TensorBoard logging 등에 사용)
        # }

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Processing Epoch {epoch:02d}") # 1 -> 01, 2 -> 02
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device) # (Batch, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (Batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (Batch, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)  # (Batch, 1, seq_len, seq_len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (Batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (Batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (Batch, seq_len, tgt_vocab_size)

            label = batch["label"].to(device) # (Batch, seq_len)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()),label.view(-1))
            batch_iterator.set_postfix({f"loss" : f"{loss.item():6.3f}"}) # 진행 바 옆 정보와 관련. Processing Epoch 01:  42%|█████       | 34/80 [00:07<00:10,  4.28it/s, loss= 0.823]

            #Log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate
            loss.backward()

            #update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch" : epoch,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "global_step" : global_step
            }, model_filename
        )

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)













