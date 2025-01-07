import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json, math
from typing import Tuple, Optional, Union

device = torch.device('cuda:0')


def get_uniform_ball_noise(input_shape, radius=0.1):
    uniform_noise_ball = torch.randn(input_shape, device=device)  # normal distribution
    uniform_noise_sphere = torch.nn.functional.normalize(uniform_noise_ball, dim=1)
    u = torch.rand(input_shape[0], device=device)  # unified distribution
    u = u ** (1. / input_shape[1])
    uniform_noise_ball = (uniform_noise_sphere.T * u * radius).T
    return uniform_noise_ball


def noise_injection(x, variance=0.001, modality_offset=None, uniform_noise=False, dont_norm=False):
    if variance == 0.0:
        return x
    std = math.sqrt(variance)
    if not dont_norm:
        x = torch.nn.functional.normalize(x, dim=1)
    if uniform_noise:
        x = x + get_uniform_ball_noise(x.shape, radius=std)
    else:
        x = x + (torch.randn(x.shape, device=device) * std)  # todo by some conventions multivraiance noise should be devided by sqrt of dim
    if modality_offset is not None:
        x = x + modality_offset
    return torch.nn.functional.normalize(x, dim=1)


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class ClipCocoDataset(Dataset):
    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        #这一行代码创建了一个布尔掩码 mask，其中 True 表示文本序列中的有效元素（非填充部分），False 表示填充部分。它通过检查每个元素是否大于等于零来创建掩码。mask大于0的地方是true，小于零的地方是false
        tokens[~mask] = 0
        #false的地方 = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        #最后，这一行代码在掩码 mask 的前面添加了一个长度为 self.prefix_length 的前缀掩码，以将前缀部分标记为 1（有效）。
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        #获得截断或填充后的tokens
        prefix = self.prefixes[item]
        #这一行代码从数据集的 self.prefixes 属性中选择了一个前缀。self.caption2embedding[item] 用于获取当前样本对应的前缀索引
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
            #标准化前缀

        return tokens, mask, prefix

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False, use_image_embedding_as_clipcap=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        #这一行代码创建了一个 GPT-2 模型的分词器（tokenizer）。分词器用于将文本数据分解成标记（tokens），以便后续处理。
        self.prefix_length = prefix_length
        #这一行代码将构造函数的参数 prefix_length 存储为数据集对象的属性，以便在后续的方法中使用。
        self.normalize_prefix = normalize_prefix
        print("data_path is", data_path)

        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
            #这一行代码使用 Python 的 pickle 模块从文件中加载数据，数据被存储在 all_data 变量中。这个数据包含了数据集的各种信息，如前缀、图像 ID 和原始标题。
        print("Data size is %0d" % len(all_data["clip_embedding_text_dave"]))
        sys.stdout.flush()

        self.prefixes = all_data["clip_embedding_text_dave"]

        #captions_raw = all_data["captions"]

        self.captions = all_data["captions"]
        #这一行代码创建了一个列表 self.captions，其中包含了每个样本的原始标题文本,self.captions源自于all_data的captions
        
        print("i am creating")
        self.captions_tokens = []
        self.caption2embedding = []#初始化一个列表来存储标题到嵌入的映射。
        max_seq_len = 0
        iterations = 0
        for caption in self.captions: #caption is text
            #self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption["step"]), dtype=torch.int64)) #text
            if "triplets" in caption:
                encoded_caption = self.tokenizer.encode(caption["triplets"]) + [self.tokenizer.eos_token_id]
            if "phases" in caption:
                encoded_caption = self.tokenizer.encode(caption["phases"]) + [self.tokenizer.eos_token_id]
            self.captions_tokens.append(torch.tensor(encoded_caption, dtype=torch.int64))

            #self.caption2embedding.append(caption["clip_embedding"])
            max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
                #行代码的作用是将字典 caption 中名为 "clip_embedding" 的键所对应的值添加到列表 self.caption2embedding 中。
            iterations = iterations + 1

            # self.max_seq_len = max_seq_len
        with open(f"{data_path[:-4]}_tokens_text.pkl", 'wb') as f:
            pickle.dump([self.captions_tokens, max_seq_len], f)
                #这一行代码使用 Pickle 将处理后的数据保存到文件中。具体来说，它将一个包含三个元素的列表（标题文本张量、标题到嵌入的映射和 max_seq_len）写入了文件。
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))




class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def train(dataset: ClipCocoDataset, model: ClipCaptionModel, args, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    if args.add_modality_offset:
        with open('others/CLIP_embeddings_centers_info.pkl', 'rb') as f:
            modality_offset = pickle.load(f)['offset_to_add_in_training'].to(device)
    else:
        modality_offset = None
    loss_per_epoch_train = []
    loss_per_epoch_val = []
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch} / {epochs}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        accumulated_loss = 0.0
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            prefix = noise_injection(prefix, args.noise_variance, modality_offset=modality_offset, uniform_noise=args.uniform_noise, dont_norm=args.dont_norm)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_value = loss.item()
            progress.set_postfix({"loss": loss_value})
            progress.update()
            accumulated_loss += loss_value
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        loss_per_epoch_train.append(accumulated_loss / len(train_dataloader))
        print('loss_per_epoch_train: ', loss_per_epoch_train)
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
        if args.val_pt:
            val_dataset = ClipCocoDataset(args.val_pt, args.prefix_length, normalize_prefix=not args.dont_norm,
                            use_image_embedding_as_clipcap=args.use_image_embedding_as_clipcap)

            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                torch.cuda.empty_cache()
                for idx, (tokens, mask, prefix) in enumerate(val_dataloader):
                    tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                    outputs = model(tokens, prefix, mask)
                    logits = outputs.logits[:, dataset.prefix_length - 1: -1]
                    loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                    val_loss += loss.item()
            model.train()
            loss_per_epoch_val.append(val_loss / len(val_dataloader))
            print('loss_per_epoch_val: ', loss_per_epoch_val)
        with open(os.path.join(output_dir, f"loss_per_epoch.json"), 'w') as f:
            json.dump({'train': loss_per_epoch_train, 'val': loss_per_epoch_val}, f)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/ubuntu/meddataset/cholec_process/phase_train/embedding/V2/nips3/phase_text_train.pkl', help='path to clip embeddings of captions generated by the attached embeddings_generator script')
    parser.add_argument('--val_pt', default='', help='path to clip embeddings of validations set')
    parser.add_argument('--pretrain_weights', default='', help='path to pretrained weights, if not specified, will train from scratch')
    parser.add_argument('--out_dir', default='/home/ubuntu/meddataset/cholec_process/model_combine/v2/noise0text', help='path to output directory')
    parser.add_argument('--add_modality_offset', dest='add_modality_offset', action='store_true', default=False, help='train with modality offset that was pre calculated at others/CLIP_embeddings_centers_info.pkl')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--noise_variance', type=float, default=0, help='noise variance')
    parser.add_argument('--uniform_noise', dest='uniform_noise', action='store_true', default=False, help='use uniform noise instead of gaussian')
    parser.add_argument('--dont_norm', dest='dont_norm', action='store_true', default=False, help='dont normalize CLIP embeddings')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1, help='save every n epochs')
    parser.add_argument('--prefix_length', type=int, default=40, help='prefix length')
    parser.add_argument('--prefix_length_clip', type=int, default=40, help='prefix length for clip')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true', default=False, help='train only the mapper between CLIP and GPT, while GPT is frozen')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='type of architurctre between CLIP and GPT (mlp/transformer)')
    parser.add_argument('--num_layers', type=int, default=8, help='number of layers in the mapper')
    parser.add_argument('--is_not_rn', dest='is_not_rn', action='store_true', default=False, help='Choose the CLIP backbone: False for RN, True for ViT')
    parser.add_argument('--use_image_embedding_as_clipcap', dest='use_image_embedding_as_clipcap', action='store_true', default=False, help='use image embedding as ClipCap')
    args = parser.parse_args()
    directory_path = args.out_dir
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

    prefix_length = args.prefix_length
    dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=not args.dont_norm, use_image_embedding_as_clipcap=args.use_image_embedding_as_clipcap)
    prefix_dim = 768
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train both prefix and GPT")
        sys.stdout.flush()
    if args.pretrain_weights != '':
        model.load_state_dict(torch.load(args.pretrain_weights, map_location=device))
    print(f'modality_offset={args.add_modality_offset}')
    from pathlib import Path
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{args.out_dir}/train_commandline_args.txt', 'w') as f:
        args_at_dict = args.__dict__
        args_at_dict.pop('mapping_type')
        json.dump(dict(args_at_dict), f, indent=2)
        print(f'args saved to file {args.out_dir}/train_commandline_args.txt')
    train(dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    main()