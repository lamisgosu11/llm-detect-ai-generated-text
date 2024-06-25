import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup
import tensorflow as tf

class DAIGTModel(nn.Module):
    def __init__(self, model_path, config, tokenizer, pretrained=False):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(model_path, config=config)
        else:
            self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)  
        #self.model.gradient_checkpointing_enable()    
    def forward_features(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return embeddings
    def forward(self, input_ids, attention_mask):
        embeddings = self.forward_features(input_ids, attention_mask)
        logits = self.classifier(embeddings)
        return logits
    

#bert tensorflow
class bertTF(nn.Module):
    def __init__(self, model_path, config, tokenizer, pretrained=False):
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = tokenizer
        self.config = config
        self.pretrained = pretrained
    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.cpu().numpy()
        attention_mask = attention_mask.cpu().numpy()
        input_ids = self.tokenizer(input_ids, return_tensors='tf', padding=True, truncation=True, max_length=self.config.max_length)
        attention_mask = self.tokenizer(attention_mask, return_tensors='tf', padding=True, truncation=True, max_length=self.config.max_length)
        logits = self.model.predict([input_ids, attention_mask])
        return logits

    