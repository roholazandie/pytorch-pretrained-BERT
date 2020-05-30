import torch
from transformers import AlbertTokenizer, AlbertModel, BertTokenizer, BertModel

## BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]
print(last_hidden_states)


# RoBRETa
from transformers import RobertaTokenizer, RobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=labels)
loss, logits = outputs[:2]
print(logits)

# ALBERT
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained('albert-base-v2')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]
print(last_hidden_states)


## BART
from transformers import BartTokenizer, BartForConditionalGeneration
tokenizer = BartTokenizer.from_pretrained('bart-large')
model = BartForConditionalGeneration.from_pretrained('bart-large')
TXT = "My friends are good but they eat too many <mask> ."
input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
logits = model(input_ids)[0]
masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
probs = logits[0, masked_index].softmax(dim=0)
values, predictions = probs.topk(5)
print(tokenizer.decode(predictions).split())


## ELECTRA
from transformers import ElectraForMaskedLM, ElectraTokenizer
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
model = ElectraForMaskedLM.from_pretrained('google/electra-small-generator')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, masked_lm_labels=input_ids)

loss, prediction_scores = outputs[:2]
print(prediction_scores)


## Longformer
from transformers import LongformerModel, LongformerTokenizer

model = LongformerModel.from_pretrained('longformer-base-4096')
tokenizer = LongformerTokenizer.from_pretrained('longformer-base-4096')

SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

# Attention mask values -- 0: no attention, 1: local attention, 2: global attention
attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
attention_mask[:, [1, 4, 21,]] = 2  # Set global attention based on the task. For example,
                                    # classification: the <s> token
                                    # QA: question tokens
                                    # LM: potentially on the beginning of sentences and paragraphs
#sequence_output, pooled_output = model(input_ids, attention_mask=attention_mask)
#print(sequence_output)

##T5
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
input_ids = tokenizer.encode("summarize: Very nice set of pencils! I did receive all 5 pieces. To be honest I was concerned when I saw so many reviews stating they only received 4 pieces. Figured it was worth the risk, for such a great price. The pens feel great. Nice and sturdy, grips feel good too. The highlight of this set is the color coding!! Tell which is which at a glance", return_tensors="pt")  # Batch size 1
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs.squeeze(0), skip_special_tokens=True))