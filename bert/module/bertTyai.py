from transformers import BertForSequenceClassification
# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
import torch
from transformers import BertTokenizer
import numpy as np

def pad_sequences(sequences, maxlen=None, dtype='int64',
                  padding='pre', truncating='pre', value=0.):
    """Pads each sequence to the same length (length of the longest sequence).

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)

    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 4, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
MAX_LEN = 200
# model Load
# Load
device = torch.device('cpu')
PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型
# 取得此預訓練模型所使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

def process_input(Q, text, tokenizer, MAX_LEN):
  model.load_state_dict(torch.load('./bert/module/model_save_'+Q+'/pytorch_model.bin', map_location=device))
  
  encoded_sent = tokenizer.encode(
                        text,                      # text to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                   )
  input_ids = pad_sequences([encoded_sent], maxlen=MAX_LEN, 
                           truncating="post", padding="post")
  attention_masks = []
  for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
  prediction_inputs = torch.tensor(input_ids)
  prediction_masks = torch.tensor(attention_masks)
  return prediction_inputs, prediction_masks


def getResult(Q, text, tokenizer=tokenizer, MAX_LEN=MAX_LEN):
  prediction_inputs, prediction_masks = process_input(Q, text, tokenizer, MAX_LEN)
  model.eval()
  model.cpu()
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(prediction_inputs, token_type_ids=None, 
                    attention_mask=prediction_masks)
  result = np.argmax(outputs[0], axis=1).flatten()
  return '您目前對本題的理解為 %.f 級(可預測最高為 3 級)' % (result)