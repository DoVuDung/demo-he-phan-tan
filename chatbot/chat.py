from transformers import RobertaTokenizer, RobertaModel
import numpy as np
import random as rd
import re
import pandas as pd
import torch


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
data_path = "Chatbot_bosungcautraloi_ver01.1.xlsx"
df_answer = pd.read_excel(data_path, sheet_name='Answering') 
ans_dict = {k: g["Answer"].tolist() for k,g in df_answer.groupby("Type")}

def get_prediction(model, tokenizer, txt):
  str = re.sub('r[^a-zA-Z ]+', '', txt)
  test_text = [str]
  model.eval()
  
  tokens_test_data = tokenizer(
  test_text,
  truncation=True,
  return_token_type_ids=False
  )
  test_seq = torch.tensor(tokens_test_data['input_ids'])
  test_mask = torch.tensor(tokens_test_data['attention_mask'])
  
  preds = None
  with torch.no_grad():
    preds = model([test_seq, test_mask])
  preds = preds.detach().cpu().numpy()
  preds = np.argmax(preds, axis=1) 
  return int(preds)

def get_response(message, ans_dict, model, tokenizer): 
  preds = get_prediction(model, tokenizer, message)

  result = rd.choice(ans_dict[preds])
  return result

  
