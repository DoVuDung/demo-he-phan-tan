import torch
from torch import nn

# Build model
class BertForClassification(nn.Module):
  def __init__(self, bert):
    super(BertForClassification, self).__init__()
    self.bert = bert

    self.dropout = nn.Dropout(0.2)
    self.relu = nn.ReLU()

    self.fc1 = nn.Linear(768, 512)
    self.fc2 = nn.Linear(512, 7)
    
    self.softmax = nn.LogSoftmax(dim=1)
    
  def forward(self, x_mask):
    x, mask = x_mask[0], x_mask[1]
    cls_hs = self.bert(x, attention_mask=mask)[0][:,0]
    x = self.fc1(cls_hs)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.fc2(x)
    x = self.softmax(x)
    return x

