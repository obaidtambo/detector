from django.shortcuts import render

import torch
from torch.nn import Softmax
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import RobertaTokenizer
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class SequenceClassifierOutputWithLastLayer(SequenceClassifierOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaForContrastiveClassification(RobertaForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.soft_max = Softmax(dim=1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        softmax_logits = self.soft_max(logits)

        if not return_dict:
            output = (softmax_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithLastLayer(
            loss=loss,
            logits=softmax_logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
device=torch.device('cpu')   
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

import os
print("Current working directory:", os.getcwd())
# change the model path and name as required
checkpoint = torch.load('app\model\model_gpt.pt', map_location=device) #for cpu 
model = RobertaForContrastiveClassification.from_pretrained('roberta-base').to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("************* Model Loaded from: app\model\model_gpt.pt **************" )

def predict_label(model, tokenizer, text, device, threshold=0.68):
    # Tokenize input text
    # try:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs['logits']


    # Get predicted label and its confidence score
    _, predicted_label = torch.max(logits, 1)
    predicted_label = predicted_label.item()


    # Calculate confidence score -contributed @ ObaidTamboli
    confidence_score = torch.nn.functional.softmax(logits, dim=1)[0][predicted_label].item()

    return predicted_label, confidence_score

# Define the home page view
def home(request):
    input_text = ''
    text_label= ''
    confidence_score = ''

    if request.method == 'POST':
        # Retrieve input text from the POST request
        input_text = request.POST.get('input_text', '').strip()

        # Predict label and confidence score using the predict_label function
        predicted_label, confidence_score = predict_label(model, tokenizer, input_text, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print("Predicted Label: ", predicted_label, "Conficence Score",confidence_score )
        if predicted_label== 1:
            text_label="Human Written"
        else:
            text_label="Machine Written" 

    # Pass input text, predicted label, and confidence score to the home page as context
    context = {
        'input_text': input_text,
        'predicted_label': text_label,
        'confidence_score': confidence_score
    }
    return render(request, 'index.html', context)


# # Define the home page view
# def home(request):
#     if request.method == 'POST':
#         # Retrieve input text from the POST request
#         input_text = request.POST['input_text'].strip()

#         # Predict label and confidence score using the predict_label function
#         predicted_label, confidence_score = predict_label(model, tokenizer, input_text, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

#         # Pass predicted label and confidence score to the home page as context
#         context = {
#             'input_text': input_text,
#             'predicted_label': predicted_label,
#             'confidence_score': confidence_score
#         }
#         return render(request, 'index.html', context)
#     else:
#         return render(request, 'index.html')
