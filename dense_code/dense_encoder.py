from transformers import BertModel, BertPreTrainedModel, RobertaModel, RobertaPreTrainedModel

class RobertaEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaEncoder, self).__init__(config)

        self.roberta = RobertaModel(config)
        # 변경점
        self.init_weights()

    def forward(self, input_ids, attention_mask = None, token_type_ids = None):
        outputs =self.roberta(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs['pooler_output']
        return pooled_output
    
class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        # 변경점
        self.init_weights()

    def forward(self, input_ids, attention_mask = None, token_type_ids = None):
        outputs =self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output =outputs['pooler_output']
        return pooled_output