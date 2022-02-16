
import os

from torch.nn.parameter import Parameter
from openprompt.utils.logging import logger



from openprompt.data_utils import InputExample, InputFeatures
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt.config import get_config
import torch
from torch import nn

config, args = get_config()

if config.soft_template.method == "prefix-tuning":
    if config.soft_template.instance and config.soft_template.encoder == "mlp":
        from  sequence_models_prefix.pt_model_mlp import *
    if config.soft_template.instance and config.soft_template.encoder == "lstm":
        from  sequence_models_prefix.pt_model_lstm import *
    if config.soft_template.instance and config.soft_template.encoder == "cnn":
        from  sequence_models_prefix.pt_model_textcnn import *

else:
    if config.soft_template.instance and config.soft_template.encoder == "mlp":
        from  sequence_models.pt_model_mlp import *
    if config.soft_template.instance and config.soft_template.encoder == "lstm":
        from  sequence_models.pt_model_lstm import *
    if config.soft_template.instance and config.soft_template.encoder == "cnn":
        from  sequence_models.pt_model_textcnn import *

class SoftTemplate(Template):
    r"""This is the implementation of `The Power of Scale for Parameter-Efficient
    Prompt Tuning <https://arxiv.org/pdf/2104.08691v1.pdf>`_ . Similar to :obj:`PrefixTuningTemplate`,
    This template also does not need any textual template. Addition tokens are directly
    concatenated into the input ids. There are two initializations of the new tokens. 
    (1). random initialization. (2) initialize with the tokens of the plm (We simply take 
    the first n_tokens similar to their implementation).

    Note that this template can be simply achieved by :obj:`SoftManualTemplate`, in which
    you set `n_token` <soft> tokens template before the <text_a> will give the same result.
    """
    registered_inputflag_names = ["loss_ids", "shortenable_ids"]

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 mask_token: str = '<mask>',
                 num_tokens: int=20,
                 initialize_from_vocab: Optional[bool] = True,
                 random_range: Optional[float] = 0.5,
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer,
                         mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
       
        self.raw_embedding = model.get_input_embeddings()
        self.model_is_encoder_decoder = model.config.is_encoder_decoder
        self.random_range = random_range
        self.num_tokens = num_tokens
        self.initialize_from_vocab = initialize_from_vocab
        self.model = model 
        self.prefix_tokens = torch.arange(self.num_tokens).long()
        self.embedding = torch.nn.Embedding(self.num_tokens, 2*self.model.config.hidden_size*self.model.config.num_hidden_layers)
        self.dropout = torch.nn.Dropout(0.1) 
        
        if config.soft_template.instance == True and (config.soft_template.encoder == "mlp" or config.soft_template.encoder == "cnn" or config.soft_template.encoder == "lstm"):
            self.pt_init_model =  get_model()         

        self.text = text 

        if self.num_tokens>0:
            if config.soft_template.method == "prefix-tuning":
                self.generate_parameters_prefix()
            else:
                self.generate_parameters()

    def on_text_set(self):
        self.text = self.parse_text(self.text)
    def wrap_one_example(self, example) -> List[Dict]:  #TODO this automatic generated template may not be able to process diverse data format.
        if self.text is None:
            logger.warning("You didn't provide text templat efor softprompt. Using default template, is this intended?")
            if example.text_b is None:
                self.text = self.default_text1
            else:
                self.text = self.default_text2
        return super().wrap_one_example(example)


    def generate_parameters(self) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        if config.soft_template.instance == True:
            if config.soft_template.instance_pretrain == True:
                soft_embeds = torch.load(config.soft_template.emb_checkpoints)
            else:
                soft_embeds = torch.FloatTensor(self.model.config.vocab_size, self.raw_embedding.weight.size(1)).uniform_(-self.random_range, self.random_range)
        else:
            soft_embeds = torch.FloatTensor(self.num_tokens, self.raw_embedding.weight.size(1)).uniform_(-self.random_range, self.random_range)
        self.soft_embeds = nn.Parameter(soft_embeds, requires_grad=True)

    def generate_parameters_prefix(self) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        if config.soft_template.instance == True and (config.soft_template.instance_pretrain or config.soft_template.encoder == "mlp" or config.soft_template.encoder == "cnn" or config.soft_template.encoder == "lstm"):
            soft_embeds = torch.load(config.soft_template.emb_checkpoints)
        else:
            soft_embeds = torch.FloatTensor(self.model.config.vocab_size, self.raw_embedding.weight.size(1) * 2).uniform_(-self.random_range, self.random_range)
        self.soft_embeds = nn.Parameter(soft_embeds, requires_grad=True)

    
    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        inputs_embeds = self.raw_embedding(batch['input_ids'])
        
        input_ids = batch['input_ids']
        batch_size = inputs_embeds.size(0)
        if self.num_tokens>=0:
        


            soft_embeds = None
            if config.soft_template.method != "prefix-tuning":
                if config.soft_template.instance != True:
                    soft_embeds = self.soft_embeds.repeat(batch_size, 1, 1)
                      
                if config.soft_template.instance == True:
                    input_ids = input_ids[:,:self.num_tokens]
                    for i in range(input_ids.shape[0]):
                        if soft_embeds is None:
                            soft_embeds = torch.index_select(self.soft_embeds,0,input_ids[i])
                        else:
                            tmp_embeds = torch.index_select(self.soft_embeds,0,input_ids[i])
                            soft_embeds = torch.cat((soft_embeds,tmp_embeds),0) 

                
                if config.soft_template.instance == True:
                    soft_embeds = soft_embeds.reshape(input_ids.shape[0],-1,soft_embeds.shape[-1])
                
                if config.soft_template.encoder == "cnn":
                    soft_embeds = self.pt_init_model(torch.unsqueeze(soft_embeds,1).cuda())[1]
                elif config.soft_template.encoder == "mlp" or config.soft_template.encoder == "lstm":
                    soft_embeds = self.pt_init_model(soft_embeds)[1]

                inputs_embeds = torch.cat([soft_embeds, inputs_embeds], 1)
                batch['input_ids'] = None
                batch['inputs_embeds'] = inputs_embeds
                if 'attention_mask' in batch and self.num_tokens>0:
                    am = batch['attention_mask']
                    batch['attention_mask'] = torch.cat([torch.ones((batch_size,self.num_tokens), dtype = am.dtype,device=am.device), am], dim=-1)
                return batch


    
            propose_embeds = None
            if config.soft_template.instance == True:
                input_ids = input_ids[:,:self.num_tokens]
                for i in range(input_ids.shape[0]):
                    if propose_embeds is None:
                        propose_embeds = torch.index_select(self.soft_embeds,0,input_ids[i])
                    else:
                        tmp_embeds = torch.index_select(self.soft_embeds,0,input_ids[i])
                        propose_embeds = torch.cat((propose_embeds,tmp_embeds),0)
                propose_embeds = propose_embeds.reshape(input_ids.shape[0],-1,propose_embeds.shape[-1])
           
                if config.soft_template.encoder == "cnn":
         
                    propose_embeds = self.pt_init_model(torch.unsqueeze(propose_embeds,1).cuda())[1]
                elif config.soft_template.encoder == "mlp" or config.soft_template.encoder == "lstm":
                    propose_embeds = self.pt_init_model(propose_embeds)[1]
           
            prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).cuda()
            past_key_values = self.embedding(prefix_tokens).cuda()
            if config.soft_template.instance == True:
                past_key_values =  past_key_values.view(batch_size,self.num_tokens,self.model.config.num_hidden_layers,2*self.model.config.hidden_size).cuda()
                past_key_values[:,:,0,:] = past_key_values[:,:,0,:] + 0.05 * propose_embeds
            

            past_key_values = past_key_values.view(batch_size,self.num_tokens,self.model.config.num_hidden_layers * 2,self.model.config.num_attention_heads,int(self.model.config.hidden_size/self.model.config.num_attention_heads)).cuda()
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            batch['past_key_values'] = past_key_values
            

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        if 'attention_mask' in batch and self.num_tokens>0:
            am = batch['attention_mask']
            batch['attention_mask'] = torch.cat([torch.ones((batch_size,self.num_tokens), dtype = am.dtype,device=am.device), am], dim=-1)
        return batch
    

    def post_processing_outputs(self, outputs: torch.Tensor):
        r"""Post processing the outputs of language models according
        to the need of template. Most templates don't need post processing,
        The template like SoftTemplate, which appends soft template as a module
        (rather than a sequence of input tokens) to the input,
        should remove the outputs on these positions to keep the seq_len the same
        """
        if not self.model_is_encoder_decoder:
            outputs = outputs[1]
            #outputs.logits = outputs.logits[:, self.num_tokens:,: ]
        return outputs
