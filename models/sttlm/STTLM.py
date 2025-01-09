import torch.nn as nn
import torch
from transformers import LlamaConfig, LlamaModel
import pywt

from peft import LoraConfig, get_peft_model
from models.sttlm.attn_layer import Selfattention_Layer

class STTLM(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        positional_embedding_dim=24,
        Wav_embedding_dim =24,
        feed_forward_dim=256,
        num_heads=4,
        num_layers_t=3,
        num_layers_s=4,
        dropout=0.1,
        llm_layers = 1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.positional_embedding_dim = positional_embedding_dim
        self.Wav_embedding_dim = Wav_embedding_dim
        self.model_dim_t = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + positional_embedding_dim
        )
        self.model_dim_s = (
            self.model_dim_t
            + Wav_embedding_dim
            
        )
        self.num_heads = num_heads
        self.num_layers_t = num_layers_t
        self.num_layers_s = num_layers_s
        self.d_llm = 4096
        self.llm_layers = llm_layers
        #self.llama_config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.llama_config = LlamaConfig.from_pretrained("/root/data1/HY/09LLaMa/llama-7b")
        self.llama_config.num_hidden_layers = self.llm_layers
        self.llama_config.output_hidden_states = True
        self.input_proj = nn.Linear(1, input_embedding_dim, dtype=torch.float32)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim) 
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.positional_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, positional_embedding_dim))
            )
        self.Wav_embedding = nn.Linear(1, Wav_embedding_dim)
        self.attn_layers_t = nn.ModuleList(
            [
                Selfattention_Layer(self.model_dim_t, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers_t)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                Selfattention_Layer(self.model_dim_s, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers_s)
            ]
        )
        self.llama = LlamaModel.from_pretrained(
            #"meta-llama/Llama-2-7b-hf",
            "/root/data1/HY/09LLaMa/llama-7b",
            trust_remote_code=True,
            local_files_only=False,
            config=self.llama_config,
        )
        self.LoRA_config = LoraConfig(
                                        r=32, #Rank
                                        lora_alpha=32,
                                        target_modules=[
                                            'q_proj',
                                            'k_proj',
                                            'v_proj',
                                            'dense' 
                                        ],
                                        bias="none",
                                        lora_dropout=0.05,
                                        task_type="CAUSAL_LM",
                                    )
        self.peft_model = get_peft_model(self.llama, self.LoRA_config)
        self.pad_size = 4096 - self.model_dim_s
        self.padding = (0, self.pad_size)  # padding 0
        
        self.output_projection = nn.Linear(self.d_llm,self.output_dim)
        
    def cal_wavelet(self,input):   #wavelet 
        x = input.cpu().detach().numpy()
        # [16,1,207,12]
        wavelet = 'db1'  # Daubechies 1
        coeffs = pywt.wavedec(x, wavelet, level=1, axis=-1,mode='zero')
        xA = torch.from_numpy(coeffs[0]).to('cuda')
        xD = torch.from_numpy(coeffs[1]).to('cuda')
        embeddings = []
        embeddings.append(torch.cat([xA, xD], dim=-1))
        final_embeddings = torch.cat(embeddings, dim=-1)
        return final_embeddings

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        #（16,12,207,3）
        xn0 = x[:,:,:,0:1]
        batch_size = x.shape[0]
        tod = x[..., 1]
        dow = x[..., 2]   
        x0 = x[..., : self.input_dim]
        x1 = self.input_proj(x0[...,0:1]) 
        features = [x1]
        tod_emb = self.tod_embedding(
            (tod * self.steps_per_day).long()
        )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
        features.append(tod_emb)
        dow_emb = self.dow_embedding(
            dow.long()
        )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
        features.append(dow_emb)
        pos_emb = self.positional_embedding.expand(
            size=(batch_size, *self.positional_embedding.shape)
        )
            #print(adp_emb.shape)
        features.append(pos_emb)
        Et = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim_t)
        
        for attn in self.attn_layers_t:
            x = attn(Et, dim=1)
        features1 = [x]  
        
        if self.Wav_embedding_dim>0:
            xn1=xn0.permute(0,3,2,1)   #[16,1,207,12]
            x_final = self.cal_wavelet(xn1)  #[16,1,207,12]
            xn1=x_final.permute(0,3,2,1)
            xn1 = self.Wav_embedding(xn1)
            Es = xn1
            features1.append(xn1)
        x = torch.cat(features1, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim_s)

        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
       
        out = x
        B, T , N ,D= out.size()
        x_enc = out
        tensor_shape = (B, T, N,1 )
        dec_out_data = torch.empty(tensor_shape).to(x_enc.device)

        for i in range(x_enc.shape[2]):
            enc_slice = x_enc[:,:,i,:]
            enc_out = torch.nn.functional.pad(enc_slice, self.padding, mode='constant', value=0)
            dec_out = self.peft_model(inputs_embeds=enc_out).last_hidden_state
            dec_out = self.output_projection(dec_out)
            dec_out_data[:,:,i,:] = dec_out
            


        return dec_out_data