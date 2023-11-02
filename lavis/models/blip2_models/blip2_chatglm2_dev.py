
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from transformers import AutoTokenizer, AutoModel
from lavis.models.blip2_models.chatglm2.modelling_chatglm_dev import ChatGLMForConditionalGeneration
from lavis.models.blip2_models.chatglm.tokenization_chatglm import ChatGLMTokenizer
import re

@registry.register_model("blip2_chatglm2")
class Blip2ChatGLM2(Blip2Base):
    """
    BLIP2 ChatGLM model.
    Supported model types:
        - pretrain_chatglm: pretrained model with ChatGLM
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_chatglm": "configs/models/blip2/blip2_pretrain_chatglm.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        chatglm_model="THUDM/chatglm-6b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        print("blip2_chatglm2,blip2_chatglm2,blip2_chatglm2")
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model,img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False               
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.ChatGLM_tokenizer = ChatGLMTokenizer.from_pretrained(chatglm_model, trust_remote_code=True)
        self.ChatGLM_model = ChatGLMForConditionalGeneration.from_pretrained(chatglm_model, trust_remote_code=True, torch_dtype=torch.bfloat16).half()
        # self.ChatGLM_model.gradient_checkpointing_enable()
        for name, param in self.ChatGLM_model.named_parameters():
            param.requires_grad = False
        # self.eos_token_id = self.ChatGLM_tokenizer(
        #     "\n", add_special_tokens=False
        # ).input_ids[0]
        self.ignore_pad_token_for_loss = True
        self.ChatGLM_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.ChatGLM_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.ChatGLM_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.input_ids.size(1)

    def prompt_wrap(self, img_embeds, atts_img, prompt, use_speech=True, video=False):
        if use_speech:
            special_token = '<SpeechHere>'
        else:
            special_token = '<ImageHere>'
        if video:
            special_token = '<VideoHere>'
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split(special_token)
            p_before_tokens = self.ChatGLM_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.ChatGLM_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.ChatGLM_model.transformer.word_embeddings(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.ChatGLM_model.transformer.word_embeddings(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img



    def forward(self, samples):
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # Qformer输出
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        # Qformer输出维度变换未glm
        image_embeds = self.ChatGLM_proj(query_output.last_hidden_state)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
 
        
        text = [t + "\n" for t in samples["text_input"]]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)
        
        
        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens = self.t5_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
        
        
        print(samples)
        image_prompt = '<Image><ImageHere></Image>'
        image_embeds, image_atts = self.prompt_wrap(image_embeds, image_atts, image_prompt, use_speech=False)
        
        examples= {}
        samples["text_output"] = []
        for i in range(0,len(samples["text_input"])):
            samples["text_output"].append(samples["text_input"][i])
            samples["text_input"][i]  = 'Human: " + "Please describe this picture" + " \\nSONGBO: '
        print(samples)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            ChatGLM_tokens = self.preprocess_function_train( samples, image_atts.device, src_txt_len=200, max_tgt_len=200)
            
            empty_targets = (torch.ones(image_atts.size(),dtype=torch.long).to(image_atts.device).fill_(0))

            ChatGLM_tokens['input_ids'] = torch.cat([empty_targets, ChatGLM_tokens['input_ids']], dim=1).to(image_embeds.device)
            ChatGLM_tokens['labels'] = torch.cat([empty_targets.fill_(-100), ChatGLM_tokens['labels']], dim=1).to(image_embeds.device)


            outputs = self.ChatGLM_model(
                **ChatGLM_tokens,
                input_image=image_embeds,
                return_dict=True,
            )
            loss = outputs.loss


        return {"loss": loss}

    
    #函数preprocess_function_train(examples)的主要目标是为模型训练阶段预处理数据。给定一些训练样例examples，它将为每个样例生成模型需要的输入和标签。这个过程包括以下几个步骤：
    def preprocess_function_train(self, examples, device, src_txt_len=None, max_tgt_len=None):
        # max_seq_length #定义最大序列长度为源长度上限（即问题长度上限）加上目标长度上限（即答案长度上限）再加1。这个1通常是为特殊标记（比如序列结束标记）预留的空间。
        max_seq_length = src_txt_len+ max_tgt_len + 1  
        model_inputs = {  #初始化model_inputs字典，用于存储模型输入的数据。
            "input_ids": [],
            "labels": [],
        }
        for query, answer in zip(examples["text_input"], examples["text_output"]):  #遍历每一个样例。
            history = []
            # prompt = self.ChatGLM_tokenizer.build_prompt(query, history)  #用tokenizer.build_prompt方法来根据问题和历史对话构建提示。
            prompt = query  #在提示前面添加前缀。
            a_ids = self.ChatGLM_tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,  #将提示编码成模型可以理解的形式，得到输入ID序列a_ids。
                                                    max_length=src_txt_len)
            b_ids = self.ChatGLM_tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,  #同样地，将答案编码成模型可以理解的形式，得到答案ID序列b_ids。
                                                    max_length=max_tgt_len)

            context_length = len(a_ids)  #计算输入的长度。
            input_ids = a_ids + b_ids + [self.ChatGLM_tokenizer.eos_token_id]  #将输入和答案的ID序列拼接起来，并在最后添加一个序列结束标记的ID，得到完整的输入序列。
            labels = [self.ChatGLM_tokenizer.pad_token_id] * context_length + b_ids + [self.ChatGLM_tokenizer.eos_token_id]  #标签序列的前context_length部分是填充标记的ID，后面是答案的ID序列和一个序列结束标记的ID。这样设置的原因是，我们只关心模型对答案部分的预测。
            
            pad_len = max_seq_length - len(input_ids)  #计算需要填充的长度。
            input_ids = input_ids + [self.ChatGLM_tokenizer.pad_token_id] * pad_len  #在输入序列后面添加填充标记，使其长度达到max_seq_length。
            labels = labels + [self.ChatGLM_tokenizer.pad_token_id] * pad_len  #同样地，也在标签序列后面添加填充标记。
            if self.ignore_pad_token_for_loss:  #如果设置了忽略填充标记的损失，那么将标签中的填充标记的ID替换为-100。
                labels = [(l if l != self.ChatGLM_tokenizer.pad_token_id else -100) for l in labels]  #将处理好的输入和标签添加到model_inputs字典中。

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)  #在处理完所有样例后，返回model_inputs字典，它包含了所有样例的输入和标签，可以直接用于模型的训练。
        model_inputs["input_ids"] = torch.LongTensor(model_inputs["input_ids"]).to(device)
        model_inputs["labels"] = torch.LongTensor(model_inputs["labels"]).to(device)
        return model_inputs

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        **kwargs
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        max_length=2048
        num_beams=1
        do_sample=True
        top_p=0.7
        temperature=0.95
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, **kwargs}

        image = samples["image"]
        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):          
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = self.ChatGLM_proj(query_output.last_hidden_state)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.bool).to(image.device)


            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = samples["text_input"]
                # prompt = self.prompt

            image_prompt = '<Image><ImageHere></Image>'
                   
            image_embeds, image_atts = self.prompt_wrap(image_embeds, image_atts, image_prompt, use_speech=False)


            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            ChatGLM_tokens = self.ChatGLM_tokenizer([prompt], return_tensors="pt", padding=True).to(self.device)
            ChatGLM_tokens = ChatGLM_tokens.to(self.device)
            context_length = ChatGLM_tokens.input_ids.size(1)   


            empty_targets = (
                torch.ones(image_atts.size(), dtype=torch.long).to(image.device).fill_(0)
            )
            ChatGLM_tokens['input_ids'] = torch.cat([empty_targets, ChatGLM_tokens.input_ids], dim=1)
            # ChatGLM_tokens['attention_mask'] = torch.cat([atts_opt, ChatGLM_tokens.attention_mask], dim=1)
            ChatGLM_tokens = ChatGLM_tokens.to(image.device)

            del ChatGLM_tokens['attention_mask']
            del ChatGLM_tokens['position_ids']
            context_length = ChatGLM_tokens.input_ids.size(1)

            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):

                outputs = self.ChatGLM_model.generate(
                    **ChatGLM_tokens, **gen_kwargs, input_image=image_embeds,
                )

                outputs = outputs.tolist()[0][context_length -  2:]
                response = self.ChatGLM_tokenizer.decode(outputs)
                response = response.strip()
                response = response.replace("[[训练时间]]", "2023年")
                return [response]
            

    @torch.no_grad()
    def generate_demo(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        **kwargs
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        max_length=2048
        num_beams=1
        do_sample=True
        top_p=0.7
        temperature=0.95
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, **kwargs}

        use_image = samples["use_image"]
        image = samples["image"]
        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):          
            if use_image:
                image = samples["image"].to(self.device)
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                    image.device
                )

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                image_embeds = self.ChatGLM_proj(query_output.last_hidden_state)
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.bool).to(image.device)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = samples["text_input"]
                # prompt = self.prompt

            # if isinstance(prompt, str):
            #     prompt = [prompt] * image.size(0)
            # else:
            #     assert len(prompt) == image.size(
            #         0
            #     ), "The number of prompts must be equal to the batch size."


            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            ChatGLM_tokens = self.ChatGLM_tokenizer([prompt], return_tensors="pt", padding=True).to(image.device)
            ChatGLM_tokens = ChatGLM_tokens.to(image.device)
            context_length = ChatGLM_tokens.input_ids.size(1)    
            
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                if use_image:
                    image_prompt = '<Image><ImageHere></Image>'
                   
                    image_embeds, image_atts = self.prompt_wrap(image_embeds, image_atts, image_prompt, use_speech=False)

                    empty_targets = (
                        torch.ones(image_atts.size(), dtype=torch.long).to(image.device).fill_(0)
                    )
                    ChatGLM_tokens['input_ids'] = torch.cat([empty_targets, ChatGLM_tokens.input_ids], dim=1)
                    # ChatGLM_tokens['attention_mask'] = torch.cat([atts_opt, ChatGLM_tokens.attention_mask], dim=1)
                    ChatGLM_tokens = ChatGLM_tokens.to(image.device)

                    del ChatGLM_tokens['attention_mask']
                    del ChatGLM_tokens['position_ids']
                    context_length = ChatGLM_tokens.input_ids.size(1)
                    print('context_length: ', context_length)
                    outputs = self.ChatGLM_model.generate(
                            **ChatGLM_tokens, **gen_kwargs, input_image=image_embeds,
                        )
                else:
                    outputs = self.ChatGLM_model.generate(
                        **ChatGLM_tokens, **gen_kwargs, 
                    )
                print('output length: ', len(outputs.tolist()[0]))
                # print(outputs.tolist()[0])
                # response = self.ChatGLM_tokenizer.decode(outputs.tolist()[0])
                # print(response)
                outputs = outputs.tolist()[0][context_length -  2:]
                response = self.ChatGLM_tokenizer.decode(outputs)
                response = response.strip()
                response = response.replace("[[训练时间]]", "2023年")
                response = self.process_response(response)
                return [response]
            
    def process_response(self, response):
        response = response.strip()
       
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response
            
    @torch.no_grad()
    def _generate(
            self,
            **kwargs,
    ):
        MASK, gMASK = 150000, 150001
        bos, eos = 150004, 150005

        if "eos_token_id" not in kwargs:
            kwargs["eos_token_id"] = eos

        stop = False

        return_seqs = []

        while True:
            output_ids = super().generate(**kwargs)
            kwargs['inputs_opt'] = None
            return_seqs = []
            max_length = 0

            for i in range(output_ids.shape[0]):
                output_seq = output_ids[i].tolist()
                mask_token = MASK if MASK in output_seq else gMASK
                mask_position = output_seq.index(mask_token)
                bos_position = output_seq.index(bos)
                if eos in output_seq:
                    eos_position = output_seq.index(eos)
                else:
                    eos_position = len(output_seq)

                return_seq = output_seq[:mask_position] + output_seq[bos_position + 1:eos_position] + output_seq[
                                                                                                      mask_position + 1:bos_position]
                max_length = max(max_length, len(return_seq))
                return_seqs.append(return_seq)

            for i in range(output_ids.shape[0]):
                return_seqs[i] = [0] * (max_length - len(return_seqs[i])) + return_seqs[i]  # padding
                if mask_token not in return_seqs[i]:
                    stop = True

            if stop:
                break

            for return_seq in return_seqs:
                return_seq += [bos]

            kwargs['input_ids'] = torch.tensor(return_seqs, dtype=torch.long, device=kwargs['input_ids'].device)

        return torch.tensor(return_seqs, dtype=torch.long, device=kwargs['input_ids'].device)

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        ChatGLM_model = cfg.get("chatglm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            chatglm_model=ChatGLM_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)




        return model
    
    
    
    
    # def preprocess_function_train(self, examples, device, src_txt_len=None, max_tgt_len=None):
    #     if src_txt_len is None:
    #         src_txt_len = self.max_txt_len
    #     if src_txt_len is None:
    #         max_tgt_len = self.max_txt_len

    #     max_seq_length = src_txt_len + max_tgt_len

    #     model_inputs = {
    #         "input_ids": [],
    #         "labels": [],
    #     }
    #     for question, answer in zip(examples["text_input"], examples["text_output"]):
    #         print(question, answer)
    #         prompt = question
    #         a_ids = self.ChatGLM_tokenizer.encode(text=prompt, add_special_tokens=False)
    #         b_ids = self.ChatGLM_tokenizer.encode(text=answer, add_special_tokens=False)
    #         print("a_ids",a_ids)
    #         print("b_ids",b_ids)
    #         if len(a_ids) > src_txt_len - 1:
    #             a_ids = a_ids[: src_txt_len - 1]

    #         if len(b_ids) > max_tgt_len- 2:
    #             b_ids = b_ids[: max_tgt_len - 2]
            
    #         input_ids = self.ChatGLM_tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
    #         print("input_ids",input_ids)
    #         print(self.ChatGLM_tokenizer.eos_token_id)
    #         context_length = input_ids.index(self.ChatGLM_tokenizer.eos_token_id )
    #         mask_position = context_length - 1
    #         labels = [-100] * context_length + input_ids[mask_position+1:]
            
    #         pad_len = max_seq_length - len(input_ids)
    #         input_ids = input_ids + [self.ChatGLM_tokenizer.pad_token_id] * pad_len
    #         labels = labels + [self.ChatGLM_tokenizer.pad_token_id] * pad_len
    #         if self.ignore_pad_token_for_loss:
    #             labels = [(l if l != self.ChatGLM_tokenizer.pad_token_id else -100) for l in labels]

    #         model_inputs["input_ids"].append(input_ids)
    #         model_inputs["labels"].append(labels)
    #     model_inputs["input_ids"] = torch.LongTensor(model_inputs["input_ids"]).to(device)
    #     model_inputs["labels"] = torch.LongTensor(model_inputs["labels"]).to(device)
    #     return model_inputs