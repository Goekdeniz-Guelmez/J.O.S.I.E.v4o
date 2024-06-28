import os.path
from typing import List

import torch
from .header import *
import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
from transformers import StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM, AutoTokenizer
from .layers import *
from .common.utils import *


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            _stop = torch.tensor(stop).to(input_ids[0].device)
            indices = torch.where(_stop[0] == input_ids)
            for i in indices:
                if len(i) > 0:
                    if torch.all(input_ids[0][i:i + len(_stop)] == _stop):
                        stop_count += 1
        if stop_count >= self.ENCOUNTERS:
            return True
        return False


class JOSIEv4o(nn.Module):
    def __init__(self, **args):
        super(JOSIEv4o, self).__init__()
        self.args = args

        self.max_length = args['max_length']
        self.device = torch.cuda.current_device()
        self.stage = args['stage']
        print('args max_length', args['max_length'])

        imagebind_ckpt_path = os.path.join(self.args['pretrained_ckpt_path'], 'imagebind_ckpt', self.args['imagebind_version'])
        print(f'Initializing visual encoder from {imagebind_ckpt_path} ...')
        self.encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(pretrained=True, store_path=imagebind_ckpt_path)

        # freeze vision encoder
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        self.encoder.eval()
        print('Encoder initialized.')

        self.resoner_path = os.path.join(self.args['pretrained_ckpt_path'])
        print(f'Initializing language decoder from {self.resoner_path} ...')

        self.resoner = AutoModelForCausalLM.from_pretrained(self.resoner_path)
        if self.args.get('freeze_lm'):
            print("Freezing the Resoner ...")
            for param in self.resoner.parameters():
                param.requires_grad = False
            self.resoner.eval()
        else:
            print("Instruct tuning the LLaMa ...")
            # add the lora module
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.args['lora_r'],
                lora_alpha=self.args['lora_alpha'],
                lora_dropout=self.args['lora_dropout'],
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
            )

            self.resoner = get_peft_model(self.resoner, peft_config)
            self.resoner.print_trainable_parameters()
        print('Language decoder initialized.')

        # use the new trained tokenizer
        print(f'Initializing tokenizer from {self.resoner_path} ...')
        self.resoner_tokenizer = AutoTokenizer.from_pretrained(self.resoner_path)
        self.resoner_tokenizer.pad_token = self.resoner_tokenizer.eos_token
        self.resoner_tokenizer.padding_side = "right"
        self._add_image_token()
        self._add_video_token()
        self._add_audio_token()
        self.resoner.resize_token_embeddings(len(self.resoner_tokenizer))
        print('Tokenizer initialized.')

        self.input_proj = nn.Linear(self.visual_hidden_size, self.resoner.config.hidden_size)
        if self.args.get('freeze_input_proj'):
            for param in self.input_proj.parameters():
                param.requires_grad = False

        self.input_embeddings = self.resoner.get_input_embeddings()

    def _add_image_token(self):
        self.resoner_tokenizer.add_tokens(["<|image_start|>"])
        # self.resoner_tokenizer.add_tokens(["<|image_end|>"])
        self.args['gen_img_token_idx'] = []
        for i in range(self.args['num_gen_img_tokens']):
            print(f'Adding <|image_{i}|> token to vocabulary.')
            print(f'Before adding new token, tokenizer("<|image_{i}|>") =', self.resoner_tokenizer(f'<|image_{i}|>', add_special_tokens=False))
            num_added_tokens = self.resoner_tokenizer.add_tokens(f'<|image_{i}|>')
            print(f'After adding {num_added_tokens} new tokens, tokenizer("<|image_{i}|>") =', self.resoner_tokenizer(f'<|image_{i}|>', add_special_tokens=False))
            gen_token_idx = self.resoner_tokenizer(f'<|image_{i}|>', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_img_token_idx'].append(gen_token_idx[0])

    def _add_video_token(self):
        self.resoner_tokenizer.add_tokens({"<|video_start|>"})
        # self.resoner_tokenizer.add_tokens({"<|video_end|>"})
        self.args['gen_video_token_idx'] = []
        for i in range(self.args['num_gen_video_tokens']):
            print(f'Adding <|video_{i}|> token to vocabulary.')
            print(f'Before adding new token, tokenizer("<|video_{i}|>") =', self.resoner_tokenizer(f'<|video_{i}|>', add_special_tokens=False))
            num_added_tokens = self.resoner_tokenizer.add_tokens(f'<|video_{i}|>')
            print(f'After adding {num_added_tokens} new tokens, tokenizer("<|video_{i}|>") =', self.resoner_tokenizer(f'<|video_{i}|>', add_special_tokens=False))
            gen_token_idx = self.resoner_tokenizer(f'<|video_{i}|>', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_video_token_idx'].append(gen_token_idx[0])

    def _add_audio_token(self):
        self.resoner_tokenizer.add_tokens({"<|audio_start|>"})
        # self.resoner_tokenizer.add_tokens({"<|audio_end|>"})
        self.args['gen_audio_token_idx'] = []
        for i in range(self.args['num_gen_audio_tokens']):
            print(f'Adding <|audio_{i}|> token to vocabulary.')
            print(f'Before adding new token, tokenizer("<|audio_{i}|>") =', self.resoner_tokenizer(f'<|audio_{i}|>', add_special_tokens=False))
            num_added_tokens = self.resoner_tokenizer.add_tokens(f'<|audio_{i}|>')
            print(f'After adding {num_added_tokens} new tokens, tokenizer("<|audio_{i}|>") =', self.resoner_tokenizer(f'<|audio_{i}|>', add_special_tokens=False))
            gen_token_idx = self.resoner_tokenizer(f'<|audio_{i}|>', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_audio_token_idx'].append(gen_token_idx[0])

    def encode_video(self, video_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_video_data(video_paths, self.device)}
        inputs = {key: inputs[key].to(self.resoner.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.encoder(inputs)
            video_embeds = embeddings[ModalityType.VISION]  # bsz x 1024
        input_proj = self.input_proj(video_embeds).unsqueeze(1)  # bsz x 1 x llama_size
        atts_llama = torch.ones(input_proj.size()[:-1], dtype=torch.long).to(self.device)  # bsz x 1
        return input_proj, atts_llama

    def encode_audio(self, audio_paths):
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device)}
        inputs = {key: inputs[key].to(self.resoner.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.encoder(inputs)
            audio_embeds = embeddings[ModalityType.AUDIO]  # bsz x 1024
        input_proj = self.input_proj(audio_embeds).unsqueeze(1)  # bsz x 1 x llama_size
        atts_llama = torch.ones(input_proj.size()[:-1], dtype=torch.long).to(self.device)  # bsz x 1
        return input_proj, atts_llama

    def encode_image(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        inputs = {key: inputs[key].to(self.resoner.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.encoder(inputs)
            image_embeds = embeddings['vision']  # bsz x 1024
        input_proj = self.input_proj(image_embeds).unsqueeze(1)  # bsz x 1 x llama_size
        atts_llama = torch.ones(input_proj.size()[:-1], dtype=torch.long).to(self.device)  # bsz x 1
        return input_proj, atts_llama

    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        batch_size = input_ids.shape[0]
        bos = torch.ones([batch_size, 1], dtype=input_ids.dtype, device=input_ids.device) * self.resoner_tokenizer.bos_token_id  # bsz x 1
        if self.args['freeze_lm']:
            p_after_embeds = self.resoner.model.embed_tokens(input_ids).expand(batch_size, -1, -1)  # bsz x s2 x embed_dim
            bos_embeds = self.resoner.model.embed_tokens(bos)  # bsz x 1 x embed_dim
        else:
            p_after_embeds = self.resoner.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1)  # bsz x s2 x embed_dim
            bos_embeds = self.resoner.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
        if img_embeds is not None:
            p_before = '\n<|im_start|>user\n<|image_start|>'
            p_before_tokens = self.resoner_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(self.device)
            if self.args['freeze_lm']:
                p_before_embeds = self.resoner.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)  # bsz x s1 x embed_dim
            else:
                p_before_embeds = self.resoner.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)  # bsz x s1 x embed_dim
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1).to(self.device)  # bsz x (1+s1+1+s2) x embed_dim
            empty_targets = (torch.ones([batch_size, 1 + p_before_embeds.size()[1] + 1], dtype=torch.long).to(self.device).fill_(-100))  # bsz x (1 + s1)
            targets = torch.cat([empty_targets, target_ids], dim=1).to(self.device)  # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]
            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1] + 1], dtype=torch.long).to(self.device)  # bsz x (1 + s1 + 1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1).to(self.device)
            assert attention_mask.size() == targets.size()  # bsz x (1 + s1 + 1 + s2)
        else:
            p_before = '\n<|im_start|>user\n'
            p_before_tokens = self.resoner_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(self.device)
            if self.args['freeze_lm']:
                p_before_embeds = self.resoner.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)  # bsz x s1 x embed_dim
            else:
                p_before_embeds = self.resoner.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)  # bsz x s1 x embed_dim
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, p_after_embeds], dim=1).to(self.device)  # bsz x (1+s1+s2) x embed_dim
            empty_targets = (torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(self.device).fill_(-100))  # bsz x (1 + s1)
            targets = torch.cat([empty_targets, target_ids], dim=1).to(self.device)  # bsz x (1 + s1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]
            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(self.device)  # bsz x (1 + s1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1).to(self.device)
            assert attention_mask.size() == targets.size()  # bsz x (1 + s1 + s2)
        return inputs_embeds, targets, attention_mask

    def _train_with_mode(self, texts, img_embeds=None, modality='text', num_gen_tokens='8', text_hidden_fcs=None, gen_token_idx=None, text_emb_layers=None, text_prompt_embeddins=None, loss_scale=1.0, stage=2):
        if stage == 2:
            input_ids, target_ids, attention_mask = process_batch_stage_2(self.resoner_tokenizer, texts, self.max_length, num_gen_tokens, modality)
        elif stage == 3:
            input_ids, target_ids, attention_mask = process_batch_stage_3(self.resoner_tokenizer, texts, self.max_length, self.args['num_gen_img_tokens'], self.args['num_gen_video_tokens'],self.args['num_gen_audio_tokens'])
        else:
            raise NotImplementedError
        inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask)

        outputs = self.resoner(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        loss = outputs.loss
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)

        if modality == 'text':
            return loss, gen_acc, torch.zeros_like(loss)
        else:
            hidden_states = []
            # text_hidden_fcs = self.embeds_to_modality

            # based on the targets to obtain the hidden state, targets includes the [BOS] token
            start_pos = (targets == gen_token_idx[0]).nonzero(as_tuple=False)[:, 1].tolist()
            end_pos = (targets == gen_token_idx[-1]).nonzero(as_tuple=False)[:, 1].tolist()
            # logging.info(f'targets : {targets}')
            # logging.info(f'start_pos : {start_pos}')
            # logging.info(f'end_pos : {end_pos}')
            assert 0 < len(start_pos) == len(end_pos) == input_ids.size(0) and len(end_pos) > 0, (start_pos, end_pos)
            for idx, fc_layer in zip(text_emb_layers, text_hidden_fcs):
                hidden_embedding = []
                input_embedding = []
                for b, (s, e) in enumerate(zip(start_pos, end_pos)):
                    assert e - s + 1 == num_gen_tokens, (s, e)
                    hidden_embedding.append(outputs.hidden_states[idx][b, s:e + 1, :])
                    input_embedding.append(self.input_embeddings(targets[b, s:e + 1]))
                hidden_embedding = torch.stack(hidden_embedding, dim=0)
                input_embedding = torch.stack(input_embedding, dim=0)
                hidden_states.append(fc_layer(hidden_embedding, input_embedding))  # (N, seq_len, 2048)
            embeddings = torch.stack(hidden_states, dim=-1).sum(dim=-1)  # (N, 77, 768)
            # embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (N, T_I_V_A.txt, 256)

            # Obtain the embeddings produced by the text encoder of a frozen text-to-image generation model
            input_text = [conversation for conversation in texts]

            if modality == 'image':
                mse_loss = l2_loss(embeddings, torch.stack(text_prompt_embeddins, dim=0).to(self.device))
            elif modality == 'video':
                mse_loss = l2_loss(embeddings, torch.stack(text_prompt_embeddins, dim=0).to(self.device))
            else:
                text_prompt_embeddins = torch.stack(text_prompt_embeddins, dim=0).to(self.device)
                assert len(text_prompt_embeddins.shape) == 2, text_prompt_embeddins.shape
                text_prompt_embeddins = text_prompt_embeddins.view(text_prompt_embeddins.size(0), 1,
                                                                   text_prompt_embeddins.size(1))
                mse_loss = l2_loss(embeddings, text_prompt_embeddins)
            mse_loss = mse_loss.mean()
            loss += loss_scale * mse_loss

            return loss, gen_acc, mse_loss

    def _enc_align_training_stage_1(self, inputs):
        dataset_type = inputs['dataset_types'][0]
        if dataset_type == 'ImageToText':
            image_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_image(image_paths)
        elif dataset_type == 'VideoToText':
            video_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_video(video_paths)
        elif dataset_type == 'AudioToText':
            audio_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_audio(audio_paths)
        else:
            raise NotImplementedError
        input_ids, target_ids, attention_mask = process_batch_stage_1(self.resoner_tokenizer, inputs['output_texts'], self.max_length, self.args['prompt'])
        inputs_embeds, targets, attention_mask = self.prompt_wrap(mm_embeds, input_ids, target_ids, attention_mask)
        outputs = self.resoner(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True, output_hidden_states=True, labels=targets)
        loss = outputs.loss
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        return loss, gen_acc

    def _instruction_tuning_stage_3(self, inputs):
        loss = 0
        gen_acc = 0
        mse_loss = []

        dataset_type = inputs['dataset_types'][0]
        if dataset_type == 'TextToImage':
            loss, gen_acc, mse_loss = self._train_with_mode(inputs['output_texts'], None, 'image', self.args['num_gen_img_tokens'], self.embeds_to_modality, self.args['gen_img_token_idx'], self.args['text_emb_to_img_layers'], inputs['caption_embs'], stage=self.stage)
        elif dataset_type == 'TextToVideo':
            loss, gen_acc, mse_loss = self._train_with_mode(inputs['output_texts'], None, 'video', self.args['num_gen_video_tokens'], self.embeds_to_modality_video, self.args['gen_video_token_idx'], self.args['text_emb_to_video_layers'], inputs['caption_embs'], loss_scale=2, stage=self.stage)
        elif dataset_type == 'TextToAudio':
            loss, gen_acc, mse_loss = self._train_with_mode(inputs['output_texts'], None, 'audio', self.args['num_gen_audio_tokens'], self.embeds_to_modality_audio, self.args['gen_audio_token_idx'], self.args['text_emb_to_audio_layers'], inputs['caption_embs'], stage=self.stage)
        elif dataset_type == 'ImageToText':
            image_paths = inputs['mm_paths']
            img_embeds, _ = self.encode_image(image_paths)
            loss, gen_acc, _ = self._train_with_mode(inputs['output_texts'], img_embeds, modality='text', stage=self.stage)
        elif dataset_type == 'TextToText':
            loss, gen_acc, _ = self._train_with_mode(inputs['output_texts'], None, modality='text', stage=self.stage)
        else:
            raise NotImplementedError
        return loss, gen_acc, mse_loss

    def _stage_4_training(self, inputs):
        """
        In the stage 4, we employ the modality-switch dataset to instruction-tune the overall framework
        """
        pass

    def forward(self, inputs):
        loss = 0
        gen_acc = 0
        mse_loss = None

        if self.stage == 1:
            loss, gen_acc = self._enc_align_training_stage_1(inputs)
        elif self.stage == 2:
            loss, gen_acc, mse_loss = self._dec_align_training_stage_2(inputs)
        elif self.stage == 3:
            loss, gen_acc, mse_loss = self._instruction_tuning_stage_3(inputs)
        else:
            raise NotImplementedError(f"stage {self.stage} is not implemented, now it only support [1, 2, 3]")

        return loss, gen_acc, mse_loss

    def extract_multimodal_feature(self, inputs):
        features = []
        if inputs['image_paths']:
            image_embeds, _ = self.encode_image(inputs['image_paths'])
            features.append(image_embeds)
        if inputs['audio_paths']:
            audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
            features.append(audio_embeds)
        if inputs['video_paths']:
            video_embeds, _ = self.encode_video(inputs['video_paths'])
            features.append(video_embeds)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return feature_embeds

    def _prepare_image_embed(self, text, batch_size):
        pattern = r'image_start|>(.*?)<|image_end'
        matches = re.findall(pattern, text)
        features = []
        p_before_token = self.resoner_tokenizer('<|image_start|>', add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_token = self.resoner_tokenizer('<|image_end|>', add_special_tokens=False, return_tensors='pt').to(self.device)
        if self.args['freeze_lm']:
            p_before_embeds = self.resoner.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1, -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.resoner.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1, -1)  # bsz x s2 x embed_dim
        else:
            p_before_embeds = self.resoner.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1, -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.resoner.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1, -1)  # bsz x s2 x embed_dim
        for m in matches:
            print('image path: ', m)
            if m.startswith('temp'):
                m = os.path.join('./', m)
                print('image path: ', m)
            _temp_embedding, _ = self.encode_image([m])
            features.append(_temp_embedding)
        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)

    def _prepare_video_embed(self, text, batch_size):
        pattern = r'video_start|>(.*?)<|video_end'
        matches = re.findall(pattern, text)
        features = []
        p_before_token = self.resoner_tokenizer('<|video_start|>', add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_token = self.resoner_tokenizer('<|video_end|>', add_special_tokens=False, return_tensors='pt').to(self.device)
        if self.args['freeze_lm']:
            p_before_embeds = self.resoner.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.resoner.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1, -1)
        else:
            p_before_embeds = self.resoner.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.resoner.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1, -1)
        for m in matches:
            print('Video path: ', m)
            if m.startswith('temp'):
                m = os.path.join('./', m)
                print('Video path: ', m)
            _temp_embedding, _ = self.encode_video([m])
            features.append(_temp_embedding)
        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)

    def _prepare_audio_embed(self, text, batch_size):
        pattern = r'audio_start|>(.*?)<|audio_end'
        matches = re.findall(pattern, text)
        features = []
        p_before_token = self.resoner_tokenizer('<|audio_start|>', add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_token = self.resoner_tokenizer('<|audio_end|>', add_special_tokens=False, return_tensors='pt').to(self.device)
        if self.args['freeze_lm']:
            p_before_embeds = self.resoner.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.resoner.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1, -1)
        else:
            p_before_embeds = self.resoner.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.resoner.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1, -1)
        for m in matches:
            print('Audio path: ', m)
            if m.startswith('temp'):
                m = os.path.join('./', m)
                print('Video path: ', m)
            _temp_embedding, _ = self.encode_audio([m])
            features.append(_temp_embedding)
        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)

    def prepare_generation_embedding(self, inputs):
        prompt = inputs['prompt']
        text = prompt + '\n<|im_start|>assistant\n'
        print("text prompt: ", text)
        batch_size = 1
        input_embeds = []
        split_text = re.split(r' <|> ', text)
        for st in split_text:
            if st.startswith('image_start|>'):
                input_embeds.append(self._prepare_image_embed(st, batch_size))
            elif st.startswith('audio_start|'):
                input_embeds.append(self._prepare_audio_embed(st, batch_size))
            elif st.startswith('video_start|'):
                input_embeds.append(self._prepare_video_embed(st, batch_size))
            else:
                text_tokens = self.resoner_tokenizer(st, add_special_tokens=False, return_tensors='pt').to(self.device)
                bos = torch.ones([batch_size, 1], dtype=text_tokens.input_ids.dtype, device=text_tokens.input_ids.device) * self.resoner_tokenizer.bos_token_id  # bsz x 1
                if self.args['freeze_lm']:
                    text_embeds = self.resoner.model.embed_tokens(text_tokens.input_ids).expand(batch_size, -1, -1)
                    bos_embeds = self.resoner.model.embed_tokens(bos)  # bsz x 1 x embed_dim
                else:
                    text_embeds = self.resoner.model.model.embed_tokens(text_tokens.input_ids).expand(batch_size, -1, -1)
                    bos_embeds = self.resoner.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
                input_embeds.append(bos_embeds)
                input_embeds.append(text_embeds)
        inputs_embeds = torch.cat(input_embeds, dim=1)  # bsz x (1+s2) x embed_dim
        return inputs_embeds

    def generate_tokens_embeddings(self, inputs, input_embeds, temperature: float = 0.0, top_p: float = 1.0):
        """
        This function is used to generate the tokens and output embeddings
        inputs: dict
        input_embeds: tensor
        return:
            out: the output tokens index
            output_embeddings: output embeddings for synthesizing images
            video_output_embedding: output embeddings for synthesizing video
            audio_output_embedding: output embeddings for synthesizing audio
        """
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=inputs['stops_id'], encounters=1)])

        outputs = self.resoner.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            # repeat_pen,
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_attentions=True
        )
        output_embeddings = []
        out = outputs.sequences
        output_embeddings = torch.cat(output_embeddings, dim=1)
        return out, output_embeddings


    def generate(self, inputs):
        """
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'thermal_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature, Used to modulate logit distribution.
                'modality_embeds': None or torch.tensor,
                'modality_cache': save the image cache,

                'filter_value': Value to assign to tokens that should never be generated,
                'min_word_tokens': Minimum number of words to generate before allowing a <|image_start|> output.
                'gen_scale_factor': float = 1.0,
                'stops_id': the default value is [[835], [2277, 29937]] the stop token is '###', which has two types of tokenization ways, [835] and [2277, 29937]
                'ENCOUNTERS': the times that the generated sentence will be ended.

                'load_sd': whether use SD for image generation
                'max_num_imgs': Maximum number of images to return in one generation pass.
                'guidance_scale_for_img': the guidance ratio of conditioner, if it is None, the default value will be applied in SD
                'num_inference_steps_for_img': the number of inference step for image generation in the stable diffusion model

                'load_vd': whether use VD for video generation
                'max_num_vids': Maximum number of videos to return in one generation pass.
                'guidance_scale_for_vid': the guidance ratio of conditioner, if it is None, the default value will be applied in VD
                'num_inference_steps_for_vid': the number of inference step for video generation in the stable diffusion model
                'height': (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                        The height in pixels of the generated video.
                'width': (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                        The width in pixels of the generated video.
                'num_frames': (`int`, *optional*, defaults to 16):
                        The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
                        amounts to 2 seconds of video.

                'load_ad': whether use AD for audio generation
                'max_num_auds': Maximum number of audios to return in one generation pass.
                'guidance_scale_for_aud': the guidance ratio of conditioner, if it is None, the default value will be applied in AD
                'num_inference_steps_for_aud': the number of inference step for audio generation in the stable diffusion model
                'audio_length_in_s': the seconds for generated audio length
            }
        """
        input_embeds = self.prepare_generation_embedding(inputs)
        generated_ids, generated_image_embeddings, generated_video_embeddings, generated_audio_embeddings = self.generate_tokens_embeddings(inputs, input_embeds)

        return_outputs = []

        all_gen_img_idx = [i for i, x in enumerate(generated_ids[0, :] == self.args['gen_img_token_idx'][0]) if x][:inputs['max_num_imgs']]
        print('all_gen_img_idx: ', all_gen_img_idx)

        all_gen_vid_idx = [i for i, x in enumerate(generated_ids[0, :] == self.args['gen_video_token_idx'][0]) if x][:inputs['max_num_vids']]
        print('all_gen_vid_idx: ', all_gen_vid_idx)

        all_gen_aud_idx = [i for i, x in enumerate(generated_ids[0, :] == self.args['gen_audio_token_idx'][0]) if x][:inputs['max_num_auds']]
        print('all_gen_aud_idx: ', all_gen_aud_idx)

        if len(all_gen_img_idx) == 0 and len(all_gen_vid_idx) == 0 and len(all_gen_aud_idx) == 0:
            caption = self.resoner_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return_outputs.append(caption)
        else:
            if len(all_gen_img_idx) > 0:
                img_outputs = self.generate_images(generated_ids, generated_image_embeddings, all_gen_img_idx, None, guidance_scale=inputs['guidance_scale_for_img'], num_inference_steps=inputs['num_inference_steps_for_img'])
                return_outputs.append({'img': img_outputs})
            if len(all_gen_vid_idx) > 0:
                vid_outputs = self.generate_videos(generated_ids, generated_video_embeddings, all_gen_vid_idx, None, guidance_scale=inputs['guidance_scale_for_vid'], num_inference_steps=inputs['num_inference_steps_for_vid'], height=inputs['height'], width=inputs['width'], num_frames=inputs['num_frames'])
                return_outputs.append({'vid': vid_outputs})
            if len(all_gen_aud_idx) > 0:
                aud_outputs = self.generate_audios(generated_ids, generated_audio_embeddings, all_gen_aud_idx, None, guidance_scale=inputs['guidance_scale_for_aud'], num_inference_steps=inputs['num_inference_steps_for_aud'], audio_length_in_s=inputs['audio_length_in_s'])
                return_outputs.append({'aud': aud_outputs})

        return return_outputs
