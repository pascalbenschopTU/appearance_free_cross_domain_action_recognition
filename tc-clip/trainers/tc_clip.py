"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn

from trainers.tc_clip_text_encoder import VPTextEncoder
from trainers.tc_clip_prompt_learner import VPPromptLearner


class TCCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        self.prompt_learner = VPPromptLearner(cfg, classnames, clip_model, logger)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = VPTextEncoder(cfg, clip_model, logger)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype if cfg.opt_level != 'O0' else torch.float32
        self.prompt_generation_layer_level = self.text_encoder.prompt_generation_layer_level
        self.return_layer_num = self.prompt_generation_layer_level.copy()
        if 11 not in self.return_layer_num:
            self.return_layer_num.append(11)
        self.text_group_indices = None
        self.text_group_weights = None
        logger.info(f"Using context tokens from vision layer {self.return_layer_num}")

    def _rebuild_classnames(self, cfg, classnames, clip_model, logger):
        self.prompt_learner._rebuild_classnames(cfg, classnames, clip_model, logger)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    def _set_text_aggregation(self, group_indices=None, group_weights=None):
        self.text_group_indices = group_indices
        self.text_group_weights = group_weights

    def _aggregate_text_features(self, text_features: torch.Tensor) -> torch.Tensor:
        if self.text_group_indices is None or self.text_group_weights is None:
            return text_features

        grouped_features = []
        for class_indices, class_weights in zip(self.text_group_indices, self.text_group_weights):
            index_tensor = torch.as_tensor(class_indices, device=text_features.device, dtype=torch.long)
            weight_tensor = torch.as_tensor(class_weights, device=text_features.device, dtype=text_features.dtype)
            prompt_features = torch.index_select(text_features, dim=0, index=index_tensor)
            class_feature = (prompt_features * weight_tensor.unsqueeze(-1)).sum(dim=0)
            class_feature = class_feature / class_feature.norm(dim=-1, keepdim=True)
            grouped_features.append(class_feature)

        return torch.stack(grouped_features, dim=0)

    def forward(self, image, return_attention=False, return_source=False):
        tokenized_prompts = self.tokenized_prompts  # (num_classes, token_len)
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner()

        # Encode visual features
        image_features, context_tokens, attn, source = self.image_encoder(image.type(self.dtype),
                                                                          return_layer_num=self.return_layer_num,
                                                                          return_attention=return_attention,
                                                                          return_source=return_source)

        # Now take the mean along the temporal direction with last layer cls tokens
        image_features_mean = image_features[:, -1, ...].mean(dim=1, keepdim=False)
        image_features_mean = image_features_mean / image_features_mean.norm(dim=-1, keepdim=True)  # [b, 512]

        # Instance-conditional prompts
        logits = []
        context_tokens = context_tokens[:, :len(self.prompt_generation_layer_level)]
        for i in range(context_tokens.size(0)): # batch iteration
            text_features = self.text_encoder(prompts=prompts,
                                              tokenized_prompts=tokenized_prompts,
                                              im_features=context_tokens[i, ...])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = self._aggregate_text_features(text_features)
            l_i = logit_scale * image_features_mean[i] @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)    # [b, n_cls]

        return {"logits": logits,
                "attention": attn,
                "source": source}
