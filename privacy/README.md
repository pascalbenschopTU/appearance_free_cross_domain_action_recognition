## Privacy

This directory contains the privacy/domain-adaptation training entrypoints that are still part of the active project surface.

Direct entrypoints:

```bash
python models/appearance_free_cross_domain_action_recognition/privacy/train_domain_adaptation.py
python models/appearance_free_cross_domain_action_recognition/privacy/train_domain_adaptation_rgb.py
python models/appearance_free_cross_domain_action_recognition/privacy/train_pa_hmdb51_privacy_cv.py
python models/appearance_free_cross_domain_action_recognition/privacy/train_pa_hmdb51_vit_attacker.py
```

Compatibility notes:

- `train_hmdb51_privacy_cv.py` forwards to the dedicated `PA-HMDB51` trainer.
- Local STPrivacy shell wrappers were retired from this checkout.
- Outputs are written under `privacy/out/...` by default.
