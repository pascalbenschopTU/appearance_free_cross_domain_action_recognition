This folder keeps only the report-ready text-embedding artifacts:

- `sentence_transformer_embeddings.npy`
- `clip_embeddings.npy`
- `sentence_transformer_tsne.pdf`
- `clip_tsne.pdf`

Generate or refresh these files with:

```bash
python debug/debug_embeddings.py \
  tc-clip/labels/labels_froster/kinetics_400_llm_labels.json \
  tc-clip/labels/labels_froster/ucf_101_llm_labels.json \
  tc-clip/labels/labels_froster/hmdb_51_llm_labels.json \
  --output-dir artifacts/text_embeddings
```

The script intentionally avoids writing extra metadata/index/by-dataset artifacts.
