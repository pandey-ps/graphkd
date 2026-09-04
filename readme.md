GNN to MLP distillation for graph free and faster inference.

data source: https://data.dgl.ai/dataset/amazon_co_buy_computer.zip

the teacher model is GCN(graph); student is MLP (features only).

results:

| Model | Test | Graph? |
|-------|------|--------|
| Teacher GCN | 0.8232 | yes |
| Vanilla MLP | 0.6650 | no |
| Distilled MLP | 0.8216 | no `+15.6 vs vanilla` |