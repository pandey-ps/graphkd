GNN to MLP distillation for graph free and faster inference.

data source: https://data.dgl.ai/dataset/amazon_co_buy_computer.zip

the teacher model is GCN(graph); student is MLP (features only).

method:

- $H = \operatorname{entropy}(\operatorname{softmax})$
- $sim = \operatorname{cosine}$
- $p = 1 - \exp\left(-\alpha \cdot sim \cdot \frac{\sqrt{H_t H_s}}{H_t[\mathrm{dst}]}\right)$, $\alpha = 10$
- $mix = (1-w)\cdot p_t[\mathrm{src}] + w\cdot p_t[\mathrm{dst}]$, $w \sim \operatorname{Beta}(0.5)$
- $\tau = 0.9$
- $KL + \lambda CE$, $\lambda = 0.2$

results:

| Model | Test | Graph? |
|-------|------|--------|
| Teacher GCN | 0.8232 | yes |
| Vanilla MLP | 0.6650 | no |
| Distilled MLP | 0.8216 | no `+15.6 vs vanilla` |