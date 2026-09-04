GNN to MLP distillation for graph free and faster inference.

data source: https://data.dgl.ai/dataset/amazon_co_buy_computer.zip

the teacher model is GCN(graph); student is MLP (features only).

method:

- $H_t, H_s = entropy(softmax) / max$ (normalized teacher / student entropy)
- $sim = cosine(p_t[src], p_t[dst]) \in [0, 1]$
- $p = 1 - \exp\left(-\alpha \cdot sim \cdot \frac{\sqrt{H_t[src] H_s[src]}}{H_t[dst]}\right)$, $\alpha = 10 / 100^{(epoch-1)/499}$ (annealed $10 \to 0.1$)
- sample edges $\sim Bernoulli(p)$ + self-loops, $w = p \cdot b$, $b \sim Beta(0.5, 0.5)$
- $mix = (1-w)\cdot softmax(t[src]/\tau) + w\cdot softmax(t[dst]/\tau)$, $\tau = 0.9$
- $loss = \lambda CE + (1-\lambda) KL(student/\tau \parallel mix)$, $\lambda = 0.2$

results:

| Model | Test | Graph? |
|-------|------|--------|
| Teacher GCN | 0.8232 | yes |
| Vanilla MLP | 0.6650 | no |
| Distilled MLP | 0.8216 | no `+15.6 vs vanilla` |