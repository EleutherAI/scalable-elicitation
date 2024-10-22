# Balancing Label Quantity and Quality for Scalable Elicitation
Scalable oversight studies methods of training and evaluating AI systems in domains where human judgment is unreliable or expensive, such as scientific research and software engineering in complex codebases. Most work in this area has focused on methods of improving the quality of labels. Recent work by Burns et al. (2023) considers the complementary problem of training models with low-quality labels, finding that large pretrained models often have an inductive bias towards producing correct answers. In practice, however, neither label quantity nor quality is fixed: practitioners face a quantity-quality tradeoff. In this paper, we explore the microeconomics of the quantity-quality tradeoff on binary NLP classification tasks used in Burns et al. (2023). While sample-efficient learning has been studied extensively, little public research has focused on scalable elicitation: eliciting capabilities from pretrained models subject to labeling cost constraints. We find that this setting has novel dynamics caused by the tradeoff between label quantity and quality, as well as the model's existing latent capabilities. We observe three regimes of eliciting classification knowledge from pretrained models using supervised finetuning: quantity-dominant, quality-dominant, and a mixed regime involving the use of low- and high-quality data together to attain higher accuracy at a lower cost than using either alone. We explore sample-efficient elicitation methods that make use of two datasets of differing qualities, and establish a Pareto frontier of scalable elicitation methods that optimally trade off labeling cost and classifier performance. We find that the accuracy of supervised fine-tuning can be improved by up to 5 percentage points at a fixed labeling budget by adding a few-shot prompt to make use of the model's existing knowledge of the task.

### Links
- Arxiv [paper](https://arxiv.org/abs/2410.13215)
- Twitter/X [thread](https://x.com/alextmallen/status/1848782532718039057)
