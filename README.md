# DR-MetaLearning
The reproduction code for the "Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning". Unofficial.

Sorry, I cannot successfully implement the MetaLearning step. So the repository only contains the code for Domain Randomization. 

The problem with MetaLearning is that it need to manually implement every layer's parameters of ResNet-18, which I think is poor in generalization. If you want to implement, you can refer to the pytorch code for MAML, which is highly similar. 

I also find that there are some python packages that can help with this issue. Like some packages designed for Meta-Learning. But I haven't try them.
