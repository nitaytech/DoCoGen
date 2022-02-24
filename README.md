# DoCoGen

### Official code repository for the  paper ["DoCoGen: Domain Counterfactual Generation for Low Resource Domain Adaptation"](...) (ACL'2022)
 
DoCoGen is a controllable generation for generating domain-counterfactual textual examples (D-con). Given an input text example, DoCoGen generates its D-con --  that is similar to the original in all aspects, including the task label, but its domain is changed to a desired one. 
DoCoGen intervenes on the domain-specific terms of its input example, replacing them with terms that are relevant for its target domain while keeping all other properties fixed, including the task label.
It is trained using only unlabeled examples from multiple domains -- no NLP task labels or pairs of textual examples and their domain-counterfactuals are required.

![DoCoGen](figures/docogen_paper.PNG)

If you use this code please cite our paper (see [recommended](#docogen_citation) citation below).

Our code is implemented in [PyTorch](https://pytorch.org/), using the [Transformers](https://github.com/huggingface/transformers) and [PyTorch-Lightning](https://www.pytorchlightning.ai/) libraries. 


## How to Cite DoCoGen
<a name="docogen_citation"/>
```

```
