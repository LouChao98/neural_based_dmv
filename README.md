# Neural Based DMV

 Python reimplementations of a series of neural extensions of the Dependency Model with Valence (DMV) for unsupervised dependency parsing.

Due to the difference in the framework, dataset processing methods and model implementation, we did not use the parameter settings in the paper and results are for reference only.

- [x] NDMV: [Unsupervised Neural Dependency Parsing](https://www.aclweb.org/anthology/D16-1073/)
- [ ] L-NDMV: [Dependency Grammar Induction with Neural Lexicalization and Big Training Data](https://www.aclweb.org/anthology/D17-1176/)
- [ ] D-NDMV: [Enhancing Unsupervised Generative Dependency Parser with Contextual Information](https://www.aclweb.org/anthology/P19-1526/)

## Requirements

```
python==3.7
pytorch==1.4
cupy-cuda100==7.2.0
```
A GPU is required.

## NDMV

| model                    | result | in paper | configure file      |
| ------------------------ | :----: | :------: | ------------------- |
| Neural DMV (Standard EM) |  55.4  |   51.3   | ndmv_em_1.json      |
| Neural DMV (Viterbi EM)  |  65.1  |   65.9   | ndmv_viterbi_1.json |
| Neural E-DMV             |  69.0  |   69.7   | ndmv_viterbi_2.json |
| Neural E-DMV (Good Init) |   -    |   72.5   | -                   |

### HOW TO RUN
1. Prepare your dataset. CoNLL format are expected and train, dev and test are splitted.
2. Modify `train_ds`, `dev_ds` and `test_ds` fields in the configure file.
3. run command `python model/ndmv.py --load_option <path_to_configure_file>`.

You can find the description of each field of the configuration file in the subclass of Options, which named as `XXXXOptions`.

If get an ModuleNotFoundError for `utils` or `module`, you need set `PYTHONPATH` to include the project`s root.

## L-NDMV

