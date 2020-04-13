# active-learning-coref
Active learning for coreference resolution

### Setup

### Usage
To train with active learning, run
```
python train.py 0 -e {save_dir} -s {entropy/qbc/score/random} --labels_to_query {num_labels_per_doc} [--save_al_queries]
``` 

To run pairwise selection
```
python train.py 0 -e {save_dir} -s {entropy/qbc/score/random} --labels_to_query {num_labels_per_doc} -p [--save_al_queries]
```

To run the selectors without clustering
```
python train.py 0 -e {save_dir} -s {entropy/qbc/score/random} --labels_to_query {num_labels_per_doc} -nc [--save_al_queries]
```
