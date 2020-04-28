from typing import Dict, Iterable, Optional
import argparse
import logging
import math
import os
import re
import shutil
import json

import torch

from allennlp.training.util import evaluate
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import Params
from allennlp.common.util import (
    prepare_environment,
    prepare_global_logging,
    get_frozen_and_tunable_parameter_names,
    dump_metrics,
    import_submodules,
)
from allennlp.data import Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.util import create_serialization_dir
import tempfile
from tempfile import TemporaryDirectory

from discrete_al_coref_module.models.ensemble_coref import CorefEnsemble
from discrete_al_coref_module.training.al_trainer import ALCorefTrainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

torch.manual_seed(1)


def datasets_from_params(params: Params) -> Dict[str, Iterable[Instance]]:
    """
    Load all the datasets specified by the config.
    """
    fully_labelled_threshold = 3000 if 'fully_labelled_threshold' not in params['dataset_reader'] else params['dataset_reader']['fully_labelled_threshold']
    dataset_reader = DatasetReader.from_params(params.pop("dataset_reader", None))
    validation_dataset_reader_params = params.pop("validation_dataset_reader", None)

    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)

    # Split train data into held out/not held out, initializing to 10% non-held-out
    # non-held-out training data will have 100% of labels (using dataset_reader)
    # held-out training data will have only 50% of labels (using held_out_dataset_reader)
    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    num_saved_labels = fully_labelled_threshold
    held_out_train_data = train_data[num_saved_labels:]     # after threshold
    train_data = train_data[:num_saved_labels]      # before threshold

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data, "held_out_train": held_out_train_data}

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets["test"] = test_data

    return datasets


def train_model(params: Params,
                serialization_dir: str,
                selector: str,
                num_ensemble_models: Optional[int],
                file_friendly_logging: bool = False,
                recover: bool = False,
                force: bool = False) -> Model:
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results in ``serialization_dir``.

    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.

    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    prepare_environment(params)

    create_serialization_dir(params, serialization_dir, recover, force)
    prepare_global_logging(serialization_dir, file_friendly_logging)

    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    if isinstance(cuda_device, list):
        for device in cuda_device:
            check_for_gpu(device)
    else:
        check_for_gpu(cuda_device)

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    all_datasets = datasets_from_params(params)
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("From dataset instances, %s will be considered for vocabulary creation.",
                ", ".join(datasets_for_vocab_creation))
    vocab = Vocabulary.from_params(
            params.pop("vocabulary", {}),
            (instance for key, dataset in all_datasets.items()
             for instance in dataset
             if key in datasets_for_vocab_creation)
    )

    model_params = params.pop('model')
    if selector == 'qbc':
        assert num_ensemble_models is not None
        models_list = [Model.from_params(vocab=vocab, params=model_params.duplicate()) for i in range(num_ensemble_models)]
        ensemble_model = CorefEnsemble(models_list)
        model = ensemble_model.submodels[0]
    else:
        model = Model.from_params(vocab=vocab, params=model_params)
        ensemble_model = None

    # Initializing the model can have side effect of expanding the vocabulary
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)
    validation_iterator_params = params.pop("validation_iterator", None)
    if validation_iterator_params:
        validation_iterator = DataIterator.from_params(validation_iterator_params)
        validation_iterator.index_with(vocab)
    else:
        validation_iterator = None
    held_out_iterator_params = params.pop("held_out_iterator", None)
    if held_out_iterator_params:
        held_out_iterator = DataIterator.from_params(held_out_iterator_params)
        held_out_iterator.index_with(vocab)
    else:
        held_out_iterator = None

    train_data = all_datasets['train']
    held_out_train_data = all_datasets.get('held_out_train')
    validation_data = all_datasets.get('validation')
    test_data = all_datasets.get('test')

    trainer_params = params.pop("trainer")
    no_grad_regexes = trainer_params.pop("no_grad", ())
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    frozen_parameter_names, tunable_parameter_names = \
                   get_frozen_and_tunable_parameter_names(model)
    logger.info("Following parameters are Frozen  (without gradient):")
    for name in frozen_parameter_names:
        logger.info(name)
    logger.info("Following parameters are Tunable (with gradient):")
    for name in tunable_parameter_names:
        logger.info(name)

    trainer_choice = trainer_params.pop("type")
    trainer = ALCorefTrainer.by_name(trainer_choice).from_params(model=model,
                                                                serialization_dir=serialization_dir,
                                                                iterator=iterator,
                                                                train_data=train_data,
                                                                held_out_train_data=held_out_train_data,
                                                                validation_data=validation_data,
                                                                params=trainer_params,
                                                                validation_iterator=validation_iterator,
                                                                held_out_iterator=held_out_iterator,
                                                                ensemble_model=ensemble_model)
    evaluate_on_test = params.pop_bool("evaluate_on_test", False)
    params.assert_empty('base train command')

    try:
        metrics, query_info = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)

    best_model = None
    logger.info("Loading the best epoch weights.")
    best_model_state_path = os.path.join(serialization_dir, 'best.th')
    best_model_state = torch.load(best_model_state_path)
    best_model = model
    best_model.load_state_dict(best_model_state)
    
    if test_data and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(
            best_model, test_data, validation_iterator or iterator,
            cuda_device=trainer._cuda_devices[0],
            batch_weight_key="",
        )
        for key, value in test_metrics.items():
            metrics["test_" + key] = value
    return best_model, metrics, query_info


def main(args):
    # validate inputs
    num_ensemble_models = None
    selector = args.selector
    if selector[:3] == 'qbc':
        assert (len(selector) > 3)
        num_ensemble_models = int(selector[3:])
        selector = 'qbc'
    assert(selector == 'entropy' or selector == 'score' or selector == 'random' or selector == 'qbc')
    # 1 and only 1 specified
    assert getattr(args, 'labels_to_query', None) or getattr(args, 'query_time_file', None)
    assert not getattr(args, 'labels_to_query', None) or not getattr(args, 'query_time_file', None)

    # parse inputs
    if getattr(args, 'labels_to_query', None):
        label_times_list = args.labels_to_query.split(",")
    else:
        label_times_list = args.query_time_file.split(":")

    # import submodule
    import_submodules('discrete_al_coref_module')

    if getattr(args, 'experiments', None):
        '''
        Default (experimental) mode
        '''
        # create save dir
        save_dir = args.experiments
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        for x in label_times_list:
            if getattr(args, 'labels_to_query', None):
                x = int(x)
                assert x >= 0
                print("Running with {} labels per doc".format(x))
                serialization_dir = os.path.join(save_dir, "checkpoint_{}".format(x))
            else:
                assert os.path.exists(x)
                print("Running with equivalent annotation time to {}".format(x))
                save_fn = x.replace('/', '%').replace('_query_info.json', '').replace(
                    '.json', '').replace('.', '')
                serialization_dir = os.path.join(save_dir, "checkpoint_{}".format(save_fn))

            print("Saving in directory: {}".format(serialization_dir))
            if os.path.exists(serialization_dir):
                print("Deleting existing directory found in same location.")
                shutil.rmtree(serialization_dir)

            # modify parameters according to passed-in arguments
            params = Params.from_file("training_config/coref.jsonnet")
            params.params['trainer']['cuda_device'] = args.cuda_device
            params.params['trainer']['active_learning']['save_al_queries'] = args.save_al_queries
            params.params['trainer']['active_learning']['query_type'] = "pairwise" if args.pairwise else "discrete"
            if selector:
                params.params['trainer']['active_learning']['selector']['type'] = selector
            params.params['trainer']['active_learning']['selector']['use_clusters'] = not args.no_clusters
            if getattr(args, 'labels_to_query', None):
                params.params['trainer']['active_learning']['num_labels'] = x
            else:
                params.params['trainer']['active_learning']['use_equal_annot_time'] = True
                params.params['trainer']['active_learning']['equal_annot_time_file'] = x

            # train model
            best_model, metrics, query_info = train_model(params, serialization_dir, selector, num_ensemble_models, recover=False)
            dump_metrics(os.path.join(save_dir, "{}.json".format(x)), metrics, log=True)
            with open(os.path.join(save_dir, "{}_query_info.json".format(x)), 'w', encoding='utf-8') as f:
                json.dump(query_info, f)
    else:
        '''
        Test mode
        '''
        params = Params.from_file('training_config/coref.jsonnet')
        if getattr(args, 'labels_to_query', None):
            params.params['trainer']['active_learning']['num_labels'] = label_times_list[0]
        else:
            params.params['trainer']['active_learning']['use_equal_annot_time'] = True
            params.params['trainer']['active_learning']['equal_annot_time_file'] = label_times_list[0]
        params.params['trainer']['active_learning']['save_al_queries'] = args.save_al_queries
        if getattr(args, 'testing', None) or getattr(args, 'testing_vocab', None):
            params.params['trainer']['active_learning']['epoch_interval'] = 0
            del params.params['test_data_path']
            ''' Uncomment if necessary
            params.params['train_data_path'] = "/checkpoint/belindali/active_learning_coref/coref_ontonotes/dev.english.v4_gold_conll"
            params.params['dataset_reader']['fully_labelled_threshold'] = 100
            #'''
            if getattr(args, 'testing', None):
                params.params['model']['text_field_embedder']['token_embedders']['tokens'] = {'type': 'embedding', 'embedding_dim': 300}
        with TemporaryDirectory() as serialization_dir:
            print("temp file path: " + str(serialization_dir))
            params.params['trainer']['cuda_device'] = args.cuda_device
            params.params['trainer']['active_learning']['query_type'] = "pairwise" if args.pairwise else "discrete"
            params.params['trainer']['active_learning']['selector']['type'] = selector if selector else "entropy"
            params.params['trainer']['active_learning']['selector']['use_clusters'] = not args.no_clusters
            best_model, metrics, query_info = train_model(params, serialization_dir, selector, num_ensemble_models)
            with open(os.path.join(serialization_dir, "query_info.json"), 'w', encoding='utf-8') as f:
                json.dump(query_info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run setting')
    parser.add_argument('cuda_device', type=int,
                        help='which cuda device to run on')
    parser.add_argument('-t', '--testing',
                        action='store_true',
                        default=False,
                        help='run testing configuration')
    parser.add_argument('-tv', '--testing_vocab',
                        action='store_true',
                        default=False,
                        help='run testing configuration, but with pretrained embeddings')
    parser.add_argument('-e', '--experiments',
                        type=str,
                        help='file to store results of x% of labels experiments')
    parser.add_argument('-p', '--pairwise',
                        action='store_true',
                        default=False,
                        help='run pairwise querying')
    parser.add_argument('-nc', '--no-clusters',
                        action='store_true',
                        default=False,
                        help='run non-clustering selectors')
    parser.add_argument('-s', '--selector',
                        type=str,
                        default='entropy',
                        help='what type of selector to use')
    parser.add_argument('--labels_to_query',
                        type=str,
                        required=False,
                        help='labels to query per doc (n >= 0). Can also pass in a comment-separated list to run experiments one after the other.')
    parser.add_argument('--query_time_file',
                        type=str,
                        required=False,
                        help='specify path to a \'*_query_info\' file here to run in the same time as that saved experiment')
    parser.add_argument("--save_al_queries",
                        action='store_true',
                        required=False,
                        help='Whether or not to save AL queries (or just simulate them using user inputs)')
   
    args = parser.parse_args()

    main(args)
