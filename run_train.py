from typing import Dict, Iterable, Optional
import argparse
import logging
import math
import os
import re
import shutil
import json

import torch

from allennlp.commands.evaluate import evaluate
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
    saved_data_file = params['dataset_reader']['saved_data_file'] if 'saved_data_file' in params['dataset_reader'] else None
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
    if saved_data_file is not None:
        num_saved_labels = len(torch.load(saved_data_file))
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


def create_serialization_dir(
        params: Params,
        serialization_dir: str,
        recover: bool,
        force: bool) -> None:
    """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.

    Parameters
    ----------
    params: ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: ``str``
        The directory in which to save results and logs.
    recover: ``bool``
        If ``True``, we will try to recover from an existing serialization directory, and crash if
        the directory doesn't exist, or doesn't match the configuration we're given.
    """
    if recover and force:
        raise ConfigurationError("Illegal arguments: both force and recover are true.")

    if os.path.exists(serialization_dir) and force:
        shutil.rmtree(serialization_dir)

    if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
        if not recover:
            raise ConfigurationError(f"Serialization directory ({serialization_dir}) already exists and is "
                                     f"not empty. Specify --recover to recover training from existing output.")

        logger.info(f"Recovering from prior training at {serialization_dir}.")

        recovered_config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError("The serialization directory already exists but doesn't "
                                     "contain a config.json. You probably gave the wrong directory.")
        else:
            loaded_params = Params.from_file(recovered_config_file)

            # Check whether any of the training configuration differs from the configuration we are
            # resuming.  If so, warn the user that training may fail.
            fail = False
            flat_params = params.as_flat_dict()
            flat_loaded = loaded_params.as_flat_dict()
            for key in flat_params.keys() - flat_loaded.keys():
                logger.error(f"Key '{key}' found in training configuration but not in the serialization "
                             f"directory we're recovering from.")
                fail = True
            for key in flat_loaded.keys() - flat_params.keys():
                logger.error(f"Key '{key}' found in the serialization directory we're recovering from "
                             f"but not in the training config.")
                fail = True
            for key in flat_params.keys():
                if flat_params.get(key, None) != flat_loaded.get(key, None):
                    logger.error(f"Value for '{key}' in training configuration does not match that the value in "
                                 f"the serialization directory we're recovering from: "
                                 f"{flat_params[key]} != {flat_loaded[key]}")
                    fail = True
            if fail:
                raise ConfigurationError("Training configuration does not match the configuration we're "
                                         "recovering from.")
    else:
        if recover:
            raise ConfigurationError(f"--recover specified but serialization_dir ({serialization_dir}) "
                                     "does not exist.  There is nothing to recover from.")
        os.makedirs(serialization_dir, exist_ok=True)


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
    # spans_per_word = params['model']['spans_per_word']
    # total_top_spans = 0
    # for instance in all_datasets['held_out_train']:
    #     total_top_spans += int(math.floor(spans_per_word * len(instance['text'])))
    # print(total_top_spans / len(all_datasets['held_out_train']))
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
                cuda_device=trainer._cuda_devices[0] # pylint: disable=protected-access
        )
        for key, value in test_metrics.items():
            metrics["test_" + key] = value
    '''
    elif test_data:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)

    '''
    return best_model, metrics, query_info

# In practice you'd probably do this from the command line:
#   $ allennlp train tutorials/tagger/experiment.jsonnet -s /tmp/serialization_dir
#
def main(cuda_device, testing=False, testing_vocab=False, experiments=None, pairwise=False, selector='entropy', num_ensemble_models=None,
         no_clusters=False, args=None):
    import_submodules('discrete_al_coref_module')
    assert(selector == 'entropy' or selector == 'score' or selector == 'random' or selector == 'qbc')
    if hasattr(args, 'labels_to_query'):
        assert args.labels_to_query >= 0
    use_percents=False
    if not hasattr(args, 'labels_to_query'):
        if cuda_device == 0:
            #percent_list = [200, 180, 160]
            percent_list = [0]#[120, 100, 80]
        if cuda_device == 1:
            #percent_list = [0, 40, 140, 100]
            percent_list = [200]#[60, 40, 140]
        if cuda_device == 2:
            #percent_list = [20, 120, 60, 80]
            percent_list = [20]#[180, 160, 20]
    else:
        percent_list = [args.labels_to_query]
    # if selector == 'qbc':
    #     cuda_device = [(cuda_device + i) % 3 for i in range(3)]
    #     os.system('rm -rf active_learning_model_states_ensemble_' + str(cuda_device))
    # else:
    #     cuda_device = cuda_device
    # ''' Make training happen
    if experiments:
        save_dir = experiments
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        os.system('cp training_config/coref.jsonnet ' + os.path.join(save_dir, 'coref.jsonnet'))
        for x in percent_list:
            print_str = "% of labels" if use_percents else " labels per doc"
            print("Running with " + str(x) + print_str)
            serialization_dir = os.path.join(save_dir, "checkpoint_" + str(x))
            os.system('rm -rf ' + serialization_dir)
            params = Params.from_file(os.path.join(save_dir, 'coref.jsonnet'))
            # restore data file
            saved_data_file = '../data/saved_data_' + str(selector) + '_' + str(num_ensemble_models) + '_' + str(x) + '.th'
            if os.path.exists(saved_data_file):
                params['dataset_reader']['saved_data_file'] = saved_data_file
            params.params['trainer']['cuda_device'] = cuda_device
            params.params['trainer']['active_learning']['save_al_queries'] = args.save_al_queries
            params.params['trainer']['active_learning']['query_type'] = "pairwise" if pairwise else "discrete"
            if selector:
                params.params['trainer']['active_learning']['selector']['type'] = selector
            params.params['trainer']['active_learning']['selector']['use_clusters'] = not no_clusters
            params.params['trainer']['active_learning']['use_percent'] = use_percents
            params.params['trainer']['active_learning']['num_labels'] = round(0.01 * x, 2) if use_percents else x
            best_model, metrics, query_info = train_model(params, serialization_dir, selector, num_ensemble_models, recover=False)
            dump_metrics(os.path.join(save_dir, str(x) + ".json"), metrics, log=True)
            with open(os.path.join(save_dir, str(x) + "_query_info.json"), 'w', encoding='utf-8') as f:
                json.dump(query_info, f)
    else:
        params = Params.from_file('training_config/coref.jsonnet')
        if use_percents:
            params.params['trainer']['active_learning']['num_labels'] = 1
        else:
            params.params['trainer']['active_learning']['num_labels'] = 100
        # restore data file
        saved_data_file = '../data/saved_data_' + str(selector) + '_' + str(num_ensemble_models) + '_' + str(params.params['trainer']['active_learning']['num_labels']) + '.th'
        if os.path.exists(saved_data_file):
            params['dataset_reader']['saved_data_file'] = saved_data_file
        params.params['trainer']['active_learning']['use_percent'] = use_percents
        params.params['trainer']['active_learning']['save_al_queries'] = args.save_al_queries
        if testing or testing_vocab:
            params.params['trainer']['active_learning']['epoch_interval'] = 0
            del params.params['test_data_path']
            '''
            params.params['train_data_path'] = "/checkpoint/belindali/active_learning_coref/coref_ontonotes/dev.english.v4_gold_conll"
            params.params['dataset_reader']['fully_labelled_threshold'] = 100
            #'''
            if testing:
                params.params['model']['text_field_embedder']['token_embedders']['tokens'] = {'type': 'embedding', 'embedding_dim': 300}
        with TemporaryDirectory() as serialization_dir:
            print("temp file path: " + str(serialization_dir))
            params.params['trainer']['cuda_device'] = cuda_device
            params.params['trainer']['active_learning']['query_type'] = "pairwise" if pairwise else "discrete"
            params.params['trainer']['active_learning']['selector']['type'] = selector if selector else "entropy"
            params.params['trainer']['active_learning']['selector']['use_clusters'] = not no_clusters
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
                        type=int,
                        required=False,
                        help='labels to query per doc (n >= 0)')
    parser.add_argument("--save_al_queries",
                        action='store_true',
                        required=False,
                        help='Whether or not to save AL queries (or just simulate them using user inputs)')

   
    args = parser.parse_args()
    num_ensemble_models = None
    if vars(args)['selector'][:3] == 'qbc':
        assert (len(vars(args)['selector']) > 3)
        num_ensemble_models = int(vars(args)['selector'][3:])
        vars(args)['selector'] = 'qbc'
    main(vars(args)['cuda_device'], vars(args)['testing'], vars(args)['testing_vocab'], vars(args)['experiments'], vars(args)['pairwise'], vars(args)['selector'], num_ensemble_models, vars(args)['no_clusters'], args)
