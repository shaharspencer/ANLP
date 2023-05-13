# In this exercise you will be asked to fine-tune pretrained large language models to perform the sentiment analysis
# task on the SST2 dataset.

from collections import defaultdict
from docopt import docopt
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModel, Trainer, \
    set_seed, \
    DataCollatorWithPadding, TrainingArguments, EvalPrediction
from transformers import AutoModelForSequenceClassification

usage = '''
EX1.py CLI.
Usage:
    EX1.py <num_seeds> <training_samples> <val_samples> <pred_samples>
'''


# Write Python code for fine-tuning the following three pretrained language models to perform sentiment
# analysis on SST2: bert-base-uncased, roberta-base, google/electra-base-generator (these are the names of
# the models in the HuggingFace hub)

# 1. Load arguments
# 2. Load dataset
# 3. Load model and tokenizer
# 4. Tokenize dataset
# 5. Define metrics for evaluation
# 6. Train
# 7. Evaluate (and maybe predict)

# Use the AutoModelForSequenceClassification class to load your models.
# TODO During prediction, unlike during training, you should not pad the samples at all.???
#TODO rename ex1.py
#TODO calculate time for stuff
#todo wandb

class FineTuner:
    """
    initializes the class
    """

    def __init__(self, seed, model_name, dataset, train_samples, val_samples,
                 test_samples):
        self.test_samples = test_samples
        self.val_samples = val_samples
        self.train_samples = train_samples
        self.seed = seed
        set_seed(seed)  # set seed to chosen seed
        self.model_name = model_name
        self.dataset_name = dataset
        self.eval_metric = load_metric("accuracy")  # TODO which metric?

    def get_labels(self):
        labels = self.train_split.features["label"].names
        self.label_to_id = {v: i for i, v in enumerate(labels)}
        return labels

    def run(self):
        self.tokenizer = self.__load_tokenizer(model_name=self.model_name)
        raw_dataset = load_dataset(self.dataset_name,
                                   cache_dir=None)  # TODO need more params? ca
        self.raw_dataset = raw_dataset.map(self.preproccess_function,
                                      batched=True,
                                      desc="running tokenizer on dataset", )  # map data to tok
        self.process_datasets()


        self.label_list = self.get_labels()
        self.config = self.__load_configs(model_name=self.model_name)

        self.model = self.__load_model(model_name=self.model_name,
                                       config=self.config)

        self.define_labels()

        data_collator = DataCollatorWithPadding(self.tokenizer,
                                                padding="longest")  # TODO args?
        training_args = self.__load_training_arguments()
        self.trainer, self.metric = self.__train(self.model, training_args,
                                               self.train_split,
                                               self.eval_split,
                                               self.compute_metrics,
                                               self.tokenizer, data_collator)
        self.metrics = self.trainer.evaluate(self.eval_split)
    def get_test(self):
        return self.test_split
    def get_trainer(self):
        return self.trainer

    def get_metrics(self):
        return self.metrics

    def get_seed(self):
        return self.seed

    def process_datasets(self):
        if self.train_samples != -1:
            self.train_split = self.raw_dataset["train"].select(range(self.train_samples))  # TODO remove select
        else:
            self.train_split = self.raw_dataset["train"]

        if self.val_samples != -1:
            self.eval_split = self.raw_dataset["val"].select(range(self.val_samples))
        else:
            self.eval_split = self.raw_dataset["val"]

        if self.test_samples != -1:
            self.test_split = self.raw_dataset["test"].select(range(self.test_samples))
        else:
            self.test_split = self.raw_dataset["test"]



    def define_labels(self):

        self.model.config.label2id = {v: i for i, v in enumerate(self.label_list)}
        self.model.config.id2label = {id: label for label, id in self.config.label2id.items()} #IS THIS CORRECT?
    """
    loads training_arguments param for trainer
    """

    def __load_training_arguments(self):
        training_args = TrainingArguments(save_total_limit=1,
                                          output_dir="../PycharmProjects/ANLP")  # TODO args need to change to default
        return training_args

    """
    loads configurations

    """

    def __load_configs(self, model_name):
        if not self.label_list:
            raise Exception("label list must be defined")
        configs = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name,
                                             num_labels = len(self.label_list),
                                            )#????
        #
        #TODO: need to define number of labels?
        return configs

    """
    loads model 
    @:param model_name name of model to load
    """

    def __load_model(self, model_name, config):
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=
            config)  # TODO other args?

    """
    loads tokenizer
    @:param model_name name of model to get tokenizer for
    """

    def __load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)  # TODO other args?

    """
    preprocessing function that receives a tokenizer and the data to tokenize and return ???
    @:param tokenizer #TODO
    @:param data unprocessed data to tokenize
    @:return res ???
    """

    def preproccess_function(self, data):
        if not self.tokenizer:
            raise Exception("need to define tokenizer")
        res = self.tokenizer(data['sentence'], max_length=self.tokenizer.model_max_length,
                             truncation=True)  # TODO check params

        # if "label" in data: #TODO need to do this? idk
        #     res["label"] = [(self.label_to_id[l] if l != -1 else -1) for l in
        #                        data["label"]]

        return res

    """
    method that trainer will recieve ref to ??#TODO
    """

    def compute_metrics(self, p: EvalPrediction):

        preds = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions
        print(f"prediction 0: {preds}")
        preds = np.argmax(preds, axis=1)
        result = self.eval_metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
        # if not self.eval_metric:
        #     raise Exception("evaluation metric must be defined")
        # preds = p.predictions[0] if isinstance(p.predictions,
        #                                        tuple) else p.predictions
        # preds = np.argmax(preds, axis=1)
        # result = self.eval_metric.compute(predictions=preds, references=p.label_ids)[
        #     "accuracy"]  # TODO bleu?
        # return result

    def __train(self, model, training_args, train_split, eval_split, metric,
                tokenizer, data_collator):
        trainer = self.__load_trainer(model=model, training_args=training_args,
                                      train_dataset=train_split,
                                      eval_dataset=train_split,
                                      compute_metrics=metric,
                                      tokenizer=tokenizer,
                                      data_collator=data_collator
                                      )
        res = trainer.train()
        return trainer, res

    """
    loads instance of Trainer class with params
    """

    def __load_trainer(self, model, training_args, train_dataset, eval_dataset,
                       compute_metrics, tokenizer, data_collator):
        trainer = Trainer(model=model, args=training_args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          compute_metrics=compute_metrics,
                          data_collator=data_collator,
                          tokenizer=tokenizer)
        return trainer

    """
    evaluate the training using the instance of Trainer with the evaluadation dataset
    @:param trainer instance of trainer class
    @:return #TODO
    """

    def evaluate(self, trainer: Trainer, eval_dataset):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        return metrics

    """
    this method returns predictions of the trained model on the test set. 
    """

    def predict(self, file):
        self.model.eval() #TODO should be self.trainer.eval()???
        predict_dataset = self.test_split.remove_columns("label")
        with open(file, mode="w") as f:
            for i in range(len(predict_dataset)):
                sent = self.test_split["sentence"][i]
                pred = self.trainer.predict(predict_dataset.select(range(i, i+1)),
                                          metric_key_prefix="predict").predictions
                f.write(f"{sent}###{pred}")


def mean_accuracy_on_seeds(trained_models_by_seed):
    accuracies = [model.get_metrics()["eval_accuracy"] for model in trained_models_by_seed]
    return np.mean(accuracies)


def std_accuracy_on_seeds(trained_models_by_seed):
    accuracies = [model.get_metrics()["eval_accuracy"] for model in trained_models_by_seed]
    return np.std(accuracies)


def write_res_file(trainer_dict):
    with open("res.txt", mode="w") as f:
        for model_name in trainer_dict.keys():
            metrics = trainer_dict[model_name]
            mean_accuracy = mean_accuracy_on_seeds(metrics)
            std_accuracy = std_accuracy_on_seeds(metrics)
            f.write(f"{model_name},{mean_accuracy} +- {std_accuracy}")
        f.write("----")

def write_predictions_file(best_model):
    predictions_on_best_model = best_model.predict()
    sentences = best_model.get_test()["sentence"] #TODO get sentences
    with open("predictions.txt", mode="w", encoding="utf-8") as f:
        for sent, pred in zip(sentences,predictions_on_best_model):
            f.write(f"{sent}###{pred}")



def max_accuracy_for_model_keyfunc(metrics):
    mean_accuracy = mean_accuracy_on_seeds(metrics)
    return mean_accuracy

def predict_best_model(trainer_dict):
    avg_accuracy = -1
    best_model = None
    for key in trainer_dict.keys():
      avg = max_accuracy_for_model_keyfunc(trainer_dict[key])
      if avg > avg_accuracy:
        avg_accuracy = avg
        best_model = key
    best_seed_on_winner_model = max(trainer_dict[best_model],
                                    key = lambda k: k.get_metrics()["eval_accuracy"])
    write_predictions_file(best_seed_on_winner_model)



def main(model_names, seeds, dataset, train_samples, val_samples, test_samples):
    trainers = defaultdict(list)
    for model in model_names:
        for seed in seeds:
            finetuner = FineTuner(model_name=model, dataset=dataset,
                                  seed=seed, train_samples=train_samples,
                                  val_samples=val_samples, test_samples=test_samples)
            finetuner.run()
            trainers[model].append(finetuner)

        write_res_file(trainer_dict=trainers) #TODO move forward
        predict_best_model(trainers)


if __name__ == '__main__':
    # models we want to finetune on
    model_names = ["bert-base-uncased", "roberta-base",
                   "google/electra-base-generator"]
    args = docopt(usage)

    seeds = range(int(args["<num_seeds"]))
    train_samples = int(args["<training_samples>"])
    val_samples = int(args["<val_samples>"])
    pred_samples = int(args["<pred_samples>"])
    main(model_names, seeds, "sst2", train_samples, val_samples, pred_samples)