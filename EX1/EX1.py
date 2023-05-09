# In this exercise you will be asked to fine-tune pretrained large language models to perform the sentiment analysis
# task on the SST2 dataset.
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel, Trainer, set_seed, \
    DataCollatorWithPadding, TrainingArguments, EvalPrediction
import transformers
from transformers import AutoModelForSequenceClassification
from evaluate import load #TODO correct evaluate??



#Write Python code for fine-tuning the following three pretrained language models to perform sentiment
#analysis on SST2: bert-base-uncased, roberta-base, google/electra-base-generator (these are the names of
#the models in the HuggingFace hub)

#1. Load arguments
#2. Load dataset
#3. Load model and tokenizer
#4. Tokenize dataset
#5. Define metrics for evaluation
#6. Train
#7. Evaluate (and maybe predict)

# Use the AutoModelForSequenceClassification class to load your models.

class FineTuner:
    """
    initializes the class
    """
    def __init__(self, seed, model_name, dataset, eval_metric):
        set_seed(seed) # set seed to chosen seed
        self.model_name = model_name
        self.dataset_name = dataset
        self.eval_metric = load("bleu")

    def run(self):
        raw_dataset = load_dataset(self.dataset_name, cache_dir=None) #TODO need more params? ca
        train_split = raw_dataset["train"]
        eval_split = raw_dataset["validation"]
        config, tokenizer, model = self.load_configs(model_name=self.model_name), self.load_model(model_name=self.model_name), \
                                   self.load_tokenizer(model_name=self.model_name)

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) #TODO args?
        training_args = self.__load_training_arguments()
        trainer, train_res = self.__train(model, training_args, train_split, eval_split,
                                                        self.compute_metrics, tokenizer, data_collator)
        metrics = trainer.evaluate(eval_split)

        # return something
        return trainer, metrics



    """
    #TODO
    """
    def __apply_data_collator(self, dataset):
        loader = DataCollatorWithPadding(dataset) #TOD0 set args?
        return loader

    """
    loads training_arguments param for trainer
    """
    def __load_training_arguments(self):
        training_args = TrainingArguments(num_train_epochs=1, output_dir="../PycharmProjects/ANLP") #TODO args?
        return training_args
    """
    loads configurations
    
    """
    def load_configs(self, model_name):
        configs = AutoConfig.from_pretrained(model_name) # TODO params
        return configs
    """
    loads model 
    @:param model_name name of model to load
    """
    def load_model(self, model_name):
        return AutoModelForSequenceClassification.from_pretrained(model_name) #TODO other args?

    """
    loads tokenizer
    @:param model_name name of model to get tokenizer for
    """
    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name) #TODO other args?
    """
    preprocessing function that receives a tokenizer and the data to tokenize and return ???
    @:param tokenizer #TODO
    @:param data unprocessed data to tokenize
    @:return res ???
    """
    def preproccess_function(self, tokenizer, data):
        res = tokenizer(data['???'], max_length=512, truncation=True) #TODO check params
        return res


    """
    method that trainer will recieve ref to ??#TODO
    """
    def compute_metrics(self, p:EvalPrediction):
        if not self.eval_metric:
            raise Exception("evaluation metric must be defined")
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = self.eval_metric.compute(predictions=preds, references=p.label_ids)["bleu"] #TODO bleu?
        return result

    def eval_metrics(self):
        pass

    def __train(self, model, training_args, train_split, eval_split, metric, tokenizer, data_collator):
        trainer = self.load_trainer(model=model, training_args=training_args,
                                    train_dataset=train_split, eval_dataset=eval_split,
                                    compute_metrics=metric,
                                    tokenizer=tokenizer, data_collator=data_collator
                                    )
        train_res = trainer.train()
        return trainer, train_res


    """
    loads instance of Trainer class with params
    """
    def load_trainer(self, model, training_args, train_dataset, eval_dataset, compute_metrics, tokenizer, data_collator):
        trainer = Trainer(model=model, training_args=training_args,
                          train_dataset=train_dataset, eval_dataset=eval_dataset,
                          compute_metrics=compute_metrics, data_collator=data_collator,
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


def main(model_names, seeds, dataset):
    trainers = defaultdict(dict)
    for model in model_names:
        for seed in seeds:
            finetuner = FineTuner(model_name=model, dataset=dataset, seed=seed)
            trainer, metrics = finetuner.run()


if __name__ == '__main__':
    # models we want to finetune on
    model_names = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]
    seeds = [1]
    main(model_names,seeds, "sst2")