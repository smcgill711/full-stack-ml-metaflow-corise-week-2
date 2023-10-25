# TODO: In this cell, write your BaselineChallenge flow in the baseline_challenge.py file.

from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
    current,
)
from metaflow.cards import Table, Markdown, Artifact, Image
import numpy as np
from dataclasses import dataclass

# TODO: Define your labeling function here.
def labeling_function(row):
    """
    A function to derive labels from the user's review data.
    Use the rating only, and return a binary positive or 
    negative sentiment score based on an arbitrarty cutoff.
    """
    CUTOFF_VALUE = 4
    x = 0
    if row['rating'] >= CUTOFF_VALUE:
        x = 1
    return x


@dataclass
class ModelResult:
    "A custom struct for storing model evaluation results."
    name: None
    params: None
    pathspec: None
    acc: None
    rocauc: None


class BaselineChallenge(FlowSpec):
    split_size = Parameter("split-sz", default=0.2)
    data = IncludeFile("data", default="Womens Clothing E-Commerce Reviews.csv")
    kfold = Parameter("k", default=5)
    scoring = Parameter("scoring", default="accuracy")

    @step
    def start(self):
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        # TODO: load the data.
        df = pd.read_csv(io.StringIO(self.data), index_col=0)
        # Look up a few lines to the IncludeFile('data', default='Womens Clothing E-Commerce Reviews.csv').
        # You can find documentation on IncludeFile here: https://docs.metaflow.org/scaling/data#data-in-local-files

        # filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df = df[~df.review_text.isna()]
        df["review"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)
        self.df = pd.DataFrame({"label": labels, **_has_review_df})

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({"review": reviews, "label": labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")

        self.next(self.baseline, self.model)

    @step
    def baseline(self):
        "Compute the baseline"

        from sklearn.metrics import accuracy_score, roc_auc_score

        self._name = "baseline"
        params = "Always predict 1"
        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"

        # TODO: predict the majority class
        MAJOR_CLASS = self.traindf.groupby('label')['label'].count().sort_values(ascending=False).index.values[0]
        self.valdf['target_pred'] = MAJOR_CLASS 
        # TODO: return the accuracy_score of these predictions
        acc = accuracy_score(self.valdf['label'], self.valdf['target_pred'])

        # TODO: return the roc_auc_score of these predictions
        rocauc = roc_auc_score(self.valdf['label'], self.valdf['target_pred'])
        self.result = ModelResult("Baseline", params, pathspec, acc, rocauc)
        self.next(self.aggregate)

    @step
    def model(self):
        # TODO: import your model if it is defined in another file.
        from model import NbowModel
        
        self._name = "model"
        # NOTE: If you followed the link above to find a custom model implementation,
        # you will have noticed your model's vocab_sz hyperparameter.
        # Too big of vocab_sz causes an error. Can you explain why?
        self.hyperparam_set = [{"vocab_sz": 100}, {"vocab_sz": 300}, {"vocab_sz": 500}]
        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"

        print('Starting NbowModel')
        self.results = []
        for params in self.hyperparam_set:
            model = NbowModel(params['vocab_sz'])  # TODO: instantiate your custom model here!
            print('Fitting NbowModel')
            model.fit(X=self.traindf["review"], y=self.traindf["label"])
            print('Finished NbowModel')
            # TODO: evaluate your custom model in an equivalent way to accuracy_score.
            acc = model.eval_acc(self.valdf["review"], self.valdf["label"])
            # TODO: evaluate your custom model in an equivalent way to roc_auc_score.
            rocauc = model.eval_rocauc(self.valdf["review"], self.valdf["label"])
            self.results.append(
                ModelResult(
                    f"NbowModel - vocab_sz: {params['vocab_sz']}",
                    params,
                    pathspec,
                    acc,
                    rocauc,
                )
            )

        self.next(self.aggregate)

    @step
    def aggregate(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    BaselineChallenge()
