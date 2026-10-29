"""Data preparation for the Early Spring project.

The Preparation notebook explains every choice made here. This code lives in a .py file (and not
only in the notebook) because `preprocessor.pkl` and `model.pkl` are pickles: to load them again,
Python has to be able to import the classes they were built from. Keep this file next to the
notebooks (it is committed with them).

    import pickle
    import early_spring_prep            # noqa: F401  (needed so pickle can find the classes)
    preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
    X_ready = preprocessor.transform(test_set)          # train set, test set, or generate_samples() output
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

TARGET = "early_spring"
SEED_COST = 0.2      # seeds cost 1/5 of the extra harvest's revenue (client)

# --------------------------------------------------------------------------------------------
# Feature sets (chosen in the Preparation notebook, from Exploration + random-forest importances)
# --------------------------------------------------------------------------------------------
# Groups of features that say almost the same thing (see the clustering in the Preparation notebook).
REDUNDANT_GROUPS = {
    "february cold": ["air_feb", "cold_days", "air_feb_late", "gdd"],
    "snow":          ["snow_days", "snow_days_late"],
    "soil temp":     ["soil_feb", "soil_feb_late", "soil_trend"],
    "soil moisture": ["soil_moisture", "soil_moisture_late"],
    "ONI":           ["oni_son", "oni_ond", "oni_ndj", "oni_trend"],
    "AO":            ["ao_nov", "ao_dec", "ao_jan", "ao_winter_mean"],
}

FEATURE_SETS = {
    # every column of the train/test sets
    "all": None,
    # one or two per redundant group, one ONI and one AO summary, and year (Preparation, section 2)
    "selected": ["year", "air_feb", "cold_days", "gdd", "snow_days", "soil_feb", "soil_moisture",
                 "oni_ndj", "ao_nov"],
    # only the features that clearly carried information in the random forest + year for the trend
    "small": ["year", "air_feb", "snow_days", "soil_moisture"],
}

SKEWED = ["gdd"]                                   # right-skewed (Exploration)
INDEX_COLS = ["oni_son", "oni_ond", "oni_ndj", "oni_trend", "ao_nov", "ao_dec", "ao_jan", "ao_winter_mean"]
AIR_COLS = ["cold_days", "air_feb_late", "air_feb"]
POLY_DEFAULT = ["air_feb", "snow_days", "soil_moisture"]   # strongest farm features


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Turn a train/test/generate_samples() frame into numeric model features.

    Parameters (all tunable in a grid search, e.g. ``preprocessor__features__log_skewed``)
    ----------
    feature_set : "all", "selected" or "small" (see FEATURE_SETS), or an explicit list of columns
    log_skewed : apply log1p to the right-skewed features (gdd)
    add_rule_flag : add `cold_and_snowy` = 1 if Feb had more snow days AND more cold days than the
        training median (the manual rule from Exploration; the medians are learned in `fit`)
    add_missing_flags : add `indices_missing` (no ONI/AO, 1947-1950) and `air_missing` (no Feb air temps, 2004)
    use_year, use_phil, use_indices : keep `year`, `phil_early`, the ONI/AO columns
    """

    def __init__(self, feature_set="small", log_skewed=False, add_rule_flag=False, add_missing_flags=False,
                 use_year=True, use_phil=True, use_indices=True):
        self.feature_set = feature_set
        self.log_skewed = log_skewed
        self.add_rule_flag = add_rule_flag
        self.add_missing_flags = add_missing_flags
        self.use_year = use_year
        self.use_phil = use_phil
        self.use_indices = use_indices

    # -- helpers ----------------------------------------------------------------------------
    @staticmethod
    def _as_float_frame(X):
        X = pd.DataFrame(X).copy()
        X = X.drop(columns=[TARGET], errors="ignore")            # the label is never a feature
        if "phil_early" in X:                                     # nullable boolean -> 0/1
            X["phil_early"] = X["phil_early"].astype("float")
        return X.apply(pd.to_numeric, errors="coerce").astype(float)

    def _columns(self, available):
        fs = self.feature_set
        cols = list(available) if (fs is None or fs == "all") else list(FEATURE_SETS[fs] if isinstance(fs, str) else fs)
        missing = [c for c in cols if c not in available]
        if missing:
            raise KeyError(f"columns not in the data: {missing}")
        if not self.use_year:
            cols = [c for c in cols if c != "year"]
        if not self.use_phil:
            cols = [c for c in cols if c != "phil_early"]
        if not self.use_indices:
            cols = [c for c in cols if c not in INDEX_COLS]
        return cols

    # -- sklearn API ------------------------------------------------------------------------
    def fit(self, X, y=None):
        X = self._as_float_frame(X)
        self.input_columns_ = list(X.columns)
        self.columns_ = self._columns(X.columns)
        # thresholds for the manual-rule flag, learned from the TRAINING rows only
        # (only possible if both columns are in the data)
        self.has_rule_inputs_ = {"snow_days", "cold_days"} <= set(X.columns)
        if self.has_rule_inputs_:
            self.snow_median_ = float(X["snow_days"].median())
            self.cold_median_ = float(X["cold_days"].median())
        self.feature_names_out_ = list(self.transform(X.iloc[:1]).columns)
        return self

    def transform(self, X):
        X = self._as_float_frame(X)
        out = X[self.columns_].copy()
        if self.log_skewed:
            for c in SKEWED:
                if c in out:
                    out[c] = np.log1p(out[c].clip(lower=0))
        if self.add_rule_flag and self.has_rule_inputs_:
            rule = (X["snow_days"] > self.snow_median_) & (X["cold_days"] > self.cold_median_)
            out["cold_and_snowy"] = rule.astype(float)
            out.loc[X["snow_days"].isna() | X["cold_days"].isna(), "cold_and_snowy"] = np.nan
        if self.add_missing_flags:
            if self.use_indices and any(c in out for c in INDEX_COLS):
                out["indices_missing"] = X[[c for c in INDEX_COLS if c in X]].isna().any(axis=1).astype(float)
            if any(c in out for c in AIR_COLS):
                out["air_missing"] = X[[c for c in AIR_COLS if c in X]].isna().any(axis=1).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


class PairwiseProducts(BaseEstimator, TransformerMixin):
    """Polynomial features for a few chosen columns only.

    degree=1 -> nothing added. degree=2 -> adds the square of each chosen column and the product of
    every pair (e.g. snow_days*air_feb). Used on the strongest few features only, because degree-2
    features of all ~20 columns would be ~200 columns for 57 rows.
    """

    def __init__(self, degree=1, columns=tuple(POLY_DEFAULT), interaction_only=False):
        self.degree = degree
        self.columns = columns
        self.interaction_only = interaction_only

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = [c for c in self.columns if c in X.columns]
        self.feature_names_out_ = list(self.transform(X.iloc[:1]).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.degree < 2:
            return X
        new = {}
        if not self.interaction_only:
            for c in self.columns_:
                new[f"{c}^2"] = X[c] ** 2
        for a, b in combinations(self.columns_, 2):
            new[f"{a}*{b}"] = X[a] * X[b]
        return pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


class ToFrame(BaseEstimator, TransformerMixin):
    """Run an sklearn step (imputer, scaler) but keep the pandas column names.

    `step` can also be "passthrough" (do nothing), so a grid search can switch scaling off with
    ``prep__scale__step="passthrough"``.
    """

    def __init__(self, step="passthrough"):
        self.step = step

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        self.step_ = None if self.step in (None, "passthrough") else clone(self.step).fit(X, y)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        if self.step_ is None:
            return X.copy()
        return pd.DataFrame(self.step_.transform(X), columns=self.columns_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def make_preprocessor(feature_set="small", log_skewed=False, add_rule_flag=False, add_missing_flags=False,
                      use_year=True, use_phil=True, use_indices=True, impute_strategy="median",
                      scaler="standard", poly_degree=1, poly_columns=tuple(POLY_DEFAULT)):
    """The full preparation pipeline: engineer -> impute -> scale -> (optional) polynomial features."""
    scalers = {"standard": StandardScaler(), "robust": RobustScaler(), "none": "passthrough"}
    return Pipeline([
        ("features", FeatureEngineer(feature_set, log_skewed, add_rule_flag, add_missing_flags,
                                     use_year, use_phil, use_indices)),
        ("impute", ToFrame(SimpleImputer(strategy=impute_strategy))),
        ("scale", ToFrame(scalers[scaler])),
        ("poly", PairwiseProducts(degree=poly_degree, columns=tuple(poly_columns))),
    ])


# --------------------------------------------------------------------------------------------
# The business measure (used in Preparation and Training)
# --------------------------------------------------------------------------------------------
def profit_per_year(y_true, buy):
    """Average profit per year in units of R: +0.8 if we buy and spring is early, -0.2 if we buy and it isn't."""
    y_true, buy = np.asarray(y_true, dtype=bool), np.asarray(buy, dtype=bool)
    return float(np.mean(buy * np.where(y_true, 1 - SEED_COST, -SEED_COST)))


def profit_score(estimator, X, y, threshold=SEED_COST):
    """Scorer for cross_val_score / GridSearchCV: buy when P(early) > threshold (break-even = 0.2)."""
    return profit_per_year(y, estimator.predict_proba(X)[:, 1] > threshold)
