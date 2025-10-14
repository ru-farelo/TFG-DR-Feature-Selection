import argparse
from typing import Union


def parse_percentage(value):
    """Convierte valores con % en números"""
    if isinstance(value, str) and value.endswith('%'):
        return float(value.strip('%'))
    return float(value)

def read_config() -> argparse.Namespace:
    """
    Read the configuration parameters from the command line.
    Returns:
        dict: Configuration parameters
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--classifier", type=str, help="Classifier name")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--cv_outer", type=int, default=10, help="Outer cross-validation folds")
    parser.add_argument("--cv_inner", type=int, default=5, help="Inner cross-validation folds")
    parser.add_argument("--binary_threshold", type=float, default=0.005, help="Binary threshold")
    parser.add_argument("--pu_learning", type=str, default=False, help="PU learning")
    parser.add_argument("--pu_k", type=int, nargs="+", default=10, help="PU k")
    parser.add_argument("--pu_t", type=float, nargs="+", default=None, help="PU t values")
    parser.add_argument("--neptune", type=bool, default=False, help="Neptune logging")

    # FAST mRMR
    parser.add_argument("--fast_mrmr", action="store_true", help="Enable Fast MRMR feature selection")
    parser.add_argument("--fast_mrmr_k", type=parse_percentage, default=0, 
                        help="Number of features to select with Fast MRMR (use '5%' for percentage)")    

    # Bagging disjoint
    parser.add_argument("--bagging", action="store_true", help="Activar selección de características con bagging")
    parser.add_argument("--bagging_n", type=parse_percentage, default=0.0, 
                        help="Porcentaje de características por subconjunto de bagging (ej. '5%')")
    parser.add_argument("--bagging_groups", type=int, default=5, help="Número de grupos disjuntos para bagging")

    # Carbon tracking
    parser.add_argument("--carbon_tracking", action="store_true", 
                        help="Activar medición de emisiones CO2 del 10-fold CV (solo proceso Python)")

    args = parser.parse_args()
    args = vars(args)

    return args
