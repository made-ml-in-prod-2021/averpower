from dataclasses import dataclass, field


@dataclass()
class GridSearchParams:
    param_grid: dict
    cv: int = field(default=5)
    scoring: str = field(default='roc_auc')
