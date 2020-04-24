# pylint: disable=access-member-before-definition
from typing import Dict

from overrides import overrides
import torch

from allennlp.data.fields.field import Field


class PairField(Field[torch.Tensor]):
    def __init__(self, first_item: Field, sec_item: Field) -> None:
        self.first_item = first_item
        self.sec_item = sec_item

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        # pylint: disable=no-self-use
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        # pylint: disable=unused-argument
        tensor = torch.stack([self.first_item.as_tensor(padding_lengths), self.sec_item.as_tensor(padding_lengths)], dim=0)
        return tensor

    @overrides
    def empty_field(self):
        return PairField(self.first_item.empty_field(), self.sec_item.empty_field())

    def __str__(self) -> str:
        return f"SpanField with spans: ({self.first_item}, {self.sec_item})."
