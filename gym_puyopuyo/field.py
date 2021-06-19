from __future__ import unicode_literals, annotations

import sys

import numpy as np
from typing import Tuple, List, Optional, Dict, Final

import puyocore as core
from gym_puyopuyo import util
from gym_puyopuyo.bitboard import popcount


class BottomField(object):
    WIDTH: int = 8
    HEIGHT: int = 8
    CLEAR_THRESHOLD: int = 4

    def __init__(self, num_layers, has_garbage=False) -> None:
        self.num_layers: int = num_layers
        self.has_garbage: bool = has_garbage
        self.offset: int = 0
        self.data = bytearray(0)
        if has_garbage:
            self.num_colors: int = num_layers - 1
        else:
            self.num_colors: int = num_layers
        self.reset()

    def reset(self) -> None:
        self.data: bytearray = bytearray(8 * self.num_layers)

    def render(self, outfile=sys.stdout, width: Optional[int] = None, height: Optional[int] = None, in_place: bool = False) -> None:
        height: int = height or self.HEIGHT
        width: int = width or self.WIDTH
        if not in_place:
            for _ in range(height):
                outfile.write("\n")
            util.print_up(height, outfile=outfile)
        for i in range(self.HEIGHT - height, self.HEIGHT):
            for j in range(width):
                empty = True
                for k in range(self.num_colors):
                    puyo = self.data[i + self.HEIGHT * k] & (1 << j)
                    if puyo:
                        util.print_puyo(k, outfile=outfile)
                        empty = False
                if self.has_garbage:
                    garbage_puyo = self.data[i + self.HEIGHT * self.num_colors] & (1 << j)
                    if garbage_puyo:
                        util.print_color(6, outfile=outfile)
                        outfile.write("\u25ce ")
                        empty = False
                if empty:
                    outfile.write("\u00b7 ")
                util.print_reset(outfile=outfile)
            util.print_down(1, outfile=outfile)
            util.print_back(2 * width, outfile=outfile)

    def debug(self) -> None:
        core.bottom_render(self.data, self.num_layers)

    def handle_gravity(self) -> int:
        return core.bottom_handle_gravity(self.data, self.num_layers)

    def clear_groups(self, chain_number: int) -> int:
        did_clear: bool = core.bottom_clear_groups(self.data, self.num_layers, self.has_garbage)
        if did_clear:
            return (chain_number + 1) ** 2
        return 0

    def resolve(self) -> Tuple[int, int]:
        chain: int = core.bottom_resolve(self.data, self.num_layers, self.has_garbage)
        return chain * chain, chain

    def overlay(self, stack) -> None:
        layer: BottomField = BottomField.from_list(stack, num_layers=self.num_layers)
        if layer.num_layers > self.num_layers:
            raise ValueError("Overlay has too many layers")
        mask: bytearray = bytearray(8)
        for i, mine in enumerate(self.data):
            mask[i % 8] |= mine
        for i, (mine, yours) in enumerate(zip(self.data, layer.data)):
            self.data[i] = (mine | (yours & ~mask[i % 8]))

    def encode(self) -> np.ndarray:
        data = core.bottom_encode(self.data, self.num_layers)
        return np.fromstring(data, dtype=np.int8).reshape(self.num_layers, self.HEIGHT, self.WIDTH)

    def mirror(self) -> None:
        core.mirror(self.data, self.num_layers)

    def shift(self, amount: int) -> None:
        if amount > 0:
            for i in range(len(self.data)):
                self.data[i] <<= amount
        elif amount < 0:
            amount = -amount
            for i in range(len(self.data)):
                self.data[i] >>= amount

    def _valid_moves(self, width: Optional[int] = None):
        return core.bottom_valid_moves(self.data, self.num_layers)

    def _make_move(self, action, puyo_a: int, puyo_b: int):
        core.make_move(self.data, action, puyo_a, puyo_b)

    def to_list(self) -> List[Optional[int]]:
        result: List[Optional[int]] = []
        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                puyo = None
                for k in range(self.num_layers):
                    if self.data[i + self.HEIGHT * k] & (1 << j):
                        puyo = k
                result.append(puyo)
        return result

    def puyo_at(self, x: int, y: int) -> Optional[int]:
        for k in range(self.num_layers):
            if self.data[y + self.HEIGHT * k] & (1 << x):
                return k
        return None

    def _unsafe_set_puyo_at(self, x: int, y: int, puyo: int) -> None:
        self.data[y + self.HEIGHT * puyo] |= 1 << x

    @property
    def popcount(self):
        return popcount(self.data)

    @property
    def sane(self) -> bool:
        mask: bytearray = bytearray(8)
        for i, line in enumerate(self.data):
            if line & mask[i % 8]:
                return False
            mask[i % 8] |= line
        return True

    @classmethod
    def from_list(cls, stack: List[Optional[int]], num_layers: Optional[int] = None, has_garbage: bool = False) -> BottomField:
        if len(stack) % cls.WIDTH != 0:
            raise ValueError("Puyos must form complete rows")
        if len(stack) > cls.WIDTH * cls.HEIGHT:
            raise ValueError("Too many puyos")
        if num_layers is None:
            num_layers = 0
            for puyo in stack:
                if puyo is not None:
                    num_layers = max(num_layers, puyo)
            num_layers += 1
        instance: BottomField = cls(num_layers, has_garbage=has_garbage)
        for index, puyo in enumerate(stack):
            if puyo is not None:
                instance.data[index // cls.WIDTH + cls.HEIGHT * puyo] |= 1 << (index % cls.WIDTH)
        return instance


class TallField(object):
    WIDTH: int = 8
    HEIGHT: int = 16
    CLEAR_THRESHOLD: int = 4

    def __init__(self, num_layers: int, tsu_rules=False, has_garbage=False) -> None:
        if has_garbage:
            self.num_colors: int = num_layers - 1
        else:
            self.num_colors: int = num_layers
        self.num_layers: int = num_layers
        self.tsu_rules: bool = tsu_rules
        self.has_garbage: bool = has_garbage
        self.offset: int = 3 if tsu_rules else 0
        self.data: bytearray = bytearray(0)
        self.reset()

    def reset(self) -> None:
        self.data = bytearray(16 * self.num_layers)

    def render(self, outfile=sys.stdout, width: Optional[int] = None, height: Optional[int] = None, in_place=False):
        height = height or self.HEIGHT
        width = width or self.WIDTH
        if not in_place:
            for _ in range(height):
                outfile.write("\n")
            util.print_up(height, outfile=outfile)
        for i in range(self.HEIGHT - height, self.HEIGHT):
            offset, row = divmod(i, 8)
            for j in range(width):
                empty = True
                for k in range(self.num_colors):
                    if self.data[row + 8 * k + 8 * self.num_layers * offset] & (1 << j):
                        if self.tsu_rules and i <= self.offset:
                            util.print_puyo(k + 7, outfile=outfile)
                        else:
                            util.print_puyo(k, outfile=outfile)
                        empty = False
                if self.has_garbage:
                    garbage_puyo = self.data[row + 8 * self.num_colors + 8 * self.num_layers * offset] & (1 << j)
                    if garbage_puyo:
                        if self.tsu_rules and i <= self.offset:
                            util.print_color(4, outfile=outfile)
                        else:
                            util.print_color(6, outfile=outfile)
                        outfile.write("\u25ce ")
                        empty = False
                if empty:
                    outfile.write("\u00b7 ")
                util.print_reset(outfile=outfile)
            util.print_down(1, outfile=outfile)
            util.print_back(2 * width, outfile=outfile)

    def debug(self) -> None:
        core.tall_render(self.data, self.num_layers)

    def handle_gravity(self) -> int:
        return core.tall_handle_gravity(self.data, self.num_layers)

    def clear_groups(self, chain_number: int) -> int:
        return core.tall_clear_groups(self.data, self.num_layers, chain_number, self.tsu_rules, self.has_garbage)

    def resolve(self) -> Tuple[int, int]:
        return core.tall_resolve(self.data, self.num_layers, self.tsu_rules, self.has_garbage)

    def encode(self) -> np.ndarray:
        data: bytes = core.tall_encode(self.data, self.num_layers)
        return np.fromstring(data, dtype=np.int8).reshape(self.num_layers, self.HEIGHT, self.WIDTH)

    def overlay(self, stack) -> None:
        layer: TallField = TallField.from_list(stack, num_layers=self.num_layers)
        if layer.num_layers > self.num_layers:
            raise ValueError("Overlay has too many layers")
        top_mask = bytearray(8)
        bottom_mask = bytearray(8)
        half = 8 * self.num_layers
        for i, mine in enumerate(self.data[:half]):
            top_mask[i % 8] |= mine
        for i, mine in enumerate(self.data[half:]):
            bottom_mask[i % 8] |= mine
        for i, (mine, yours) in enumerate(zip(self.data[:half], layer.data[:half])):
            self.data[i] = (mine | (yours & ~top_mask[i % 8]))
        for i, (mine, yours) in enumerate(zip(self.data[half:], layer.data[half:])):
            self.data[i + half] = (mine | (yours & ~bottom_mask[i % 8]))

    def mirror(self) -> None:
        core.mirror(self.data, 2 * self.num_layers)

    def shift(self, amount: int) -> None:
        if amount > 0:
            for i in range(len(self.data)):
                self.data[i] <<= amount
        elif amount < 0:
            amount = -amount
            for i in range(len(self.data)):
                self.data[i] >>= amount

    def _valid_moves(self, width: int) -> int:
        return core.tall_valid_moves(self.data, self.num_layers, width, self.tsu_rules)

    def _make_move(self, action: int, puyo_a: int, puyo_b: int) -> None:
        core.make_move(self.data, action, puyo_a, puyo_b)

    def to_list(self) -> List[Optional[int]]:
        result: List[Optional[int]] = []
        for i in range(self.HEIGHT):
            offset, row = divmod(i, 8)
            for j in range(self.WIDTH):
                puyo = None
                for k in range(self.num_layers):
                    if self.data[row + 8 * k + 8 * self.num_layers * offset] & (1 << j):
                        puyo = k
                result.append(puyo)
        return result

    def puyo_at(self, x: int, y: int) -> Optional[int]:
        offset, row = divmod(y, 8)
        for k in range(self.num_layers):
            if self.data[row + 8 * k + 8 * self.num_layers * offset] & (1 << x):
                return k
        return None

    def _unsafe_set_puyo_at(self, x: int, y: int, puyo: int):
        offset, row = divmod(y, 8)
        self.data[row + 8 * puyo + 8 * self.num_layers * offset] |= 1 << x

    @property
    def popcount(self) -> int:
        return popcount(self.data)

    @property
    def sane(self) -> bool:
        half = 8 * self.num_layers
        mask = bytearray(8)
        for i, line in enumerate(self.data[:half]):
            if line & mask[i % 8]:
                return False
            mask[i % 8] |= line
        mask = bytearray(8)
        for i, line in enumerate(self.data[half:]):
            if line & mask[i % 8]:
                return False
            mask[i % 8] |= line
        return True

    @classmethod
    def from_list(cls, stack: List[Optional[int]], num_layers: Optional[int] = None, tsu_rules=False, has_garbage=False) -> TallField:
        if len(stack) % cls.WIDTH != 0:
            raise ValueError("Puyos must form complete rows")
        if len(stack) > cls.WIDTH * cls.HEIGHT:
            raise ValueError("Too many puyos")
        if num_layers is None:
            num_layers = 0
            for puyo in stack:
                if puyo is not None:
                    num_layers = max(num_layers, puyo)
            num_layers += 1
        instance = cls(num_layers, tsu_rules=tsu_rules, has_garbage=has_garbage)
        for index, puyo in enumerate(stack):
            if puyo is not None:
                row, column = divmod(index, cls.WIDTH)
                offset = row // 8
                row %= 8
                instance.data[row + 8 * puyo + 8 * num_layers * offset] |= 1 << column
        return instance
