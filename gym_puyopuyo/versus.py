from __future__ import unicode_literals, annotations
from typing import Tuple, List, Optional, Dict, Final
import sys

import numpy as np
from gym.utils import seeding

from gym_puyopuyo import util
from gym_puyopuyo.state import State


class VersusState(State):
    def __init__(
            self,
            height: int,
            width: int,
            num_colors: int,
            num_deals: Optional[int] = None,
            tsu_rules: bool = False,
            deals=None,
            seed: Optional[int] = None,
            step_bonus: int = 0,
            all_clear_bonus: int = 0,
            target_score: int = 0,
            max_received_garbage: float = float("inf"),
    ):
        super(VersusState, self).__init__(
            height,
            width,
            num_colors + 1,
            num_deals=num_deals,
            tsu_rules=tsu_rules,
            deals=deals,
            seed=seed,
            has_garbage=True,
        )
        self.step_bonus: int = step_bonus
        self.all_clear_bonus: int = all_clear_bonus
        self.target_score: int = target_score
        self.max_received_garbage: float = max_received_garbage
        self.all_clear_pending: bool = False
        self.step_score: int = 0
        self.chain_score: int = 0
        self.chain_number: int = 0
        self.pending_garbage: int = 0

    def reset(self) -> None:
        super(VersusState, self).reset()
        self.all_clear_pending = False
        self.step_score = 0
        self.chain_score = 0
        self.chain_number = 0
        self.pending_garbage = 0

    def clone(self) -> VersusState:
        deals: List[Tuple[int, int]] = self.deals[:]
        if self.num_deals is None:
            clone_deals = deals
        else:
            clone_deals = None
        clone: VersusState = VersusState(
            self.height,
            self.width,
            self.num_colors,
            self.num_deals,
            tsu_rules=self.tsu_rules,
            deals=clone_deals,
            step_bonus=self.step_bonus,
            all_clear_bonus=self.all_clear_bonus,
            target_score=self.target_score,
            max_received_garbage=self.max_received_garbage,
        )
        clone.field.data[:] = self.field.data
        clone.deals = deals
        clone.all_clear_pending = self.all_clear_pending
        clone.step_score = self.step_score
        clone.chain_score = self.chain_score
        clone.chain_number = self.chain_number
        clone.pending_garbage = self.pending_garbage
        return clone

    def encode(self) -> Dict:
        return {
            "deals": self.encode_deals(),
            "field": self.encode_field(),
            "chain_number": self.chain_number,
            "pending_score": self.chain_score + self.step_score,
            "pending_garbage": self.pending_garbage,
            "all_clear": int(self.all_clear_pending),
        }

    def render(self, outfile=sys.stdout, in_place=False) -> None:
        if not in_place:
            for _ in range(self.height + 1):
                outfile.write("\n")
            util.print_up(self.height + 1, outfile=outfile)
        super(VersusState, self).render(outfile=outfile, in_place=True)
        status_text = "x{} c{} s{} p{} {}".format(
            self.chain_number,
            self.chain_score,
            self.step_score,
            self.pending_garbage,
            "!" if self.all_clear_pending else ""
        )
        outfile.write(status_text)
        util.print_down(1)
        util.print_back(len(status_text))

    def step(self, x: int, orientation: int) -> Tuple[int, bool]:
        if not self.chain_number:
            if not self.deals:
                return 0, True
            if not self.validate_action(x, orientation):
                return 0, True

            self.play_deal(x, orientation)
            self.step_score += self.step_bonus

        had_chain: bool = bool(self.chain_number)

        iterations: int = self.field.handle_gravity()
        fell: bool = (iterations > 1)

        score: int = self.field.clear_groups(self.chain_number)
        if score:
            self.chain_score += score
            self.chain_number += 1

        released_garbage = 0
        if had_chain and not (fell or score):
            self.chain_number = 0

            self.chain_score += self.step_score
            self.step_score = 0

            if self.all_clear_pending:
                self.chain_score += self.all_clear_bonus
            self.all_clear_pending = False

            if self.chain_score >= 0:
                released_garbage, self.chain_score = divmod(self.chain_score, self.target_score)

            if not any(self.field.data):
                self.all_clear_pending = True

        if self.pending_garbage <= released_garbage:
            released_garbage -= self.pending_garbage
            self.pending_garbage = 0
        else:
            self.pending_garbage -= released_garbage
            released_garbage = 0

        if not self.chain_number:
            amount: int = self.pending_garbage
            if amount > self.max_received_garbage:
                amount = self.max_received_garbage
            self.pending_garbage -= amount
            self.add_garbage(amount)
            # Make garbage above the ghost line dissapear
            if self.tsu_rules:
                score, chain = self.field.resolve()
                assert (not score)

        if self.TESTING:
            assert (self.field.sane)
        return released_garbage, False

    def get_children(self, complete: bool = False) -> List[Tuple[Optional[VersusState], int]]:
        result: List[Tuple[Optional[VersusState], int]] = []
        for action in self.actions:
            child = self.clone()
            released_garbage, done = child.step(*action)
            if done and complete:
                result.append((None, released_garbage))
            else:
                result.append((child, released_garbage))
        return result

    def get_action_mask(self) -> int:
        if self.chain_number:
            return np.ones(len(self.actions))
        return super(VersusState, self).get_action_mask()


class Game(object):
    def __init__(self, state_params: Optional[Dict[str, int]], num_players: int = 2, seed=None) -> None:
        self._seed: int = 0
        _, self._seed = seeding.np_random(seed)
        self.game_over: bool = False
        if state_params is None:
            return
        params = {"seed": self._seed}
        params.update(state_params)
        self.players: List[VersusState] = []
        for _ in range(num_players):
            self.players.append(VersusState(**params))

    def render(self, outfile=sys.stdout, in_place=False) -> bool:
        height: int = self.players[0].height
        width: int = self.players[0].width
        if not in_place:
            for _ in range(height + 1):
                outfile.write("\n")
            util.print_up(height + 1, outfile=outfile)
        for player in self.players:
            player.render(outfile=outfile, in_place=True)
            util.print_up(height + 1)
            util.print_forward(2 * width + 9)
        util.print_down(height + 1)
        util.print_back(len(self.players) * (2 * width + 9))
        outfile.flush()

    def step(self, player_actions: List[Tuple[int, int]]) -> Tuple[int, int, bool]:
        """
        Return (result, garbage sent, done), tuple
        """
        if self.game_over:
            return 0, 0, True
        garbages: List[int] = []
        dones: List[bool] = []
        for player, action in zip(self.players, player_actions):
            amount, done = player.step(*action)
            garbages.append(amount)
            dones.append(done)

        if any(dones):
            self.game_over = True
            player_1_lost = dones[0]
            an_opponent_lost = any(dones[1:])
            if player_1_lost:
                if an_opponent_lost:
                    return 0, 0, True
                return -1, 0, True
            return 1, 0, True

        offset = min(garbages)

        for i, amount in enumerate(garbages):
            amount -= offset
            for j, opponent in enumerate(self.players):
                if i == j:
                    continue
                opponent.pending_garbage += amount
        garbage_sent = garbages[0] - sum(garbages[1:])
        return 0, garbage_sent, False

    def seed(self, seed=None) -> int:
        seed: int = self.players[0].seed(seed)
        for player in self.players[1:]:
            player.seed(seed)
        return seed

    def reset(self) -> None:
        self.game_over: bool = False
        self._seed: int = self.players[0].np_random.randint(0, 1234567890)
        for player in self.players:
            player.seed(self._seed)
            player.reset()

    def encode(self):
        return [p.encode() for p in self.players]

    def clone(self) -> Game:
        clone: Game = Game(None)
        clone.game_over = self.game_over
        clone.players = [p.clone() for p in self.players]
        return clone
