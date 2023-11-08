import heapq
import os
from typing import Any, Self

import torch


class Checkpointer:
    def __init__(
        self,
        checkpoint_dir: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        max_checkpoints: int = 5,
    ):
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.max_checkpoints = max_checkpoints
        self._heap: list[tuple[_ComparableIter, str]] = []

        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler

    def get_most_recent_checkpoint(self) -> dict[str, Any] | None:
        """Returns the state dicts at the most recent checkpoint."""
        checkpoints = os.listdir(self.checkpoint_dir)
        # Extract latest
        if checkpoints:
            sorted_checkpoints = sorted(
                checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )

            most_recent_checkpoint = sorted_checkpoints[-1]
            print("Loading checkpoint:", most_recent_checkpoint)

            checkpoint = torch.load(  # type: ignore
                os.path.join(self.checkpoint_dir, most_recent_checkpoint)
            )
            return checkpoint  # type: ignore
        else:
            print("No Checkpoint Found in:", self.checkpoint_dir)
            return None

    def save_checkpoint(self, iteration: int, epoch: int | None = None) -> None:
        """Saves a checkpoint."""
        if epoch is not None:
            checkpoint_name = f"checkpoint_{epoch}_{iteration}.pt"
        else:
            checkpoint_name = f"checkpoint_{iteration}.pt"

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "iteration": iteration,
        }

        if epoch:
            checkpoint["epoch"] = epoch
        if self._scheduler:
            checkpoint["scheduler_state_dict"] = self._scheduler.state_dict()

        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)  # type: ignore

        if len(self._heap) > self.max_checkpoints:
            _, oldest_checkpoint = heapq.heappop(self._heap)
            os.remove(os.path.join(self.checkpoint_dir, oldest_checkpoint))

        neg_epoch = -epoch if epoch is not None else epoch

        # negation is used because heapq is a min heap
        iter_count = _ComparableIter(-iteration, neg_epoch)
        heapq.heappush(self._heap, (iter_count, checkpoint_filename))  # type: ignore

    def load_checkpoint(self, state_dict: dict[str, Any]) -> dict[str, Any] | None:
        """Loads the most recent checkpoint.

        Also returns it.

        """
        self._model.load_state_dict(state_dict["model_state_dict"])
        self._optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        if self._scheduler:
            self._scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        return state_dict


class _ComparableIter:
    def __init__(self, iteration: int, epoch: int | None = None):
        self.iteration = iteration
        self.epoch = epoch

    def __eq__(self, other: Self) -> bool:
        return self.iteration == other.iteration and self.epoch == other.epoch

    def __lt__(self, other: Self) -> bool:
        if self.epoch is None or other.epoch is None:
            return self.iteration < other.iteration
        else:
            if self.epoch == other.epoch:
                return self.iteration < other.iteration
            return self.epoch < other.epoch

    def __gt__(self, other: Self) -> bool:
        return not self.__lt__(other)
