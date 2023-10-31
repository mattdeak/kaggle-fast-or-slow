import random
from typing import Iterator


class ConfigCrossoverBatchSampler:
    def __init__(
        self,
        groups: list[list[int]],
        batch_size: int,
        shuffle_groups: bool = True,
        shuffle_within_groups: bool = True,
        out_of_config_crossover_prob: float = 0.1,
    ):
        # TODO: implement crossover. it's my idea, theres no link
        self.groups = groups
        self.batch_size = batch_size
        self.shuffle = shuffle_groups
        self.shuffle_within_groups = shuffle_within_groups
        self.out_of_config_crossover_prob = out_of_config_crossover_prob

        self.batch_list: list[list[int]] = self.get_batches()

    def get_batches(self) -> list[list[int]]:
        """Returns a list of batches, where each batch is a list of indices."""
        if self.shuffle:
            random.shuffle(self.groups)

        batch_list: list[list[int]] = []

        for group in self.groups:
            if self.shuffle_within_groups:
                random.shuffle(group)

            for i in range(0, len(group), self.batch_size):
                new_batch = group[i : i + self.batch_size]

                # We're going to just drop it if it's not the right size.
                if len(new_batch) == self.batch_size:
                    batch_list.append(new_batch)

        random.shuffle(batch_list)

        # apply crossover
        # for each batch, each element has a chance of being replaced by an element from another batch
        if self.out_of_config_crossover_prob > 0:
            return self.apply_crossover(batch_list)

        return batch_list

    def apply_crossover(self, batch_list: list[list[int]]) -> list[list[int]]:
        for batch in batch_list:
            for i in range(len(batch)):
                if random.random() < self.out_of_config_crossover_prob:
                    # choose a random batch
                    random_batch = random.choice(batch_list)
                    # choose a random element from that batch
                    random_element = random.choice(random_batch)
                    # replace the current element with the random element
                    batch[i] = random_element

        return batch_list

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.batch_list:
            yield batch

    def __len__(self):
        return len(self.batch_list)
