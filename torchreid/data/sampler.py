from __future__ import division, absolute_import

import copy
import numpy as np
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
from tools.prepare_lookups_for_hierarchical_sampling import create_lookupdict

AVAI_SAMPLERS = [
    'RandomIdentitySampler', 'SequentialSampler', 'RandomSampler',
    'RandomDomainSampler', 'RandomDatasetSampler', 'RandomIdentitySampler_Hierarchical'
]

class RandomIdentitySampler_Hierarchical(Sampler):
    """Hierarchically samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        list_of_files = []
        for index, items in enumerate(data_source):
            pid = items[1]
            self.index_dic[pid].append(index)
            list_of_files.append(items[0])
        self.pids = list(self.index_dic.keys())
        assert len(self.pids) >= self.num_pids_per_batch

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        # We cannot prepare lookupdict upfront since we might be training with a subset of pids
        # So create it on the fly
        lookupdict = create_lookupdict(list_of_files, data_source)

        self.pid_action = lookupdict['pid_action']
        self.action_pid = lookupdict['action_pid']
        self.pid_match = lookupdict['pid_match']
        self.match_pid = lookupdict['match_pid']
        self.pid_team_year_league = lookupdict['pid_team_year_league']
        self.team_year_league_pid = lookupdict['team_year_league_pid']
        self.pid_team_league = lookupdict['pid_team_league']
        self.team_league_pid = lookupdict['team_league_pid']
        self.pid_atleast_one_team_year_league = lookupdict['pid_atleast_one_team_year_league']
        self.atleast_one_team_year_league_pid = lookupdict['atleast_one_team_year_league_pid']
        self.pid_atleast_one_team_league = lookupdict['pid_atleast_one_team_league']
        self.atleast_one_team_league_pid = lookupdict['atleast_one_team_league_pid']


    def sample_next_hierarchy(self, selected_pid, selected_pids, pid_to_some_identifier, some_identifier_to_pid, selected_pids_extra = None):
        '''
        A function that samples a specified hierarchy level for pids
        :param selected_pid:
        :param selected_pids:
        :param pid_to_some_identifier:
        :param some_identifier_to_pid:
        :param selected_pids_extra:
        :return:
        '''
        # We find the extra pids not in selected_pids
        if selected_pids_extra is None:
            some_identifier = pid_to_some_identifier[selected_pid]
            # It is possible that in some cases some_identifier is a set
            if isinstance(some_identifier, set):
                selected_pids_extra = set()
                for each_identifier in some_identifier:
                    selected_pids_extra = selected_pids_extra.union(some_identifier_to_pid[each_identifier])
                selected_pids_extra = selected_pids_extra - selected_pids
            else:
                selected_pids_extra = some_identifier_to_pid[some_identifier] - selected_pids

        # We convert selected_pids_extra to a list so that
        # we can sample at random. set does not allow random sampling
        selected_pids_extra = list(selected_pids_extra)
        while len(selected_pids_extra) > 0 and len(selected_pids) < self.num_pids_per_batch:
            ridx = random.randrange(len(selected_pids_extra))
            # Swap the element at ridx and the last element of selected_pids_extra
            # popping list is very fast
            selected_pids_extra[ridx], selected_pids_extra[-1] = selected_pids_extra[-1], selected_pids_extra[ridx]
            selected_pids.add(selected_pids_extra.pop())

    def __iter__(self):
        # random.seed(0) ## Set this if you want same order every epoch
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        # Make a deepcopy of variables that we will modify in the loop below
        avai_pids = copy.deepcopy(self.pids)
        action_pid = copy.deepcopy(self.action_pid)
        match_pid = copy.deepcopy(self.match_pid)
        team_year_league_pid = copy.deepcopy(self.team_year_league_pid)
        team_league_pid = copy.deepcopy(self.team_league_pid)
        atleast_one_team_year_league_pid = copy.deepcopy(self.atleast_one_team_year_league_pid)
        atleast_one_team_league_pid = copy.deepcopy(self.atleast_one_team_league_pid)

        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:

            # Randomly pick a pid
            selected_pid = random.sample(avai_pids, 1)[0]

            # For this pid, hierarchically select other pids
            # First we try to select from the same action
            # Note that selected_pids will contain selected_pid
            # and that it is a set, so no idx is repeated
            # Make a deep copy
            selected_pids = copy.deepcopy(action_pid[self.pid_action[selected_pid]])

            if len(selected_pids) > self.num_pids_per_batch:
                # We will hit this iff the original action contains self.num_pids_per_batch elements
                # So we can pop any random element without worry as they are all from the
                # same action
                # Randomly choose these extra pids
                extra_pids = random.sample(selected_pids, len(selected_pids) - self.num_pids_per_batch)
                for extra_p in extra_pids:
                    selected_pids.remove(extra_p)

            if len(selected_pids) < self.num_pids_per_batch:
                # If we do not have enough images from the same action
                # for one batch, then we can select from the same match
                self.sample_next_hierarchy(selected_pid, selected_pids, self.pid_match,
                                           match_pid)

            if len(selected_pids) < self.num_pids_per_batch:
                # Third we can select from any other matches between the same pair of teams in this year in the same
                # league. Maybe a second-leg
                self.sample_next_hierarchy(selected_pid, selected_pids, self.pid_team_year_league,
                                           team_year_league_pid)

            if len(selected_pids) < self.num_pids_per_batch:
                # Fourth we can select from matches between the same pair of teams across years in the same league
                # Reasoning is that jersey colors do not change much over the years. Arsenal always
                # plays in some form of red in epl.
                self.sample_next_hierarchy(selected_pid, selected_pids, self.pid_team_league,
                                           team_league_pid)

            if len(selected_pids) < self.num_pids_per_batch:
                # I do not believe we will hit this step. But if we do, then sample from
                # any match involving atleast one of the two teams in the same year in the same
                # league
                self.sample_next_hierarchy(selected_pid, selected_pids, self.pid_atleast_one_team_year_league,
                                           atleast_one_team_year_league_pid)

            if len(selected_pids) < self.num_pids_per_batch:
                # I do not believe we will hit this step. But if we do, then sample from
                # any match involving atleast one of the two teams across years in the same
                # league
                self.sample_next_hierarchy(selected_pid, selected_pids, self.pid_atleast_one_team_league,
                                           atleast_one_team_league_pid)

            # In this case, just sample from all remaining available stuff
            if len(selected_pids) < self.num_pids_per_batch:
                self.sample_next_hierarchy(selected_pid, selected_pids, None,
                                           None, selected_pids_extra=set(avai_pids) - selected_pids)

            if len(selected_pids) < self.num_pids_per_batch:
                # Then just remove this pid from all relevant dicts
                # and continue to the next random pid
                avai_pids.remove(selected_pid)
                # Also pop selected pid from all the lookup dictionaries
                action_pid[self.pid_action[selected_pid]].remove(selected_pid)
                match_pid[self.pid_match[selected_pid]].remove(selected_pid)
                team_year_league_pid[self.pid_team_year_league[selected_pid]].remove(selected_pid)
                team_league_pid[self.pid_team_league[selected_pid]].remove(selected_pid)
                for identifier in self.pid_atleast_one_team_year_league[selected_pid]:
                    atleast_one_team_year_league_pid[identifier].remove(selected_pid)
                for identifier in self.pid_atleast_one_team_league[selected_pid]:
                    atleast_one_team_league_pid[identifier].remove(selected_pid)
                continue

            # At this point selected_pids will have exactly self.num_pids_per_batch elements
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    # Remove this pid from everything as it is not useful anymore
                    avai_pids.remove(pid)
                    # Also pop selected pid from all the lookup dictionaries
                    action_pid[self.pid_action[pid]].remove(pid)
                    match_pid[self.pid_match[pid]].remove(pid)
                    team_year_league_pid[self.pid_team_year_league[pid]].remove(pid)
                    team_league_pid[self.pid_team_league[pid]].remove(pid)
                    for identifier in self.pid_atleast_one_team_year_league[pid]:
                        atleast_one_team_year_league_pid[identifier].remove(pid)
                    for identifier in self.pid_atleast_one_team_league[pid]:
                        atleast_one_team_league_pid[identifier].remove(pid)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, items in enumerate(data_source):
            pid = items[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        assert len(self.pids) >= self.num_pids_per_batch

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomDomainSampler(Sampler):
    """Random domain sampler.

    We consider each camera as a visual domain.

    How does the sampling work:
    1. Randomly sample N cameras (based on the "camid" label).
    2. From each camera, randomly sample K images.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        n_domain (int): number of cameras to sample in a batch.
    """

    def __init__(self, data_source, batch_size, n_domain):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, items in enumerate(data_source):
            camid = items[2]
            self.domain_dict[camid].append(i)
        self.domains = list(self.domain_dict.keys())

        # Make sure each domain can be assigned an equal number of images
        if n_domain is None or n_domain <= 0:
            n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain

        self.batch_size = batch_size
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_domains = random.sample(self.domains, self.n_domain)

            for domain in selected_domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomDatasetSampler(Sampler):
    """Random dataset sampler.

    How does the sampling work:
    1. Randomly sample N datasets (based on the "dsetid" label).
    2. From each dataset, randomly sample K images.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        n_dataset (int): number of datasets to sample in a batch.
    """

    def __init__(self, data_source, batch_size, n_dataset):
        self.data_source = data_source

        # Keep track of image indices for each dataset
        self.dataset_dict = defaultdict(list)
        for i, items in enumerate(data_source):
            dsetid = items[3]
            self.dataset_dict[dsetid].append(i)
        self.datasets = list(self.dataset_dict.keys())

        # Make sure each dataset can be assigned an equal number of images
        if n_dataset is None or n_dataset <= 0:
            n_dataset = len(self.datasets)
        assert batch_size % n_dataset == 0
        self.n_img_per_dset = batch_size // n_dataset

        self.batch_size = batch_size
        self.n_dataset = n_dataset
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        dataset_dict = copy.deepcopy(self.dataset_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_datasets = random.sample(self.datasets, self.n_dataset)

            for dset in selected_datasets:
                idxs = dataset_dict[dset]
                selected_idxs = random.sample(idxs, self.n_img_per_dset)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    dataset_dict[dset].remove(idx)

                remaining = len(dataset_dict[dset])
                if remaining < self.n_img_per_dset:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_train_sampler(
    data_source,
    train_sampler,
    batch_size=32,
    num_instances=4,
    num_cams=1,
    num_datasets=1,
    **kwargs
):
    """Builds a training sampler.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        train_sampler (str): sampler name (default: ``RandomSampler``).
        batch_size (int, optional): batch size. Default is 32.
        num_instances (int, optional): number of instances per identity in a
            batch (when using ``RandomIdentitySampler``). Default is 4.
        num_cams (int, optional): number of cameras to sample in a batch (when using
            ``RandomDomainSampler``). Default is 1.
        num_datasets (int, optional): number of datasets to sample in a batch (when
            using ``RandomDatasetSampler``). Default is 1.
    """
    assert train_sampler in AVAI_SAMPLERS, \
        'train_sampler must be one of {}, but got {}'.format(AVAI_SAMPLERS, train_sampler)

    if train_sampler == 'RandomIdentitySampler':
        sampler = RandomIdentitySampler(data_source, batch_size, num_instances)

    if train_sampler == 'RandomIdentitySampler_Hierarchical':
        sampler = RandomIdentitySampler_Hierarchical(data_source, batch_size, num_instances)

    elif train_sampler == 'RandomDomainSampler':
        sampler = RandomDomainSampler(data_source, batch_size, num_cams)

    elif train_sampler == 'RandomDatasetSampler':
        sampler = RandomDatasetSampler(data_source, batch_size, num_datasets)

    elif train_sampler == 'SequentialSampler':
        sampler = SequentialSampler(data_source)

    elif train_sampler == 'RandomSampler':
        sampler = RandomSampler(data_source)

    return sampler
