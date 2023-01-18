import copy
import dataclasses
import itertools
import logging
from collections.abc import Collection, Hashable, Iterable, Sequence
from functools import cmp_to_key
from typing import Optional

import networkx as nx
import numpy as np
import scipy.linalg
import scipy.sparse.linalg as linalg

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)
fmt = logging.Formatter(
    r'%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    r'%Y/%m/%d %H:%M:%S'
)
handler.setFormatter(fmt)


def _put_relation(matrix, i, j, antisymmetric):
    if matrix[i, j] < 0:
        return False
    matrix[i, j] = 1

    if antisymmetric:
        if matrix[j, i] > 0:
            return False
        matrix[j, i] = -1

    return True


def _transitive_closure_recursion(matrix, source_index, reached, done,
                                  order_relation, antisymmetric):
    if done[source_index]:
        return True

    n = matrix.shape[0]
    for target_index in range(n):
        if matrix[source_index, target_index] <= 0:
            continue

        if order_relation and reached[target_index]:
            return False

        if not (done[target_index] or reached[target_index]):
            reached_for_recursion = reached.copy()
            reached_for_recursion[target_index] = True
            result = _transitive_closure_recursion(
                matrix, target_index, reached_for_recursion,
                done, order_relation, antisymmetric
            )
            if not result:
                return False

        result = _put_relation(matrix, source_index,
                               target_index, antisymmetric)
        if not result:
            return False

        for i in range(n):
            if matrix[target_index, i] > 0:
                result = _put_relation(matrix, source_index,
                                       i, antisymmetric)
                if not result:
                    return False

    return True


def transitive_closure(matrix, order_relation=True, antisymmetric=False):
    n = matrix.shape[0]
    reached = np.zeros((n,), dtype=bool)
    done = np.zeros((n,), dtype=bool)
    for i in range(n):
        reached[:] = False
        reached[i] = True
        result = _transitive_closure_recursion(matrix, i, reached, done,
                                               order_relation, antisymmetric)
        if not result:
            return False

        done[i] = True
        if order_relation:
            done[:] = np.maximum(matrix[i, :], done)

    return True


class MatrixGenerator:
    def __init__(self, name_set):
        name_list = sorted(name_set)
        self.name2index = {
            name: i for i, name in enumerate(name_list)
        }

        self.index2name = {num: name for name, num in self.name2index.items()}
        self.size = len(self.name2index.keys())

    def _make_matrix(self, name_seq, antisymmetric):
        order = [self.name2index[elem] for elem in name_seq]
        retval = np.zeros((self.size, self.size), dtype=int)
        zipped = zip(order[:-1], order[1:])
        for i, j in zipped:
            retval[i, j] = 1

        result = transitive_closure(retval, antisymmetric=antisymmetric,
                                    order_relation=True)
        if not result:
            logger.warning("warning: make_matrix failed")

        return retval

    def make_increment_matrix(self, name_seq):
        return self._make_matrix(name_seq, antisymmetric=False)

    def make_adjacency_matrix(self, name_seq):
        return self._make_matrix(name_seq, antisymmetric=True)

    def get_index(self, name_name):
        return self.name2index[name_name]

    def get_name(self, index):
        return self.index2name[index]

    # matrix given is assumed to be an adjacency matrix for total order
    def adjacency_matrix_to_name_sequence(self, matrix):
        index_list = list(range(self.size))
        index_list.sort(key=cmp_to_key(lambda i, j: matrix[j, i]))
        return tuple(self.index2name[index] for index in index_list)


@dataclasses.dataclass(frozen=True)
class Poset:
    elements: frozenset
    adjacency_matrix: np.ndarray

    @classmethod
    def make(cls, elements: Iterable, adjacency_matrix: np.ndarray):
        elements = frozenset(elements)
        adjacency_matrix = adjacency_matrix.copy()
        adjacency_matrix.flags.writeable = False
        return cls(elements, adjacency_matrix)


def _calculate_rank(name_seq_list, generator):
    if generator.size == 0:
        return None
    elif generator.size == 1:
        return {generator.get_name(0): 1.}

    mat = np.identity(generator.size, dtype=int)
    mat += np.ones_like(mat)
    mat += sum(
        generator.make_increment_matrix(name_seq)
        for name_seq in name_seq_list
    )

    mat_ = mat / mat.sum(axis=1, keepdims=True)
    if generator.size > 2:
        _, rank = linalg.eigs(mat_.T, k=1, return_eigenvectors=True)
    else:
        _, rank = scipy.linalg.eig(mat_.T, right=True)
        rank = rank[:, 0]
    rank = np.abs(rank)

    names = [generator.get_name(i) for i in range(generator.size)]
    ranking = {
        name: rank
        for name, rank in zip(names, rank)
    }
    ranking = dict(sorted(ranking.items(), key=lambda item: item[1]))

    return ranking


def _is_mergeable_pair(lhs, rhs):
    return np.all((lhs * rhs) >= 0)


def _merge_pair(lhs, rhs):
    return _merge_all([lhs, rhs])


# If matrices in matrix_list are pairwise mergeable,
# _merge_all returns a matirx if and only if the matrices
# are validly mergeable.
def _merge_all(matrix_list):
    if len(matrix_list) == 0:
        return None

    mask = np.any([mat == 0 for mat in matrix_list], axis=0)
    pos = np.any([mat > 0 for mat in matrix_list], axis=0)
    neg = np.any([mat < 0 for mat in matrix_list], axis=0)
    retval = matrix_list[0].copy()
    retval[mask & pos] = 1
    retval[mask & neg] = -1
    result = transitive_closure(retval, order_relation=True,
                                antisymmetric=True)
    if result:
        return retval
    else:
        return None


def _enumerate_maximal_recursion_naive(
    poset_dict, current_poset, candidate_, found_maximal
):
    candidate = candidate_.copy()

    if current_poset is None:
        elements = frozenset()
        element_mat = None
    else:
        elements = current_poset.elements
        element_mat = current_poset.adjacency_matrix
    canbe_maximal = True
    while True:
        if len(candidate) == 0:
            break

        target = candidate.pop()
        target_mat = poset_dict[target].adjacency_matrix

        merged = None
        if element_mat is None:
            new_elements = [target]
            merged = target_mat
            new_poset = Poset.make(new_elements, merged)
        elif _is_mergeable_pair(element_mat, target_mat):
            new_elements = list(current_poset.elements)
            new_elements.append(target)
            merged = _merge_pair(element_mat, target_mat)
            new_poset = Poset.make(new_elements, merged)

        if merged is not None:
            _enumerate_maximal_recursion_naive(
                poset_dict, new_poset, candidate, found_maximal
            )
            canbe_maximal = False

    if not canbe_maximal:
        return

    accept = True
    for already_found in found_maximal:
        if elements.issubset(already_found.elements):
            accept = False
            break
    if accept:
        found_maximal.append(current_poset)
    return


def _enumerate_maximal_recursion_canny(
    poset_dict, current_poset, candidate_,
    already_found_maximal, newly_found_maximal
):
    candidate = candidate_.copy()
    priority_candidate = candidate_.copy()
    if current_poset is not None:
        elements = current_poset.elements
        element_mat = current_poset.adjacency_matrix
    else:
        elements = frozenset()
        element_mat = None

    intersections = {
        frozenset(maximal_found.elements.intersection(candidate))
        for maximal_found in already_found_maximal + newly_found_maximal
    }
    including_current = {
        intersection for intersection in intersections
        if (len(elements) < len(intersection)
            and elements.issubset(intersection))
    }
    if len(including_current) > 0:
        less_priority_set = max(including_current, key=len)
        priority_candidate.difference_update(less_priority_set)

    canbe_maximal = True
    while True:
        if len(candidate) == 0:
            break
        elif len(priority_candidate) == 0:
            return

        target = priority_candidate.pop()
        candidate.remove(target)
        target_mat = poset_dict[target].adjacency_matrix
        merged = None
        if element_mat is None:
            new_elements = [target]
            merged = target_mat
            new_poset = Poset.make(new_elements, merged)
        elif _is_mergeable_pair(element_mat, target_mat):
            new_elements = list(elements)
            new_elements.append(target)
            merged = _merge_pair(element_mat, target_mat)
            new_poset = Poset.make(new_elements, merged)

        if merged is not None:
            _enumerate_maximal_recursion_canny(
                poset_dict, new_poset, candidate,
                already_found_maximal, newly_found_maximal
            )
            canbe_maximal = False
            if (len(candidate) == len(priority_candidate)):
                already_found_elements =\
                    {found.elements for found in newly_found_maximal}
                including_current = {
                    found_elements
                    for found_elements in already_found_elements
                    if (len(elements) < len(found_elements)
                        and elements.issubset(found_elements))
                }
                if len(including_current) > 0:
                    less_priority_set = max(including_current, key=len)
                    priority_candidate.difference_update(less_priority_set)

    if not canbe_maximal:
        return

    # reject the maximal found if it is included in
    # a maximal already found
    accept = True
    for already_found in already_found_maximal + newly_found_maximal:
        if elements.issubset(already_found.elements):
            accept = False
            break
    if not accept:
        return

    # remove the maximals found in the other cliques
    # which are included in a maximal found in the current clique
    not_included = [
        already_found for already_found in already_found_maximal
        if not already_found.elements.issubset(elements)
    ]
    if len(not_included) < len(already_found_maximal):
        already_found_maximal.clear()
        already_found_maximal.extend(not_included)
    newly_found_maximal.append(current_poset)
    return


def _enumerate_maximal_naive(poset_dict, already_found=[]):
    nodes = set(poset_dict.keys())
    maximal_list = copy.copy(already_found)
    _enumerate_maximal_recursion_naive(
        poset_dict, None, nodes, maximal_list
    )

    return maximal_list


def _enumerate_maximal_canny(poset_dict, already_found=[]):
    nodes = set(poset_dict.keys())
    maximal_list = copy.copy(already_found)
    newly_found = []
    _enumerate_maximal_recursion_canny(
        poset_dict, None, nodes, maximal_list, newly_found
    )

    maximal_list.extend(newly_found)
    return maximal_list


def _enumerate_maximal_clique(poset_dict):
    # maximal clique enumeration
    nodes = set(poset_dict.keys())
    edges = [(i, j) for i, j in itertools.combinations(nodes, 2)
             if _is_mergeable_pair(poset_dict[i].adjacency_matrix,
                                   poset_dict[j].adjacency_matrix)]
    graph = nx.Graph(edges)
    cliques = nx.find_cliques(graph)

    # enumerate mergeable maximal ordered set within each maximal clique
    maximal_list = []
    not_mergeable_cliques = []
    for clique in cliques:
        merged = _merge_all([poset_dict[node].adjacency_matrix
                             for node in clique])
        if merged is not None:
            clique_set = frozenset(clique)
            # remove the maximals found in the other cliques
            # which are included in this clique
            maximal_list = [
                maximal_found for maximal_found in maximal_list
                if not maximal_found.elements.issubset(clique_set)
            ]
            maximal_list.append(Poset.make(clique_set, merged))
        else:
            not_mergeable_cliques.append(clique)

    for clique in not_mergeable_cliques:
        seq_subdict = {node: poset_dict[node] for node in clique}
        maximal_list = _enumerate_maximal_canny(seq_subdict, maximal_list)

    return maximal_list


def make_maximal_totally_ordered(maximal_list, default_order):
    size = default_order[0].shape[0]
    identity = np.identity(size, dtype=int)
    return_list = []
    for maximal in maximal_list:
        if np.any((maximal.adjacency_matrix + identity) == 0):
            seq_dict_order = {0: maximal}
            seq_dict_order.update({
                (i + 1): Poset.make(frozenset((i + 1,)), mat)
                for i, mat in enumerate(default_order)
            })
            candidate_ = set(range(1, len(default_order) + 1))
            interpolated_list = []
            poset = Poset.make((0,), maximal.adjacency_matrix)
            _enumerate_maximal_recursion_canny(
                seq_dict_order, poset, candidate_, [], interpolated_list
            )
            return_list.extend((
                Poset.make(maximal.elements, interpolated.adjacency_matrix)
                for interpolated in interpolated_list
            ))
        else:
            return_list.append(maximal)

    return return_list


class EnumerateMaximal:
    def __init__(
        self,
        name_seq_list: Collection[Sequence[Hashable]],
        default_order: Optional[Sequence[Hashable]] = None
    ):
        name_seq_set = set(tuple(name_seq) for name_seq in name_seq_list)
        self.unique_name_seq_list = list(name_seq_set)
        self.unique_name_seq_list.sort()
        name_set = set(name for name_seq in name_seq_set for name in name_seq)
        if len(self.unique_name_seq_list) > 0:
            self.generator = MatrixGenerator(name_set)
            self.poset_dict = {
                i: Poset.make(
                    (i,), self.generator.make_adjacency_matrix(seq_tuple)
                )
                for i, seq_tuple in enumerate(self.unique_name_seq_list)
            }

            if default_order is None:
                rank_dict = _calculate_rank(name_seq_list, self.generator)
                ranking_seq = tuple(rank_dict.keys())
            else:
                ranking_seq =\
                    tuple(name for name in default_order if name in name_set)
                if len(ranking_seq) < len(name_set):
                    logger.error("error: default_order specified "
                                 "does not give total order")
                    raise ValueError(
                        f"specified default_order {default_order} "
                        f"does not give total order on {name_set}"
                    )
            indices = [(i, j) if i < j else (j, i) for i, j in
                       itertools.combinations(range(len(ranking_seq)), 2)]
            self.rank_order = [
                self.generator.make_adjacency_matrix(
                    (ranking_seq[i], ranking_seq[j])
                )
                for i, j in indices
            ]
        else:
            self.generator = None
            self.rank_order = None

    def enumerate_totally_ordered_maximal_subset(self, method="canny",
                                                 return_membership=False):
        if self.generator is None or self.generator.size == 0:
            return []
        elif self.generator.size == 1:
            name = self.generator.get_name(0)
            return [[name]]
        elif len(self.unique_name_seq_list) == 1:
            return [[self.unique_name_seq_list[0]]]

        if method == "naive":
            enumerate_mehod = _enumerate_maximal_naive
        elif method == "canny":
            enumerate_mehod = _enumerate_maximal_canny
        elif method == "clique":
            enumerate_mehod = _enumerate_maximal_clique
        else:
            logger.error("error: unknown method type specified")
            raise ValueError(f"unknown method type {method}")

        maximal_poset_list = enumerate_mehod(self.poset_dict)
        maximal_poset_list = make_maximal_totally_ordered(
            maximal_poset_list, self.rank_order
        )
        to_seq = self.generator.adjacency_matrix_to_name_sequence
        if return_membership:
            maximal_seq_list = []
            for poset in maximal_poset_list:
                seqence = to_seq(poset.adjacency_matrix)
                consistent =\
                    [self.unique_name_seq_list[i] for i in poset.elements]
                consistent.sort()
                inconsistent =\
                    [self.unique_name_seq_list[i] for i
                     in set(self.poset_dict.keys()).difference(poset.elements)]
                inconsistent.sort()
                maximal_seq_list.append({
                    "sequence": seqence,
                    "consistent": consistent,
                    "inconsistent": inconsistent
                })
            maximal_seq_list.sort(key=lambda x: x["sequence"])
        else:
            maximal_seq_list = [
                to_seq(poset.adjacency_matrix) for poset in maximal_poset_list
            ]
            maximal_seq_list.sort()

        return maximal_seq_list
