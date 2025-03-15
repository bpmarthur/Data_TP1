#! /usr/bin/env python3
import sys
import unittest
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from TD import nearest_neighbor, LinearScan, KDTree
from TD.kdtree import median as med, partition as part, Node
from scripts.iris import main, error_rate, iris
from timeit import timeit

"""
Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 8,
  "names" : [
      "nearest_neighbor.py::nearest_neighbor",
      "linear_scan.py::linear_scan",
      "kdtree.py::median",
      "kdtree.py::partition",
      "kdtree.py::kdtree",
      "kdtree.py::defeatist",
      "kdtree.py::backtracking",
      "kdtree.py::xaggle"
      ],
  "points" : [10, 10, 10, 10, 20, 10, 20, 10]
}
[END-AUTOGRADER-ANNOTATION]
"""


def print_help():
    print(
        "./grader script. Usage: ./grader.py test_number, e.g. ./grader.py 1 for 1st exercise."
    )
    print("N.B.: ./grader.py 0 will run all tests.}")
    print(f"You provided {sys.argv}")
    exit(1)


epsilon = 1e-6
simple_array = lambda x=10: np.array(list(range(x))).reshape((-1, 1))


def test_implementation(implementation: Type[LinearScan | KDTree], length, *args):
    impl = implementation(simple_array(length))
    for x in range(length - 1):
        assert (
            impl.query(np.array([x + 0.5 + epsilon]), *args)[1] == x + 1
        ), f"Query point {x + 0.5 + epsilon} should have returned index {x + 1}"


def naive_tests():
    msg = "got naive distance tests wrong"
    assert np.isclose(nearest_neighbor.euclidean_distance(np.array([1]), np.array([1])), 0), msg
    assert np.isclose(
        nearest_neighbor.euclidean_distance(np.array([1, 1]), np.array([2, 2])),
        1.414, atol=1e-3, rtol=1e-6
    ), msg
    assert np.isclose(
        nearest_neighbor.euclidean_distance(np.array([0, 1]), np.array([4, 1])), 4
    ), msg


def read_dist(filepath: Path):
    msg = "incorrect distance on some generated data: {} {} dist should be {}"
    all = np.loadtxt(filepath)
    X1 = all[:, :(all.shape[1] // 2)]
    X2 = all[:, (all.shape[1] // 2):]
    dist = X2[:, -1]
    X2 = X2[:, :-1]
    for x1, x2, d in zip(X1, X2, dist):
        assert np.isclose(nearest_neighbor.euclidean_distance(x1, x2), d, atol=1e-3, rtol=1e-6).all(), \
            msg.format(x1, x2, d)


def read_median(filepath: Path):
    msg = "incorrect median on some generated data, median from {} to {} on coord {} should be {}, you returned {}"
    X = np.loadtxt(filepath, max_rows=100)
    medians = np.loadtxt(filepath, skiprows=100)
    for median in medians:
        your_median = med(X, int(median[0]), int(median[1]), int(median[2]))
        assert your_median == median[3], \
            msg.format(*median.tolist(), your_median)


def test_speed(mode: str = "backtracking"):
    X_train, X_test, _, _ = iris()
    kdtree = KDTree(X_train)
    [kdtree.query(x, mode) for x in X_test]


def count_points(node: Node):
    if node is None:
        return 0
    return 1 + count_points(node.left) + count_points(node.right)


def track_indices(node: Node, indices = None):
    if node is None:
        return
    indices.append(node.idx)
    track_indices(node.left, indices)
    track_indices(node.right, indices)


class Grader(unittest.TestCase):
    def nearest_neighbor(self):
        naive_tests()
        [read_dist(file) for file in (Path(__file__).parent / "tests").glob("dist*.dat")]

    def linear_scan(self):
        # "Unit" tests
        lscan = LinearScan(simple_array())
        for i in range(10):
            assert lscan.query(np.array([i])) == (0, i),\
                f"Wrong output for query point {i}, distance should be 0, NN should be {i}"
        assert lscan.query(np.array([i + 0.5])) == (0.5, i), \
            f"Wrong output for query point {i + 0.5}, distance should be 0.5, NN should be {i}"
        # "Functional" test
        assert np.isclose(
            timeit(lambda: test_implementation(LinearScan, 50), number=100),
            100 * timeit(lambda: test_implementation(LinearScan, 5), number=100),
            rtol=0.2,
        ), "Non-linear scaling of test query time"

    def median(self):
        X, _ = load_iris(return_X_y=True, as_frame=False)
        X_copied = np.copy(X)
        _ = med(X, 0, X.shape[0], 0)
        assert (X == X_copied).all(), "Side effect in median"
        [read_median(file) for file in (Path(__file__).parent / "tests").glob("median*.dat")]

    def partition(self):
        X, _ = load_iris(return_X_y=True, as_frame=False)
        for coord in range(4):
            median = med(X, 0, X.shape[0], coord)
            idx = part(X, 0, X.shape[0], coord)
            assert X[idx, coord] == median, f"Incorrect median at {idx} is {X[idx, 0]}, should be {median} for coord {coord}"
            assert (X[:idx, coord] <= median).all(), "All before median are not less than median for coord {coord}"

    def kdtree(self):
        X, _ = load_iris(return_X_y=True, as_frame=False)
        kdtree = KDTree(X)
        assert isinstance(kdtree.root, Node), "Root is not a Node"
        assert count_points(kdtree.root) == X.shape[0], "Not as many nodes as there are points"
        indices = []
        track_indices(kdtree.root, indices)
        indices.sort()
        assert indices == list(range(X.shape[0])), "All indices are not in the tree - there might be duplicates"

    def defeatist(self):
        X_train, X_test, _, _ = iris()
        kdtree = KDTree(X_train)
        ls = LinearScan(X_train)
        wrong = 0
        for x in X_test:
            _, true_dist = ls.query(x)
            _, defeatist_dist = kdtree.query(x, "defeatist")
            if defeatist_dist > true_dist:
                wrong += 1
        print(f"\nDefeatist was wrong {wrong} times over {X_test.shape[0]}")

    def backtracking(self):
        assert np.isclose(
            timeit(lambda: test_implementation(KDTree, 100), number=10),
            20 * timeit(lambda: test_implementation(KDTree, 10), number=10),
            rtol=0.5, atol=10
        ), "Non-log scaling of test query time"

        defeat = timeit(lambda: test_speed("defeatist"), number=10)
        back = timeit(lambda: test_speed("backtracking"), number=10)
        print(f"\nDefeatist was {back/defeat:.2f}x faster.")

        main()

    def xaggle(self):
        try:
            private = pd.read_csv(
                next(Path(__file__).parent.parent.parent.rglob("private_test_set.*")),
                header=None,
            ).values
            X_test, y_test = private[:, :4], private[:, 4]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e}: you probably don't have access to the private test set!"
            )
        X, _ = load_iris(return_X_y=True, as_frame=False)
        impl = KDTree(X)
        impl.set_xaggle_config()  # this sets impl.mode, see below
        print(f"You chose {impl.mode} search")
        y_hat_test = []
        for x in X_test:
            # For each x_test, y_hat_test is x_test's nearest neighbor's y_test
            y_hat_test.append(y_test[impl.query(x, impl.mode)[1]])
        assert (
            error_rate(y_test, np.array(y_hat_test)) < 0.45
        ), f"Not good enough on private test set for {impl.mode} search, try again."


def suite(test_nb):
    suite = unittest.TestSuite()
    test_name = [
        "nearest_neighbor",
        "linear_scan",
        "median",
        "partition",
        "kdtree",
        "defeatist",
        "backtracking",
        "xaggle",
    ]

    if test_nb > 0:
        suite.addTest(Grader(test_name[test_nb - 1]))
    else:
        for name in test_name:
            suite.addTest(Grader(name))

    return suite


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
    try:
        test_nb = int(sys.argv[1])
    except ValueError as e:
        print(
            f"You probably didn't pass an int to ./grader.py: passed {sys.argv[1]}; error {e}"
        )
        exit(1)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite(test_nb))
