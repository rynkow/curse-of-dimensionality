import math
import os
import numpy as np
import matplotlib.pyplot as plt


def fill_hypercube(dimension, no_of_points):
    """
    Uniformly fills a hypercube with a desired number of points.
    Arguments:
    dimension: hypercube dimension
    no_of_points: desired no of points
    Returns:
        list of points filling the hypercube
    """
    points = [np.random.uniform(0, 1, dimension) for i in range(no_of_points)]
    return points


def choose_vector(points):
    """
    Get random vector by choosing and removing 2 points from a given list of points.
    Arguments:
    points: list of points
    Returns:
        vector
    """
    start = points.pop(np.random.randint(0, len(points)))
    end = points.pop(np.random.randint(0, len(points)))
    vector = end - start
    return vector


def choose_point(points):
    """
    Get random point by choosing and removing a point from a given list of points.
    Arguments:
    points: list of points
    Returns:
        randomly chosen point
    """
    return points.pop(np.random.randint(0, len(points)))


def angle_between_vectors(vector1, vector2):
    """
    Calculate an angle between two vectors
    Arguments:
    vector1: a vector
    vector2: a vector
    Returns:
        angle between vector1 and vector2
    """
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot_product)
    return angle


def euclidean_distance(point1, point2):
    """
    Calculate euclidean distance between two points
    Arguments:
    point1: a point
    point2: a point
    Returns:
        euclidean distance between point1 and point1
    """
    distance = point1 - point2
    distance = np.power(distance, 2)
    distance = np.sqrt(np.sum(distance))
    return distance


def random_angles_experiment(tested_dimensions, no_of_points, no_of_chosen_vector_pairs, output_dir):
    """
    Check how angles between vectors, given by selecting random points in filled hypercube, change in higher-dimensional spaces.
    Generate charts in specified directory.

    Arguments:
        tested_dimensions: list of dimensions to include in experiment
        no_of_points: number of points to fill the hypercube with
        no_of_chosen_vector_pairs: number of vector pairs to choose while testing a dimension
        output_dir: path to a directory in which output charts will be saved (relative to the script location)
    """

    def test_dimension(tested_dimension):
        points = fill_hypercube(tested_dimension, no_of_points)
        angles = [angle_between_vectors(choose_vector(points), choose_vector(points)) for _ in
                  range(no_of_chosen_vector_pairs)]
        return angles

    test_results = [test_dimension(dimension) for dimension in tested_dimensions]

    # generate histograms
    fig, axs = plt.subplots(math.ceil(len(tested_dimensions) / 3), 3, figsize=(9, 9))
    fig.suptitle(f'Histograms of angles between random vector pairs in different dimensions\n')
    fig.tight_layout()
    for i in range(len(tested_dimensions)):
        ax_row = i // 3
        ax_col = i % 3
        ax = axs[ax_row][ax_col]

        ax.hist(test_results[i], 50)
        ax.set_title(f"dimension = {tested_dimensions[i]}")
        ax.set_xlim(0, 3.14)

    # save histograms
    script_dir = os.path.dirname(__file__)
    output_dir_path = os.path.join(script_dir, output_dir)
    if not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)
    plt.savefig(f"{output_dir}/histograms.png")

    # clear plot
    plt.close('all')

    # generate error-bar plot
    avg_angles = [np.average(test_result) for test_result in test_results]
    angle_std = [np.std(test_result) for test_result in test_results]
    plt.errorbar(tested_dimensions, avg_angles, marker='o', linestyle='', yerr=angle_std)
    plt.xscale('log')
    plt.title('Average angle between two random vectors in different dimensions\n')
    plt.xlabel('Dimension')
    plt.ylabel('Average angle [rad]')

    # save error-bar plot
    plt.savefig(f"{output_dir}/avg_angle_value_with_std.png")

    # clear plot
    plt.clf()


def random_distances_experiment(tested_dimensions, no_of_points, no_of_samples, output_dir):
    """
    Check how distance between points change in higher-dimensional spaces, by:
        1. filling a hypercube with points uniformly
        2. choosing a random starting point from the hypercube
        3. choosing two random points from the hypercube and calculating their euclidean distance to the starting point
        4. checking how big the difference between those distances is in comparison to the average distance.
    Saves chart with experiment results in specified directory

    Arguments:
        tested_dimensions: list of dimensions to include in experiment
        no_of_points: number of points to fill the hypercube with
        no_of_samples: how many checks perform for any tested dimension
        output_dir: path to a directory in which output chart will be saved (relative to the script location)
    """

    def test_difference_in_distance_relative_to_avg_distance(point1, point2, point3):
        distance_1_2 = euclidean_distance(point1, point2)
        distance_1_3 = euclidean_distance(point1, point3)
        avg_distance = (distance_1_2 + distance_1_3) / 2
        difference_in_distance = abs(distance_1_3 - distance_1_2)
        return difference_in_distance / avg_distance

    def test_dimension(tested_dimension):
        points = fill_hypercube(tested_dimension, no_of_points)
        return [test_difference_in_distance_relative_to_avg_distance(*[choose_point(points) for _ in range(3)]) for _ in range(no_of_samples)]

    test_results = [test_dimension(dimension) for dimension in tested_dimensions]

    # generate error-bar plot
    avg_test_results = [np.average(test_result) for test_result in test_results]
    test_results_std = [np.std(test_result) for test_result in test_results]
    plt.errorbar(tested_dimensions, avg_test_results, marker='o', linestyle='', yerr=test_results_std)
    plt.xscale('log')
    plt.title('difference in distance relative to average distance in higher dimensions\n')
    plt.xlabel('Dimension')
    plt.ylabel('distance difference/avg distance')

    # save error-bar plot
    script_dir = os.path.dirname(__file__)
    output_dir_path = os.path.join(script_dir, output_dir)
    if not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)
    plt.savefig(f"{output_dir}/distance_difference.png")

    # clear plot
    plt.clf()


if __name__ == '__main__':
    random_angles_experiment(
        tested_dimensions=[2, 3, 5, 10, 25, 50, 100, 500, 1000],
        no_of_points=100000,
        no_of_chosen_vector_pairs=10000,
        output_dir="out/randomAngles"
    )

    random_distances_experiment(
        tested_dimensions=[2, 3, 5, 10, 25, 50, 100, 500, 1000],
        no_of_points=100000,
        no_of_samples=10000,
        output_dir="out/randomDistances"
    )


