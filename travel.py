import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Union

# Set a random seed for reproducibility
random.seed(42)

class Vehicle:
    def __init__(self, capacity: float, depot: Tuple[float, float]):
        self.capacity = capacity
        self.depot = depot
        self.route = [self.depot]
        self.current_load = 0

class VehicleRoutingProblem:
    def __init__(self, depot: Tuple[float, float], customers: Dict[str, Dict[str, float]],
                 vehicle_count: int, vehicle_capacity: float):
        self.depot = depot
        self.customers = customers
        self.vehicle_count = vehicle_count
        self.vehicle_capacity = vehicle_capacity

    def get_coordinates(self, location: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
        if isinstance(location, tuple):
            return location
        return self.customers.get(location, {'coordinates': self.depot})['coordinates']

    def calculate_distance(self, point1: Union[str, Tuple[float, float]],
                           point2: Union[str, Tuple[float, float]]) -> float:
        coords1 = self.get_coordinates(point1)
        coords2 = self.get_coordinates(point2)

        return math.sqrt(
            (coords1[0] - coords2[0])**2 +
            (coords1[1] - coords2[1])**2
        )

    def reset_vehicles(self) -> List[Vehicle]:
        return [Vehicle(self.vehicle_capacity, self.depot) for _ in range(self.vehicle_count)]

    def visualize_route(self, vehicle: Vehicle, title: str, filename: str):
        plt.figure(figsize=(12, 8))
        plt.title(title)

        plt.scatter(self.depot[0], self.depot[1], color='red', s=200, label='Depot')

        for customer, details in self.customers.items():
            plt.scatter(details['coordinates'][0], details['coordinates'][1], color='blue', s=80, label=customer)

        route_coords = [
            self.get_coordinates(stop)
            for stop in vehicle.route
        ]
        xs, ys = zip(*route_coords)
        plt.plot(xs, ys, color='green', linewidth=2, label='Route')

        # Annotate the route with order
        for i, (x, y) in enumerate(route_coords):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def randomized_nearest_neighbor_vrp(self) -> Vehicle:
        vehicle = self.reset_vehicles()[0]
        unvisited_customers = ['New York City, NY', 'Chicago, IL', 'Houston, TX', 'Phoenix, AZ', 'Los Angeles, CA']
        random.shuffle(unvisited_customers)

        while unvisited_customers:
            last_location = vehicle.route[-1] if len(vehicle.route) > 1 else self.depot

            possible_customers = [
                c for c in unvisited_customers
                if (vehicle.current_load + self.customers[c]['demand'] <= vehicle.capacity)
            ]

            if possible_customers:
                possible_customers.sort(
                    key=lambda c: (
                            self.calculate_distance(last_location, self.customers[c]['coordinates']) *
                            random.uniform(0.5, 2.0)
                    )
                )
                closest_customer = possible_customers[0]

                vehicle.route.append(closest_customer)
                vehicle.current_load += self.customers[closest_customer]['demand']
                unvisited_customers.remove(closest_customer)
            else:
                break

        if vehicle.route[-1] != self.depot:
            vehicle.route.append(self.depot)

        return vehicle

    def dynamic_programming_vrp(self) -> Vehicle:
        vehicle = self.reset_vehicles()[0]
        unvisited_customers = sorted(
            ['New York City, NY', 'Chicago, IL', 'Houston, TX', 'Phoenix, AZ', 'Los Angeles, CA'],
            key=lambda c: self.customers[c]['demand'],
            reverse=True
        )

        while unvisited_customers:
            last_location = vehicle.route[-1] if len(vehicle.route) > 1 else self.depot

            best_customer = None
            best_score = float('inf')

            for customer in unvisited_customers:
                if vehicle.current_load + self.customers[customer]['demand'] > vehicle.capacity:
                    continue

                distance = self.calculate_distance(last_location, self.customers[customer]['coordinates'])
                remaining_capacity = vehicle.capacity - vehicle.current_load
                demand_ratio = self.customers[customer]['demand'] / remaining_capacity
                random_factor = random.uniform(0.9, 1.1)
                score = (distance * demand_ratio * random_factor)

                if score < best_score:
                    best_customer = customer
                    best_score = score

            if best_customer is None:
                break

            vehicle.route.append(best_customer)
            vehicle.current_load += self.customers[best_customer]['demand']
            unvisited_customers.remove(best_customer)

        if vehicle.route[-1] != self.depot:
            vehicle.route.append(self.depot)

        return vehicle

# Benchmarking
def plot_benchmarking_results(iterations, nn_runtimes, dp_runtimes):
    log_iterations = np.log10(iterations)
    avg_nn_runtimes = [rt / it for rt, it in zip(nn_runtimes, iterations)]
    avg_dp_runtimes = [rt / it for rt, it in zip(dp_runtimes, iterations)]

    plt.figure(figsize=(12, 8))
    plt.plot(log_iterations, avg_nn_runtimes, label='Nearest Neighbor', marker='o', color='blue')
    plt.plot(log_iterations, avg_dp_runtimes, label='Dynamic Programming', marker='s', color='orange')

    plt.ylim(0.000000, 0.000200)
    plt.xlim(0, 5)

    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=[r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
    plt.xlabel('Number of Iterations (log scale)', fontsize=14)
    plt.ylabel('Average Runtime (seconds)', fontsize=14)
    plt.title('Benchmarking VRP Algorithms', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('benchmarking_comparison.png')
    plt.show()

def benchmark_vrp_algorithms(vrp: VehicleRoutingProblem, iterations: List[int]):
    nn_runtimes = []
    dp_runtimes = []

    for it in iterations:
        nn_runtime = 0
        dp_runtime = 0

        for _ in range(it):
            start_time = time.time()
            vrp.randomized_nearest_neighbor_vrp()
            nn_runtime += time.time() - start_time

            start_time = time.time()
            vrp.dynamic_programming_vrp()
            dp_runtime += time.time() - start_time

        nn_runtimes.append(nn_runtime)
        dp_runtimes.append(dp_runtime)

    return nn_runtimes, dp_runtimes

def main():
    depot = (40.7128, -74.0060)
    customers = {
        'New York City, NY': {'coordinates': (40.7128, -74.0060), 'demand': 0},
        'Chicago, IL': {'coordinates': (41.8781, -87.6298), 'demand': 500},
        'Houston, TX': {'coordinates': (29.7604, -95.3698), 'demand': 400},
        'Phoenix, AZ': {'coordinates': (33.4484, -112.0740), 'demand': 300},
        'Los Angeles, CA': {'coordinates': (34.0522, -118.2437), 'demand': 600}
    }

    vrp = VehicleRoutingProblem(
        depot=depot,
        customers=customers,
        vehicle_count=1,
        vehicle_capacity=1500
    )

    vehicle_nn = vrp.randomized_nearest_neighbor_vrp()
    print("Nearest Neighbor Route:")
    print(vehicle_nn.route)
    print(f"Total Load: {vehicle_nn.current_load}")
    vrp.visualize_route(vehicle_nn, 'Nearest Neighbor Approach', 'nearest_neighbor_route.png')

    vehicle_dp = vrp.dynamic_programming_vrp()
    print("\nDynamic Programming Route:")
    print(vehicle_dp.route)
    print(f"Total Load: {vehicle_dp.current_load}")
    vrp.visualize_route(vehicle_dp, 'Dynamic Programming Approach', 'dynamic_programming_route.png')

    iterations = [10, 100, 1000, 10000, 100000]
    nn_runtimes, dp_runtimes = benchmark_vrp_algorithms(vrp, iterations)
    print("\nBenchmarking Results:")
    print("Iterations:", iterations)
    print("Nearest Neighbor Runtimes:", nn_runtimes)
    print("Dynamic Programming Runtimes:", dp_runtimes)

    plot_benchmarking_results(iterations, nn_runtimes, dp_runtimes)

if __name__ == "__main__":
    main()
