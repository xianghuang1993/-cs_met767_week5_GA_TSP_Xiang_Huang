import matplotlib.pyplot as plt
import numpy as np

class GA_TSP_Problem(object):
    def __init__(self, city_list, city_names, num_grp):
        self.num_grp = num_grp  # the number of groups
        self.city_list = city_list  # the coordinate points of cities
        self.city_names = city_names # the city names
        self.num = len(city_names)  # the number of cities
        self.matrix_distance = self.matrix_distance()  # the matrix to store the distance vector
        self.route = np.array([0] * self.num_grp * self.num).reshape(self.num_grp, self.num) # the route of TSP
        self.distance = [0] * self.num_grp # the total distance of TSP

    def matrix_distance(self): # this is to store the distance between each cities in a matrix
        matrix = np.zeros((self.num,self.num)) # translate array into matrix
        for i in range(self.num):
            for j in range(i+1,self.num):
                matrix[i,j] = np.linalg.norm(self.city_list[i,:]-self.city_list[j,:])
                matrix[j,i] = matrix[i,j]
        return matrix

    def calculate_distance(self, route): # this is to calculate the distance between each cities
        distance = 0
        for i in range(self.num-1):
            distance += self.matrix_distance[route[i],route[i+1]]
        distance += self.matrix_distance[route[-1],route[0]]
        return distance

    def random_route(self): # this is to generate the random route in the beginning of the generation
        rand_ch = np.array(range(self.num))
        for i in range(self.num_grp):
            self.route[i,:]= rand_ch
            self.distance[i] = self.calculate_distance(rand_ch)

    def crossover(self, route, shortest_route): # this is to the crossover function which swaps the two portions of the parents
        r1 = np.random.randint(self.num)
        r2 = np.random.randint(self.num)
        while r2 == r1:
            r2 = np.random.randint(self.num)
        left, right = min(r1, r2), max(r1, r2)
        cross = shortest_route[left:right + 1]
        for i in range(right - left + 1):
            for k in range(self.num):
                if route[k] == cross[i]:
                    route[k:self.num - 1] = route[k + 1:self.num]
                    route[-1] = 0
        route[self.num - right + left - 1:self.num] = cross
        for j in range(self.num):
            if route[j] == 0 and j != 0 and j != self.num:
                route[j] = route[0]
                route[0] = 0
        return route

    def mutation(self,route): # this is to the crossover function which swaps the two cities of the route
        r1 = np.random.randint(self.num)
        r2 = np.random.randint(self.num)
        while r2 == r1 and r1 != 1 and r2 != 1:
            r1 = np.random.randint(self.num)
            r2 = np.random.randint(self.num)
        route[r1],route[r2] = route[r2],route[r1]
        return route

    def print_out_route(self, route): # this is to print out the route in text
        res = str(city_names[0]) + '-->'
        for i in range(1, self.num):
            for j in range (1, self.num):
                if route[i] == j:
                    res += str(city_names[j]) + '-->'
        res += str(city_names[0])
        print(res + '\n')

    def show_route(self): # this is to display the route in diagram using matplotlib
        fig, ax = plt.subplots()
        x = city_list[:, 0]
        y = city_list[:, 1]
        ax.scatter(x, y, linewidths=0.1)
        for i, w in enumerate(range(1, len(city_list) + 1)):
            label = city_names[w-1]
            # for j in range(1,len(city_list) + 1)
            #     if w == j:
            #         label =
            ax.annotate(label, (x[i], y[i]))
        res0 = self.route[0]
        x0 = x[res0]
        y0 = y[res0]
        for i in range(len(city_list) - 1):
            plt.quiver(x0[i], y0[i], x0[i + 1] - x0[i], y0[i + 1] - y0[i], color='r', width=0.005, angles='xy', scale=1,
                       scale_units='xy')
        plt.quiver(x0[-1], y0[-1], x0[0] - x0[-1], y0[0] - y0[-1], color='r', width=0.005, angles='xy', scale=1,
                   scale_units='xy')
        plt.show() # plot out the route which can be saved as a png

def Start_Test(city_list,city_names, max_n=100, num_grp=100):

    Path_short = GA_TSP_Problem(city_list,city_names, num_grp=num_grp)  # create the TSP model
    Path_short.random_route()  # initiate random route
    Path_short.show_route()  # display initial route
    best_P_route = Path_short.route.copy()  # save a copy of the route
    best_P_dis = Path_short.distance.copy()  # save a copy of the distance
    min_index = np.argmin(Path_short.distance)

    best_G_route = Path_short.route[min_index, :]  # save the shortest route as best route
    best_G_distance = Path_short.distance[min_index]  # save the shortest distance

    best_route = [best_G_route]  # save the shortest route as best route
    best_dis = [best_G_distance]  # save the shortest distance

    print('Distance at initial generation: ' + str(Path_short.distance[min_index]))
    print('Route at initial generation')
    Path_short.print_out_route(Path_short.route[min_index, :])

    x_new = Path_short.route.copy()
    for i in range(max_n):  # update the group shortest route
        for j in range(num_grp):
            if Path_short.distance[j] < best_P_dis[j]:
                best_P_dis[j] = Path_short.distance[j]
                best_P_route[j, :] = Path_short.route[j, :]

        min_index = np.argmin(Path_short.distance)  # update the shortest route
        best_G_route = Path_short.route[min_index, :]
        best_G_dis = Path_short.distance[min_index]

        if best_G_dis < best_dis[-1]:  # update the global shortest route
            best_dis.append(best_G_dis)
            best_route.append(best_G_route)
        else:
            best_dis.append(best_dis[-1])
            best_route.append(best_route[-1])

        for j in range(num_grp):
            x_new[j, :] = Path_short.crossover(x_new[j, :], best_P_route[j, :])  # where crossover happens
            dis = Path_short.calculate_distance(x_new[j, :])
            if dis < Path_short.distance[j]:  # compare distance
                Path_short.route[j, :] = x_new[j, :]
                Path_short.distance[j] = dis
            x_new[j, :] = Path_short.crossover(x_new[j, :], best_G_route)  # where crossover happens
            fit = Path_short.calculate_distance(x_new[j, :])
            if dis < Path_short.distance[j]:  # compare distance
                Path_short.route[j, :] = x_new[j, :]
                Path_short.distance[j] = dis
            x_new[j, :] = Path_short.mutation(x_new[j, :])  # where mutation happens
            dis = Path_short.calculate_distance(x_new[j, :])
            if dis <= Path_short.distance[j]:
                Path_short.route[j] = x_new[j, :]
                Path_short.distance[j] = dis

        if (i + 1) % num_grp == 0:
            print('Distance after ' + str(i + 1) + ' generations: '+ str(Path_short.distance[min_index]))
            print('Route after ' + str(i + 1) + ' generations: ')
            Path_short.print_out_route(Path_short.route[min_index, :])

    Path_short.show_route()  # show route
    Path_short.best_route = best_route
    Path_short.best_fit = best_dis
    return Path_short

if __name__ == '__main__':
    # city_list = np.random.rand(15, 2) * 15
    # city_names = ['Boston','London','Mumbai','Shanghai','Tokyo','Miami','New York','Los Angeles','Chicago','Paris','Dallas','Beijing','Singapore','Moscow','Sydney']
    # city_data = {city_names[0]: city_list[0],
    #              city_names[1]: city_list[1],
    #              city_names[2]: city_list[2],
    #              city_names[3]: city_list[3],
    #              city_names[4]: city_list[4],
    #              city_names[5]: city_list[5],
    #              city_names[6]: city_list[6],
    #              city_names[7]: city_list[7],
    #              city_names[8]: city_list[8],
    #              city_names[9]: city_list[9],
    #              city_names[10]: city_list[10],
    #              city_names[11]: city_list[11],
    #              city_names[12]: city_list[12],
    #              city_names[13]: city_list[13],
    #              city_names[14]: city_list[14]}
    city_list = np.random.rand(4, 2) * 4
    city_names = ['Boston','London','Mumbai','Shanghai']
    city_data = {city_names[0]: city_list[0],
                 city_names[1]: city_list[1],
                 city_names[2]: city_list[2],
                 city_names[3]: city_list[3]}
    print('city data: ')
    for k,v in city_data.items():
        print(k+': '+str(v))
    print()
    Start_Test(city_list,city_names)