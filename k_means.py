#Author: Michael Fulton
#CS445 Machine Learning -- Winter 2022
#Program 3, K-Means

from re import A
import numpy as np
from matplotlib import pyplot as plt
import random
import math

NUM_C = 3

def random_start(data_points, num_req):
  if len(data_points) < num_req:
    return -1
  return data_points[np.random.choice(data_points.shape[0], num_req, replace=False), :]


def k_assingment(data, centroids):
  A = np.arange(len(data))
  B = np.zeros(len(centroids))
  for i in range(len(data)):
    for j in range(len(centroids)):
      B[j] = calc_dst_squared(data[i], centroids[j])
    A[i] = np.argmin(B)
  return A


def update_centroids_means(A, data, centroids):
  newcentroids = []
  for i in range(len(centroids)):
    newcentroids.append(find_mean(A, data, i))
  return newcentroids


def find_mean(A, data, assignment):
  total = [0, 0]
  num = 0
  for i in range(len(A)):
    if A[i] == assignment:
      total += data[i]
      num += 1
  if num == 0:
    return [0, 0]
  return (total / num)


def calc_dst_squared(point, target):
  return ((point[0] - target[0]) ** 2 + (point[1] - target[1]) ** 2 )



def gen_rnd_color():
  return np.random.rand(3,)

def get_x_y(data, A, assignment):
  x = []
  y = []
  for i in range(len(A)):
    if A[i] == assignment:
      x.append(data[i][0])
      y.append(data[i][1])
  return x,y

def plot_fuzzy(data, W, centroids, clrs):
  x = []
  y = []
  for i in range(len(data)):
    my_color = [0,0,0]
    x = data[i][0]
    y = data[i][1]
    for j in range(len(W[i])):
      my_color += (clrs[j] * W[i][j])
    plt.scatter(x,y, marker='.', color=my_color)

def run_k_means(data, centroids, clrs, num, num_c):
  plt.clf()

  A = k_assingment(data, centroids)

  for i in range(len(centroids)):
    x,y = get_x_y(data, A, i)
    plt.scatter(x,y, marker='.', color=clrs[i])

  cx, cy = np.split(centroids, [-1], axis=1)
  cx = cx.reshape(-1)
  cy = cy.reshape(-1)

  plt.scatter(cx,cy, marker='x', color='red')
  
  centroids = update_centroids_means(A, data, centroids)

  plt.savefig("km/kmeans" + str(num_c) + "_" + str(num) + ".png")


  return centroids


def k_means_full(data, num_c, centroids, clrs, epochs):
  for i in range(epochs):
    centroids = run_k_means(data, centroids, clrs, i, num_c)

def fuzzy_c_full(data, num_c, m, centroids, clrs, epochs):
  for k in range(epochs):
    plt.clf()

    W = fuzzy_membership(data, centroids, m)

    plot_fuzzy(data,W,centroids, clrs)

    cx, cy = np.split(centroids, [-1], axis=1)
    cx = cx.reshape(-1)
    cy = cy.reshape(-1)
    plt.scatter(cx,cy, marker='x', color='red')

    plt.savefig("cm/cmeans" + str(num_c) + "_" + str(k) + ".png")
    centroids = fuzzy_c_update(data,W, centroids, m)


def fuzzy_membership(data, centroids, m):
  exp = (2 / (m - 1))
  W = np.zeros((len(data),len(centroids)))
  for i in range(len(data)):
    for j in range(len(centroids)):
      for k in range(len(centroids)):
        W[i][j] += ((math.sqrt(calc_dst_squared(data[i],centroids[j])) / math.sqrt(calc_dst_squared(data[i],centroids[k]))) ** exp)
      W[i][j] = 1 / W[i][j]
  return W

def fuzzy_c_update(data, W, centroids, m):
  numerator = 0
  denominator = 0

  for i in range(len(centroids)):
    for j in range(len(data)):
      numerator += (W[j][i] ** m) * data[j]
      denominator += (W[j][i]) ** m
    centroids[i] = numerator / denominator
    numerator = 0
    denominator = 0
  
  return centroids

def main():


  #Import the data
  data = np.genfromtxt("clustering_data.csv", delimiter=' ')

  m = 1.2
  epochs = 10

  num_c = 2
  centroids = random_start(data, num_c) + 0.0000001
  clrs = []
  for i in range(num_c):
    clrs.append(gen_rnd_color())
  k_means_full(data, num_c, centroids, clrs, epochs)
  fuzzy_c_full(data, num_c, m, centroids, clrs, epochs)

  num_c = 3
  centroids = random_start(data, num_c) + 0.0000001
  clrs = []
  for i in range(num_c):
    clrs.append(gen_rnd_color())
  k_means_full(data, num_c, centroids, clrs, epochs)
  fuzzy_c_full(data, num_c, m, centroids, clrs, epochs)

  num_c = 5
  centroids = random_start(data, num_c) + 0.0000001
  clrs = []
  for i in range(num_c):
    clrs.append(gen_rnd_color())
  k_means_full(data, num_c, centroids, clrs, epochs)
  fuzzy_c_full(data, num_c, m, centroids, clrs, epochs)

  num_c = 7
  centroids = random_start(data, num_c) + 0.0000001
  clrs = []
  for i in range(num_c):
    clrs.append(gen_rnd_color())
  k_means_full(data, num_c, centroids, clrs, epochs)
  fuzzy_c_full(data, num_c, m, centroids, clrs, epochs)

if __name__ == "__main__":
  main()
